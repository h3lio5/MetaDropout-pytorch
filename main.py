import torch
import torch.nn.functional as F
import argparse
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
import higher
from model import MetaDropout
from temp import *
from tqdm import tqdm
import json


def get_args():
    parser = argparse.ArgumentParser('Meta Dropout')

    parser.add_argument('--folder',
                        type=str,
                        default='./data',
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument(
        '--num_shots',
        type=int,
        default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument(
        '--num_ways',
        type=int,
        default=20,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument(
        '--maml',
        type=bool,
        default=False,
        help=
        'Specify whether to train MAML or MetaDropout (default: MetaDropout)')
    parser.add_argument('--dataset',
                        type=str,
                        default='omniglot',
                        help='Dataset to train on (default: omniglot')
    parser.add_argument(
        '--savedir',
        type=str,
        default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument(
        '--num_iters',
        type=int,
        default=40000,
        help='Number of batches the model is trained over (default: 40000).')
    parser.add_argument(
        '--num_test_iters',
        type=int,
        default=1000,
        help='Number of batches the model is tested over (default: 1000).')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Specify whether you want to train/evaluate the model')
    parser.add_argument('--num_query',
                        type=int,
                        default=15,
                        help='Number of query samples')
    parser.add_argument('--num_adapt_steps',
                        type=int,
                        default=5,
                        help='Number of inner loop steps')
    parser.add_argument('--mc_steps',
                        type=int,
                        default=1,
                        help='Number of Monte Carlo estimation steps')
    parser.add_argument('--inner_lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the inner optimization loop')
    parser.add_argument('--meta_lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the outer optimization loop')
    parser.add_argument('--grad_clip',
                        type=float,
                        default=3.0,
                        help='Gradient clipping value')

    args = parser.parse_args()
    return args


def get_dataloader(args):

    if args.dataset == 'omniglot':
        dataset = omniglot(args.folder,
                           shots=args.num_shots,
                           ways=args.num_ways,
                           shuffle=True,
                           test_shots=args.num_query,
                           meta_split=args.mode,
                           download=args.download)
    else:
        dataset = miniimagenet(args.folder,
                               shots=args.num_shots,
                               ways=args.num_ways,
                               shuffle=True,
                               test_shots=args.num_query,
                               meta_split=args.mode,
                               download=args.download)

    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers,
                                     pin_memory=True)
    return dataloader


def train(args, model, dataloader):

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), args.meta_lr)
    # optimize over main network parameters
    inner_optimizer = torch.optim.SGD([{
        'params': model.get_main_net_params(),
        'lr': args.inner_lr
    }, {
        'params': model.get_other_params(),
        'lr': 0.0
    }])
    # import pdb
    # pdb.set_trace()
    metalearner = MetaDropoutLearner(model,
                                     meta_optimizer,
                                     inner_optimizer,
                                     num_adaptation_steps=args.num_adapt_steps,
                                     mc_steps=args.mc_steps,
                                     step_size=args.inner_lr,
                                     grad_clip=args.grad_clip,
                                     device=args.device)

    # Training loop
    metalearner.train(dataloader,
                      max_batches=args.num_iters // args.batch_size,
                      verbose=args.verbose,
                      desc='Training',
                      leave=False)
    # Save best mode
    with open(args.savedir + '/model.pth', 'wb') as f:
        torch.save(model.state_dict(), f)


def test(args, model, dataloader):

    model.to(device=args.device)
    model.eval()
    # optimize over main network parameters
    inner_optimizer = torch.optim.SGD([{
        'params': model.get_main_net_params(),
        'lr': args.inner_lr
    }, {
        'params': model.get_other_params(),
        'lr': 0.0
    }])
    metalearner = MetaDropoutLearner(model,
                                     inner_optimizer=inner_optimizer,
                                     num_adaptation_steps=args.num_adapt_steps,
                                     mc_steps=args.mc_steps,
                                     step_size=args.inner_lr,
                                     device=args.device)
    results = metalearner.evaluate(dataloader,
                                   max_batches=args.num_test_iters //
                                   args.batch_size,
                                   verbose=args.verbose,
                                   desc='Test',
                                   num_test_iters=args.num_test_iters,
                                   leave=False)
    print(
        "Accuracy: {results['accuracy_mean']} +- {results['accuracy_95_conf']} "
    )
    with open(args.savedir + '/results.json', 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':

    # Get the command line arguments
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the dataloader
    dataloader = get_dataloader(args)
    # Load the model
    model = MetaDropout(args)

    if args.mode == 'train':

        train(args, model, dataloader)
    else:
        with open(args.savedir + '/model.pth', 'rb') as f:
            model.load_state_dict(torch.load(f))
        test(args, model, dataloader)
