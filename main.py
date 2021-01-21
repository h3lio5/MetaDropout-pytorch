import torch
import torch.nn.functional as F
import argparse
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from model import MetaDropout
from learner import *
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser('Meta Dropout')

    parser.add_argument('--folder',
                        type=str,
                        default='./data',
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument(
        '--num-shots',
        type=int,
        default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument(
        '--num-ways',
        type=int,
        default=20,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument(
        '--maml',
        type=bool,
        default=False,
        help=
        'Specify whether to train MAML or MetaDropout (default: MetaDropout)')
    parser.add_argument(
        '--step-size',
        type=float,
        default=0.4,
        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--dataset',
                        type=str,
                        default='omniglot',
                        help='Dataset to train on (default: omniglot')
    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument(
        '--num-iters',
        type=int,
        default=40000,
        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda',
                        action='store_true',
                        help='Use CUDA if available.')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Specify whether you want to train/evaluate the model')
    parser.add_argument('--num_steps',
                        type=int,
                        default=5,
                        help='Number of inner loop steps')
    parser.add_argument('--inner-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the inner optimization loop')
    parser.add_argument('--meta-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the outer optimization loop')
    parser.add_argument('--grad-clip',
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
                           test_shots=15,
                           meta_train=args.mode == 'train',
                           download=args.download)
    else:
        dataset = miniimagenet(args.folder,
                               shots=args.num_shots,
                               ways=args.num_ways,
                               shuffle=True,
                               test_shots=15,
                               meta_train=args.mode == 'train',
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
    metalearner = MetaDropoutLearner(model,
                                     meta_optimizer,
                                     num_adaptation_steps=args.num_steps,
                                     step_size=args.step_size,
                                     grad_clip=args.grad_clip,
                                     device=args.device)

    # Training loop
    metalearner.train(dataloader,
                      max_batches=834,
                      verbose=args.verbose,
                      desc='Training',
                      leave=False)
    # Save best mode
    with open(args.model_path, 'wb') as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':

    # Get the command line arguments
    args = get_args()
    args.device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    # Load the model
    model = MetaDropout(args)
    # load the dataloader
    dataloader = get_dataloader(args)

    if args.mode == 'train':
        train(args, model, dataloader)
