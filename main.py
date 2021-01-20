import torch
import torch.nn.functional as F
import argparse
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters
from model import MetaDropout
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
        default=5,
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
        default=60000,
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
    parser.add_argument(
        '--train',
        type=bool,
        default=True,
        help='Specify whether you want to train/evaluate the model')
    parser.add_argument('--inner-steps',
                        type=int,
                        default=5,
                        help='Number of inner loop steps')
    parser.add_argument('--inner-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the inner optimization loop')
    parser.add_argument('--outer-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate of the outer optimization loop')
    args = parser.parse_args()
    return args


def get_dataloader(args):

    if args.dataset == 'omniglot':
        dataset = omniglot(args.folder,
                           shots=args.num_shots,
                           ways=args.num_ways,
                           shuffle=True,
                           test_shots=15,
                           meta_train=args.train,
                           download=args.download)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet(args.folder,
                               shots=args.num_shots,
                               ways=args.num_ways,
                               shuffle=True,
                               test_shots=15,
                               meta_train=args.train,
                               download=args.download)
    else:
        raise ValueError(
            f'Does not support the dataset: {args.dataset}. Datasets supported: omniglot, miniimagenet'
        )

    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)
    return dataloader


def train(args, dataloader, model):

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), args.outer_lr)
    # Training loop
    with tqdm(dataloader, total=args.num_iters) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            print(train_inputs.size(), test_inputs.size())
            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(
                               zip(train_inputs, train_targets, test_inputs,
                                   test_targets)):
                print(train_input.size(), test_inputs.size())
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                # Get the main network parameters
                params = model.get_main_net_params()
                # Get the gradients of the main network
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    params,
                                                    step_size=args.inner_lr,
                                                    first_order=False)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                # with torch.no_grad():
                #     accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)

            outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_iters:
                break


if __name__ == '__main__':

    # Get the command line arguments
    args = get_args()
    args.device = torch.device(
        'cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    # Load the model
    model = MetaDropout(args)
    # load the dataloader
    dataloader = get_dataloader(args)

    if args.train:
        train(args, dataloader, model)
