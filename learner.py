import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from utils import compute_accuracy, tensors_to_device, gradient_update_parameters


class MetaDropoutLearner(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].
    """
    def __init__(self,
                 model,
                 optimizer=None,
                 step_size=0.1,
                 grad_clip=None,
                 num_adaptation_steps=1,
                 device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.num_adaptation_steps = num_adaptation_steps
        self.grad_clip = grad_clip
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks':
            num_tasks,
            'inner_losses':
            np.zeros((self.num_adaptation_steps, num_tasks), dtype=np.float32),
            'outer_losses':
            np.zeros((num_tasks, ), dtype=np.float32),
            'mean_outer_loss':
            0.
        }
        results.update({
            'accuracies_before':
            np.zeros((num_tasks, ), dtype=np.float32),
            'accuracies_after':
            np.zeros((num_tasks, ), dtype=np.float32)
        })
        print("enter outerloss")
        mean_outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            print("inside loop")
            params, adaptation_results = self.adapt(
                train_inputs,
                train_targets,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size,
            )
            results['inner_losses'][:, task_id] = adaptation_results[
                'inner_losses']
            results['accuracies_before'][task_id] = adaptation_results[
                'accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

            results['accuracies_after'][task_id] = compute_accuracy(
                test_logits, test_targets)

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def adapt(self,
              inputs,
              targets,
              num_adaptation_steps=1,
              step_size=0.1,
              mc_steps=1):
        # Get the parameters of the main network
        params = None
        results = {
            'inner_losses': np.zeros((num_adaptation_steps, ),
                                     dtype=np.float32)
        }

        for step in range(num_adaptation_steps):
            # evaluate the expected loss over input-dependent noise distribution with MC approx.
            # if meta-training then we sample once for efficiency.
            # if meta-testing then we sample as much as possible (e.g. 30) for accuracy.
            inner_loss = 0.0
            for _ in range(mc_steps):
                print("before model pass")
                logits = self.model(inputs, params=params)
                print("after model pass")
                inner_loss_per_mc = self.loss_function(logits, targets)
                inner_loss += inner_loss_per_mc

            inner_loss.div_(mc_steps)

            results['inner_losses'][step] = inner_loss.item()

            if step == 0:
                results['accuracy_before'] = compute_accuracy(logits, targets)

            self.model.zero_grad()
            print("before adapt")
            params = gradient_update_parameters(self.model,
                                                inner_loss,
                                                step_size=step_size,
                                                params=params,
                                                first_order=False)
            print("after adapt")

        return params, results

    def train(self, dataloader, max_batches=834, verbose=True, **kwargs):
        """
        max_batches = total_iters/(batch_size*iters_per_batch)
        For omniglot: 40000/(8*6) = 834
        miniimagenet: 60000/(4*6) = 2500
        """
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader,
                                           max_batches=max_batches):
                pbar.update(1)
                postfix = {
                    'loss': '{0:.4f}'.format(results['mean_outer_loss'])
                }
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=None):
        if self.optimizer is None:
            raise RuntimeError(
                'Trying to call `train_iter`, while the '
                'optimizer is `None`. In order to train `{0}`, you must '
                'specify a Pytorch optimizer as the argument of `{0}` '
                '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                print("before outerloss")
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                # Gradient clipping
                print("after backward")
                clip_grad_value_(model.parameters(), self.grad_clip)
                self.optimizer.step()
                print("after grad step")

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader,
                                              max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss'] -
                                    mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after']) -
                                      mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1
