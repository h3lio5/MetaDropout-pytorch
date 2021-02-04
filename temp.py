import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from utils import compute_accuracy, tensors_to_device, gradient_update_parameters
import higher


class MetaDropoutLearner(object):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].
    """
    def __init__(self,
                 model,
                 meta_optimizer=None,
                 inner_optimizer=None,
                 step_size=0.1,
                 grad_clip=None,
                 num_adaptation_steps=1,
                 mc_steps=1,
                 device=None):
        self.model = model.to(device=device)
        self.meta_optimizer = meta_optimizer
        self.inner_optimizer = inner_optimizer
        self.step_size = step_size
        self.num_adaptation_steps = num_adaptation_steps
        self.mc_steps = mc_steps
        self.grad_clip = grad_clip
        self.loss_function = nn.CrossEntropyLoss()
        self.device = device

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'outer_losses': np.zeros((num_tasks, ), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        results.update({
            'accuracies_before':
            np.zeros((num_tasks, ), dtype=np.float32),
            'accuracies_after':
            np.zeros((num_tasks, ), dtype=np.float32)
        })
        mean_outer_loss = 0.0
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            with higher.innerloop_ctx(self.model,
                                      self.inner_optimizer) as (fmodel,
                                                                diffopt):
                for step in range(self.num_adaptation_steps):
                    # evaluate the expected loss over input-dependent noise distribution with MC approx.
                    # if meta-training then we sample once for efficiency.
                    # if meta-testing then we sample as much as possible (e.g. 30) for accuracy.
                    inner_loss = torch.tensor(0., device=self.device)
                    for i in range(self.mc_steps):
                        logits = fmodel(train_inputs)
                        inner_loss_per_mc = self.loss_function(
                            logits, train_targets)
                        inner_loss += inner_loss_per_mc

                    inner_loss.div_(self.mc_steps)

                    if step == 0:
                        results['accuracy_before'] = compute_accuracy(
                            logits, train_targets)
                    diffopt.step(inner_loss)
                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                with torch.set_grad_enabled(self.model.training):
                    test_logits = self.model(test_inputs)
                    outer_loss = self.loss_function(test_logits, test_targets)
                    if self.model.training:
                        outer_loss.backward()
                    results['outer_losses'][task_id] = outer_loss.item()
                    mean_outer_loss += outer_loss.item()

            results['accuracies_after'][task_id] = compute_accuracy(
                test_logits, test_targets)

        mean_outer_loss = mean_outer_loss / num_tasks
        results['mean_outer_loss'] = mean_outer_loss

        return results

    def train(self, dataloader, max_batches=40000, verbose=True, **kwargs):
        """
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
        """
        """
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                self.meta_optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                results = self.get_outer_loss(batch)
                yield results
                # Gradient clipping
                clip_grad_value_(self.model.parameters(), self.grad_clip)
                self.meta_optimizer.step()

                num_batches += 1

    def evaluate(self,
                 dataloader,
                 max_batches=500,
                 num_test_iters=1000,
                 verbose=True,
                 **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        accuracies = []
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
                    accuracies.extend(results['accuracies_after'])
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}

        if 'accuracies_after' in results:
            mean_results['accuracy_mean'] = mean_accuracy

        accuracy_95_conf = 1.96 * np.std(accuracies) / float(
            np.sqrt(num_test_iters))

        mean_results['accuracy_95_conf'] = accuracy_95_conf

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                results = self.get_outer_loss(batch)
                yield results

                num_batches += 1
