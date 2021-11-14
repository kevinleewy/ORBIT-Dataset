# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class LSLR_SGD():
    
    def __init__(self, params, lrs, maximize: bool = False):
        
        assert params.keys() == lrs.keys(), 'params and lrs must have identical set of keys'

        self.params = params
        self.lrs = lrs
        self.maximize = maximize

    # @torch.no_grad()
    def step(self, step_num):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        updates_dict = {}

        for key, (param, scale_factor) in self.params.items():

            d_p = torch.clone(param.grad).detach()
            lr = self.lrs[key][step_num]
            alpha = (lr if self.maximize else -lr) * scale_factor

            with torch.no_grad():
                param.add_(d_p, alpha=alpha.detach())

            # with torch.enable_grad():
            lslr_loss = (param + d_p * alpha).sum()
            updates_dict[key] = lslr_loss

            print('lslr_loss', lslr_loss.size(), lslr_loss)
            print('before optim backward:', key, step_num, lr, lr.grad)
            lslr_loss.backward()
            print('after optim backward:', key, step_num, lr, lr.grad)

        return updates_dict

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor` s to zero.
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for _, (p, _) in self.params.items():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()


optimizers = {
        'adam': torch.optim.Adam,
        'sgd' : torch.optim.SGD,
        'lslr': LSLR_SGD
        }

def cross_entropy(test_logits, test_labels, reduction='mean'):
    return F.cross_entropy(test_logits, test_labels, reduction=reduction)
    
def init_optimizer(model, lr, optimizer_type='adam', extractor_scale_factor=1.0, additional_params=None):
    
    optimizer_fn = optimizers[optimizer_type]

    if optimizer_type == 'lslr':

        params = {}
        for k, v in model.named_parameters():
            if k.startswith('feature_extractor'):
                params[k.replace('.','-')] = (v, extractor_scale_factor)
            else:
                params[k.replace('.','-')] = (v, 1.0)
        
        optimizer = optimizer_fn(params=params, lrs=lr)
    
    else:
    
        feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params, model.parameters())

        params_list = [
                        {'params': base_params },
                        {'params': model.feature_extractor.parameters(), 'lr': lr * extractor_scale_factor}
                    ]

        if additional_params:
            params_list.append({'params': additional_params })

        optimizer = optimizer_fn(params_list, lr=lr)

    optimizer.zero_grad()
    return optimizer

def init_inner_lr_optimizer(inner_lr, outer_lr, optimizer_type='adam'):
    
    optimizer_fn = optimizers[optimizer_type]
    optimizer = optimizer_fn(inner_lr, lr=outer_lr)
    optimizer.zero_grad()
    return optimizer


