"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license

This code was based on the file normalization_layers.py (https://github.com/cambridge-mlg/cnaps/blob/master/src/normalization_layers.py)
from the cambridge-mlg/cnaps library (https://github.com/cambridge-mlg/cnaps).

The original license is included below:

Copyright (c) 2019 John Bronskill, Jonathan Gordon, and James Requeima.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import extract_top_level_dict

class BatchNorm2d(nn.BatchNorm2d):

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
        **kwargs
    ):

        if kwargs and kwargs['use_per_step_bn_statistics'] and kwargs['num_grad_steps']:
            num_features = (kwargs['num_grad_steps'], num_features)

        super(BatchNorm2d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=None,
            dtype=None
        )

        # print(self.num_features)
        # print(self.weight.size())
        # print(self.bias.size())

        if kwargs['use_per_step_bn_statistics']:
            self.use_per_step_bn_statistics = kwargs['use_per_step_bn_statistics']
        if kwargs['num_grad_steps']:
            self.num_grad_steps = kwargs['num_grad_steps']

    def forward(self, input, params=None, step_num=0):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        # Per-Step Batch Normalization Weights and Biases (BNWB):
        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[step_num]
            running_var = self.running_var[step_num]
            weight = weight[step_num]
            bias = bias[step_num]
        else:
            running_mean = self.running_mean
            running_var = self.running_var
        
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            running_mean
            if not self.training or self.track_running_stats
            else None,
            running_var if not self.training or self.track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class NormalizationLayer(nn.BatchNorm2d):
    """
    Base class for all normalization layers.
    Derives from nn.BatchNorm2d to maintain compatibility with the pre-trained resnet-18.
    """
    def __init__(self, num_features):
        """
        Initialize the class.
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(NormalizationLayer, self).__init__(
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)

    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        pass  # always override this method

    def _normalize(self, x, mean, var):
        """
        Normalize activations.
        :param x: input activations
        :param mean: mean used to normalize
        :param var: var used to normalize
        :return: normalized activations
        """
        return (self.weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)) + self.bias.view(1, -1, 1, 1)

    @staticmethod
    def _compute_batch_moments(x):
        """
        Compute conventional batch mean and variance.
        :param x: input activations
        :return: batch mean, batch variance
        """
        return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)

    @staticmethod
    def _compute_instance_moments(x):
        """
        Compute instance mean and variance.
        :param x: input activations
        :return: instance mean, instance variance
        """
        return torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)

    @staticmethod
    def _compute_layer_moments(x):
        """
        Compute layer mean and variance.
        :param x: input activations
        :return: layer mean, layer variance
        """
        return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)

    @staticmethod
    def _compute_pooled_moments(x, alpha, batch_mean, batch_var, augment_moment_fn):
        """
        Combine batch moments with augment moments using blend factor alpha.
        :param x: input activations
        :param alpha: moment blend factor
        :param batch_mean: standard batch mean
        :param batch_var: standard batch variance
        :param augment_moment_fn: function to compute augment moments
        :return: pooled mean, pooled variance
        """
        augment_mean, augment_var = augment_moment_fn(x)
        pooled_mean = alpha * batch_mean + (1.0 - alpha) * augment_mean
        batch_mean_diff = batch_mean - pooled_mean
        augment_mean_diff = augment_mean - pooled_mean
        pooled_var = alpha * (batch_var + (batch_mean_diff * batch_mean_diff)) +\
                     (1.0 - alpha) * (augment_var + (augment_mean_diff * augment_mean_diff))
        return pooled_mean, pooled_var


class TaskNormBase(NormalizationLayer):
    """TaskNorm base class."""
    def __init__(self, num_features):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormBase, self).__init__(num_features)
        # Variables to store the context moments to use for normalizing the target.
        self.context_batch_mean = torch.zeros((1, num_features, 1, 1), requires_grad=True)
        self.context_batch_var = torch.ones((1, num_features, 1, 1), requires_grad=True)
        # Variable to save the context size.
        self.context_size = 0
        self.sigmoid = torch.nn.Sigmoid()

    def register_extra_weights(self):
        """
        The parameters here get registered after initialization because the pre-trained resnet model does not have
        these parameters and would fail to load if these were declared at initialization.
        :return: Nothing
        """
        device = self.weight.device
        # Initialize and register the learned parameters 'a' (SCALE) and 'b' (OFFSET)
        # for calculating alpha as a function of context size.
        a = torch.Tensor([0.0]).to(device)
        b = torch.Tensor([0.0]).to(device)
        self.register_parameter(name='a', param=torch.nn.Parameter(a, requires_grad=True))
        self.register_parameter(name='b', param=torch.nn.Parameter(b, requires_grad=True))

    def _get_augment_moment_fn(self):
        """
        Provides the function to compute augment moemnts.
        :return: function to compute augment moments.
        """
        pass  # always override this function

    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        if self.training:  # compute the pooled moments for the context and save off the moments and context size
            alpha = self.sigmoid(self.a * (x.size())[0] + self.b)  # compute alpha with context size
            batch_mean, batch_var = self._compute_batch_moments(x)
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, batch_mean, batch_var,
                                                                   self._get_augment_moment_fn())
            self.context_batch_mean = batch_mean
            self.context_batch_var = batch_var
            self.context_size = (x.size())[0]
        else:  # compute the pooled moments for the target
            alpha = self.sigmoid(self.a * self.context_size + self.b)  # compute alpha with saved context size
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, self.context_batch_mean,
                                                                   self.context_batch_var,
                                                                   self._get_augment_moment_fn())

        return self._normalize(x, pooled_mean, pooled_var)  # normalize


class TaskNorm(TaskNormBase):
    """
    TaskNorm normalization layer. Just need to override the augment moment function with 'instance'.
    """
    def __init__(self, num_features, **kwargs):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNorm, self).__init__(num_features)

    def _get_augment_moment_fn(self):
        """
        Override the base class to get the function to compute instance moments.
        :return: function to compute instance moments
        """
        return self._compute_instance_moments

def get_normalisation_layer(batch_norm):
    if batch_norm == 'task_norm':
        return TaskNorm
    else:
        return BatchNorm2d
