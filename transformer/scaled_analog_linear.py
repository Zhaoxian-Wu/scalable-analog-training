"""S-OUT reads with native mapping and digital logical weight gradients.

Only the identity-managed dense read and L2-normalized vocabulary-head backward
read used by fig:scaling are included. Optimizer rules live in MixedPrecisionOptim.
"""
import hashlib
import math

import torch
from torch import nn
from torch.autograd import Function

from mapping_scale import MappingScaleMixin


def derive_rpu_construction_seed(base_seed, module_name):
    payload = f'aimc-rpu-v1\0{base_seed}\0{module_name}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big') or 1


def logical_weight_std(input_width, n_layer, residual_projection=False):
    std = 0.02 * math.sqrt(768 / input_width)
    return std / math.sqrt(2 * n_layer) if residual_projection else std


def fixed_output_scale(n_layer, input_width, residual_projection, omega, conductance_bound):
    return omega * logical_weight_std(input_width, n_layer, residual_projection) / conductance_bound


def _bind_logical_scale(context, scale):
    context.analog_tile.set_mapping_scales(torch.tensor(
        [scale], dtype=context.dtype, device=context.device))
    context.mapped_gradient_units = True


@torch.no_grad()
def accumulate_logical_gradient(context, inputs, errors):
    context.accumulate_digital_gradient(inputs, errors)


class _ScalingMVM(Function):
    @staticmethod
    def forward(ctx, context, inputs, tile, normalize_backward):
        ctx.context = context
        ctx.tile = tile
        ctx.normalize_backward = normalize_backward
        ctx.save_for_backward(inputs)
        return tile.joint_forward(inputs, is_test=not tile.training, ctx=None)

    @staticmethod
    def backward(ctx, errors):
        inputs, = ctx.saved_tensors
        # Accumulate the original logical pair before read normalization.
        accumulate_logical_gradient(ctx.context, inputs, errors)
        if ctx.normalize_backward:
            energy = errors.float().square().sum(dim=-1, keepdim=True)
            active = energy > 0
            scale = torch.where(active, energy.sqrt(), torch.ones_like(energy)).to(errors.dtype)
            bare = torch.where(active, errors / scale, torch.zeros_like(errors))
            grad_inputs = scale * ctx.tile.backward(bare, ctx=None)
            grad_inputs = torch.where(active, grad_inputs, torch.zeros_like(grad_inputs))
        else:
            grad_inputs = ctx.tile.backward(errors, ctx=None)
        return None, grad_inputs, None, None


class LogicalAnalogLinear(MappingScaleMixin, nn.Module):
    """Use native mapping scales for both reads and quantized logical writes."""
    def __init__(self, analog_linear, output_scale, normalize_backward=False):
        super().__init__()
        self.analog_linear = analog_linear
        self.normalize_backward = normalize_backward
        _bind_logical_scale(analog_linear.analog_module.analog_ctx, output_scale)
        self._init_mapping_checkpoint()

    @property
    def in_features(self):
        return self.analog_linear.in_features

    @property
    def out_features(self):
        return self.analog_linear.out_features

    def forward(self, inputs):
        tile = self.analog_linear.analog_module
        return _ScalingMVM.apply(tile.analog_ctx, inputs, tile, self.normalize_backward)
