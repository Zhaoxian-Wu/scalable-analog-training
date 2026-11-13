"""One learned positive read gain per frozen S-OUT ResNet matrix layer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sout_resnet_cifar import LogicalAnalogLayer, LogicalMpSGD
from train_analog_resnet18 import build_model_and_optimizer as build_fixed_sout


class LearnableLogicalAnalogLayer(LogicalAnalogLayer):
    """Learn g=exp(theta) on the matrix output, leaving bias and write scale fixed."""

    def __init__(self, fixed_layer: LogicalAnalogLayer) -> None:
        super().__init__(
            fixed_layer.analog_layer,
            float(fixed_layer.logical_scale.item()),
            fixed_layer.backward_scale_position,
        )
        self.log_gain = nn.Parameter(torch.zeros((), device=fixed_layer.logical_scale.device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logical_output = super().forward(inputs)
        gain = self.log_gain.exp()
        bias = getattr(self.analog_layer.analog_module, "bias", None)
        if not isinstance(bias, torch.Tensor):
            return gain * logical_output
        if logical_output.ndim == 2:
            bias_shape = (1, bias.numel())
        elif logical_output.ndim == 4:
            bias_shape = (1, bias.numel(), 1, 1)
        else:
            bias_shape = (1,) * (logical_output.ndim - 1) + (bias.numel(),)
        return gain * logical_output + (1.0 - gain) * bias.view(bias_shape)


def build_model_and_optimizer(args, device):
    model, _ = build_fixed_sout(args, device)
    count = 0
    for full_name, layer in list(model.named_modules()):
        if type(layer) is not LogicalAnalogLayer:
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LearnableLogicalAnalogLayer(layer))
        count += 1
    if count != 21:
        raise RuntimeError(f"Expected 21 S-OUT layers, got {count}")
    optimizer = LogicalMpSGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, disable_context_buffer=True,
    )
    optimizer.regroup_param_groups(model)
    return model, optimizer
