#!/usr/bin/env python3
"""Matched S-OUT ImageNet diagnostic with AIHWKit output-bound gradient masking disabled."""

from __future__ import annotations

import train_full_sout_sgd_learnable_scale_resnet50 as full
from sout_resnet_cifar import LogicalAnalogLayer

_original_arguments = full.arguments
_original_build = full.build_sgd
_original_rebuild = full.rebuild_optimizer_after_model_load


def disable_bound_gradient_mask(model):
    count = 0
    for module in model.modules():
        if not isinstance(module, LogicalAnalogLayer):
            continue
        tile = module.analog_layer.analog_module.analog_ctx.analog_tile
        tile.handle_output_bound = False
        if tile.handle_output_bound:
            raise RuntimeError('Failed to disable AIHWKit output-bound gradient mask')
        count += 1
    if count != 54:
        raise RuntimeError(f'Expected 54 analog layers, found {count}')
    return count


def arguments():
    args = _original_arguments()
    if args.mode != 's_out' or args.momentum != 0.9 or args.weight_decay != 0.0:
        raise ValueError('This diagnostic requires S-OUT, momentum 0.9, weight_decay 0')
    args.output_bound_gradient_mask = 'disabled'
    return args


def build_sgd(args, device):
    model, optimizer = _original_build(args, device)
    disable_bound_gradient_mask(model)
    return model, optimizer


def rebuild_optimizer_after_model_load(model, args, saved_initial):
    optimizer = _original_rebuild(model, args, saved_initial)
    disable_bound_gradient_mask(model)
    return optimizer


full.arguments = arguments
full.build_sgd = build_sgd
full.rebuild_optimizer_after_model_load = rebuild_optimizer_after_model_load

if __name__ == '__main__':
    full.main()
