#!/usr/bin/env python3
"""Checkpointed full-ImageNet ResNet-34 run of the selected S-OUT recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from aihwkit.optim.context import AnalogContext


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--wandb-id', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.epochs != 90 or args.batch_size != 64 or args.workers < 0:
        parser.error('This frozen recipe requires 90 epochs and batch size 64')
    return args


def emit(event: str, **fields: object) -> None:
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def group_kind(group: dict) -> str:
    count = sum(isinstance(parameter, AnalogContext) for parameter in group['params'])
    if count not in (0, len(group['params'])):
        raise RuntimeError('Mixed analog/digital optimizer group')
    return 'analog' if count else 'digital'


def verify_groups(optimizer, expected_ratio: float) -> tuple[int, int]:
    groups = {'analog': [], 'digital': []}
    for group in optimizer.param_groups:
        groups[group_kind(group)].append(group)
        if not math.isclose(float(group['momentum']), 0.9, abs_tol=1e-8):
            raise RuntimeError('Unexpected momentum')
        if float(group['weight_decay']) != 0.0:
            raise RuntimeError('Unexpected weight decay')
    if len(groups['analog']) != 37 or not groups['digital']:
        raise RuntimeError(f'Unexpected optimizer groups: {len(groups["analog"])}/{len(groups["digital"])}')
    digital_lr = float(groups['digital'][0]['lr'])
    for group in groups['digital']:
        if not math.isclose(float(group['lr']), digital_lr, rel_tol=1e-7):
            raise RuntimeError('Digital learning rates differ')
    for group in groups['analog']:
        if not math.isclose(float(group['lr']), expected_ratio * digital_lr, rel_tol=1e-6):
            raise RuntimeError('Analog learning-rate ratio differs')
    return len(groups['analog']), len(groups['digital'])


def verify_analog_model(model, optimizer, module) -> None:
    contexts = module.full.analog_contexts(model)
    if len(contexts) != 37:
        raise RuntimeError('Expected 37 analog contexts')
    for name, context in contexts.items():
        layer = model.get_submodule(name)
        scale = float(layer.logical_scale.item())
        layer._scale_float = scale
        context.logical_weight_scale = scale
        if not math.isclose(float(optimizer.state[context]['logical_weight_scale']), scale,
                            rel_tol=1e-6):
            raise RuntimeError(f'Optimizer scale mismatch in {name}')
        tile = layer.analog_layer.analog_module.analog_ctx.analog_tile
        if tile.handle_output_bound:
            raise RuntimeError(f'Output-bound gradient mask reenabled in {name}')
        if type(layer.analog_layer.analog_module.rpu_config.device).__name__ != 'EcRamPresetDevice':
            raise RuntimeError(f'Unexpected device in {name}')


def remap_once(model, optimizer, module, factor: float) -> float:
    errors = []
    with torch.no_grad():
        for name in module.full.analog_contexts(model):
            layer = model.get_submodule(name)
            context = layer.analog_layer.analog_module.analog_ctx
            physical, bias = layer.analog_layer.get_weights(apply_weight_scaling=False)
            old_scale = float(layer.logical_scale.item())
            old_logical = physical.detach().float() * old_scale
            layer.analog_layer.set_weights(physical / factor, bias, apply_weight_scaling=False)
            layer.logical_scale.fill_(old_scale * factor)
            new_scale = float(layer.logical_scale.item())
            layer._scale_float = new_scale
            context.logical_weight_scale = new_scale
            optimizer.state[context]['logical_weight_scale'] = new_scale
            after, _ = layer.analog_layer.get_weights(apply_weight_scaling=False)
            numerator = (after.float() * new_scale - old_logical).square().mean().sqrt()
            denominator = old_logical.square().mean().sqrt().clamp_min(1e-30)
            errors.append(float(numerator / denominator))
    maximum = max(errors)
    if len(errors) != 37 or maximum >= 1e-4:
        raise RuntimeError(f'Remap failed numerical check: {maximum}')
    return maximum


def main() -> None:
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.source / 'imagenet_reference'), str(args.source)]
    import train as reference
    import train_full_sout_ls_wd0_nomask_resnet34 as module

    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')

    source_hash = module.full.source_digest()
    cfg = argparse.Namespace(mode='s_out', tau=1.0, omega=3.0, c_out=6.0, c2_b=3.5,
                             dac_bits=16, adc_bits=16, write_profile='ecram',
                             rpu_construction_seed=1337, lr=0.025, momentum=0.9,
                             weight_decay=0.0)
    model, optimizer = module.full.build_sgd(cfg, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    start_epoch = 0
    verify_groups(optimizer, 1.0)
    if len(module.full.analog_contexts(model)) != 37:
        raise RuntimeError('ResNet-34 S-OUT matrix count differs')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint['mode'] != 's_out' or checkpoint['source_sha256'] != source_hash:
            raise RuntimeError('Source or mode differs from checkpoint')
        if checkpoint.get('recipe') != 'sout_r34_m09_remap333_analoglr01_v1':
            raise RuntimeError('Unexpected S-OUT continuation recipe')
        model.load_state_dict(checkpoint['model'])
        optimizer = module.full.rebuild_optimizer_after_model_load(model, cfg, checkpoint['initial_weights'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch'])
        verify_groups(optimizer, 0.1)
        verify_analog_model(model, optimizer, module)
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        del checkpoint
        torch.cuda.empty_cache()
        emit('resume', next_epoch=start_epoch + 1, source_sha256=source_hash,
             checkpoint=str(args.resume), gpu_allocated_mib=round(torch.cuda.memory_allocated() / 2**20),
             gpu_reserved_mib=round(torch.cuda.memory_reserved() / 2**20))
    if start_epoch >= args.epochs:
        raise RuntimeError(f'Checkpoint already at epoch {start_epoch}')
    if args.preflight_only:
        emit('preflight_complete', analog_groups=37,
             analog_lr=0.025 if start_epoch == 0 else next(float(g['lr']) for g in optimizer.param_groups if group_kind(g) == 'analog'),
             digital_lr=0.025 if start_epoch == 0 else next(float(g['lr']) for g in optimizer.param_groups if group_kind(g) == 'digital'),
             output_bound_gradient_mask='disabled')
        return

    data_args = reference.get_args_parser().parse_args(['--data-path', str(args.data_path)])
    data_args.distributed = False
    train_set, val_set, _, _ = reference.load_data(
        str(args.data_path / 'train'), str(args.data_path / 'val'), data_args)
    if len(train_set) != 1_281_167 or len(val_set) != 50_000:
        raise RuntimeError('ImageNet dataset incomplete')
    if len(train_set.classes) != 1000 or train_set.classes != val_set.classes:
        raise RuntimeError('ImageNet classes differ')
    train_generator = torch.Generator()
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              generator=train_generator, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    config = {
        'recipe': 'sout_r34_m09_remap333_analoglr01_v1', 'dataset': 'ImageNet-1K',
        'architecture': 'resnet34', 'warmup_epoch': 1,
        'source_sha256': source_hash, 'train_samples': len(train_set),
        'val_samples': len(val_set), 'epochs': args.epochs, 'batch_size': 64,
        'digital_lr': 0.025, 'analog_lr_after_warmup': 0.0025,
        'analog_lr_first_epoch': 0.025, 'momentum': 0.9,
        'weight_decay': 0.0, 'remap_factor': 10.0 / 3.0,
        'write_device': 'EcRamPresetDevice', 'dac_bits': 16, 'adc_bits': 16,
        'output_bound_gradient_mask': 'disabled',
        'amp': False, 'seed': 1337, 'checkpoint_interval_epochs': 1,
    }
    emit('config', **config)
    os.environ['WANDB_API_KEY'] = Path('/home/zw868/.config/wandb/imagenet.key').read_text().strip()
    import wandb
    wandb_dir = args.output_dir / 'wandb'
    wandb_dir.mkdir(exist_ok=True)
    run = wandb.init(entity='analog-transformer', project='ImageNet', id=args.wandb_id,
                     name='ResNet-34 S-OUT m0.9 remap3.33 analogLR0.0025 full ImageNet',
                     group='r34_full_imagenet_20261026',
                     job_type='training', tags=['full_imagenet', 'resnet34', 's_out',
                                                'momentum_0.9', 'remap_3.33', 'analog_lr_0.0025'],
                     config=config, dir=str(wandb_dir), resume='allow',
                     settings=wandb.Settings(init_timeout=120))
    (args.output_dir / 'wandb_url.txt').write_text(run.url + '\n')
    emit('wandb', url=run.url)
    criterion = nn.CrossEntropyLoss()
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            verify_groups(optimizer, 1.0 if epoch == 1 else 0.1)
            train_generator.manual_seed(1337 + epoch)
            started = time.monotonic()
            train_metrics = module.full.measure(model, train_loader, criterion, device,
                                                optimizer, log_interval=500)
            val_metrics = module.full.measure(model, val_loader, criterion, device)
            scheduler.step()
            if epoch == 1:
                remap_error = remap_once(model, optimizer, module, 10.0 / 3.0)
                for group in optimizer.param_groups:
                    if group_kind(group) == 'analog':
                        group['lr'] *= 0.1
                scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
                emit('intervention', after_epoch=1, remap_factor=10.0 / 3.0,
                     analog_lr_factor=0.1,
                     maximum_logical_weight_relative_rms_error=remap_error)
            verify_groups(optimizer, 0.1)
            verify_analog_model(model, optimizer, module)
            item = {
                'event': 'epoch', 'epoch': epoch, 'train': train_metrics,
                'val': val_metrics, 'seconds_total': round(time.monotonic() - started, 1),
                'digital_lr_next_epoch': next(float(group['lr']) for group in optimizer.param_groups
                                               if group_kind(group) == 'digital'),
                'analog_lr_next_epoch': next(float(group['lr']) for group in optimizer.param_groups
                                              if group_kind(group) == 'analog'),
                'gains': module.full.learned_gains(model),
            }
            emit('epoch', **{key: value for key, value in item.items() if key != 'event'})
            saved = {
                'mode': 's_out', 'source_sha256': source_hash, 'epoch': epoch,
                'recipe': 'sout_r34_m09_remap333_analoglr01_v1',
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'initial_weights': module.full.initial_weights(model),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(),
            }
            temporary = args.output_dir / 'checkpoint.pth.tmp'
            torch.save(saved, temporary)
            temporary.replace(args.output_dir / 'checkpoint.pth')
            run.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/top1_pct': 100 * train_metrics['top1'],
                'train/top5_pct': 100 * train_metrics['top5'],
                'train/seconds': train_metrics['seconds'],
                'val/loss': val_metrics['loss'],
                'val/top1_pct': 100 * val_metrics['top1'],
                'val/top5_pct': 100 * val_metrics['top5'],
                'val/seconds': val_metrics['seconds'],
                'lr/digital_next_epoch': item['digital_lr_next_epoch'],
                'lr/analog_next_epoch': item['analog_lr_next_epoch'],
                **{f'gain/{name}': value for name, value in item['gains'].items()},
            }, step=epoch, commit=True)
            run.summary['epochs_recorded'] = epoch
            run.summary['best_val_top1_pct'] = max(
                float(run.summary.get('best_val_top1_pct', 0.0)),
                100 * val_metrics['top1'])
            run.summary['latest_val_top1_pct'] = 100 * val_metrics['top1']
            emit('checkpoint', epoch=epoch, path=str(args.output_dir / 'checkpoint.pth'))
        (args.output_dir / 'complete.txt').write_text(f'epochs={args.epochs}\n')
        run.summary['status'] = 'complete'
    finally:
        run.finish()


if __name__ == '__main__':
    main()
