#!/usr/bin/env python3
"""Train the digital or S-OUT AdamW path of the fig:scaling comparison."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset_paths import resolve_dataset_dir
from model import GPT, GPTConfig
from scaling_config import ScalingConfig, config_dict, learning_rate_at, load_config, should_evaluate


def build_model(config):
    shape = config.shape
    model = GPT(GPTConfig(n_layer=shape['n_layer'], n_head=shape['n_head'],
                          n_embd=shape['n_embd'], block_size=config.block_size,
                          vocab_size=config.vocab_size, bias=False, dropout=0.0,
                          standard_normal_init=config.mode == 'analog',
                          analog_init_std=config.conductance_bound / config.omega))
    if model.get_num_params() != config.budget_parameters:
        raise RuntimeError('Model parameter count differs from the scaling budget.')
    if config.mode == 'analog':
        from analog_model import convert_scaling_model
        # The position lookup stays digital with nanoGPT initialization.
        torch.nn.init.normal_(model.transformer.wpe.weight, mean=0.0, std=0.02)
        model = convert_scaling_model(model, config)
    return model.to(config.device)


def build_optimizer(model, config):
    kwargs = dict(lr=config.learning_rate, betas=(config.beta1, config.beta2),
                  eps=config.adam_eps, weight_decay=config.weight_decay)
    if config.mode == 'digital':
        return torch.optim.AdamW(model.parameters(), **kwargs)
    from MixedPrecisionOptim.AdamW import MpAdamW
    optimizer = MpAdamW(model.parameters(), **kwargs, fused=False)
    optimizer.regroup_param_groups(model)
    return optimizer


def autocast(config):
    if config.mode == 'digital' and config.device.startswith('cuda'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


class TokenBatches:
    def __init__(self, config, rank=0):
        self.config = config
        directory = Path(resolve_dataset_dir(config.dataset, dataset_dir=config.dataset_dir,
                         dataset_root=config.dataset_root))
        with (directory / 'meta.pkl').open('rb') as handle:
            metadata = pickle.load(handle)
        if metadata['vocab_size'] != config.vocab_size:
            raise ValueError(f"Expected vocab_size={config.vocab_size}; got {metadata['vocab_size']}.")
        self.data = {split: np.memmap(directory / f'{split}.bin', dtype=np.uint16, mode='r')
                     for split in ('train', 'val')}
        if any(len(data) <= config.block_size for data in self.data.values()):
            raise ValueError('Each token split must be longer than block_size.')
        self.training = torch.Generator().manual_seed(config.seed + rank)
        self.validation = torch.Generator().manual_seed(config.seed + 1_000_000)

    def get(self, split, evaluation=False):
        data = self.data[split]
        generator = self.validation if evaluation else self.training
        starts = torch.randint(len(data) - self.config.block_size,
                               (self.config.batch_size,), generator=generator).tolist()
        length = self.config.block_size
        x = torch.stack([torch.from_numpy(data[i:i + length].astype(np.int64)) for i in starts])
        y = torch.stack([torch.from_numpy(data[i + 1:i + length + 1].astype(np.int64)) for i in starts])
        return x.to(self.config.device), y.to(self.config.device)


@torch.no_grad()
def evaluate(model, batches, config):
    model.eval()
    result = {}
    for split in ('train', 'val'):
        losses, accuracies = [], []
        for _ in range(config.eval_iters):
            inputs, targets = batches.get(split, evaluation=True)
            with autocast(config):
                logits, loss = model(inputs, targets)
            losses.append(loss.item())
            accuracies.append((logits.argmax(-1) == targets).float().mean().item())
        result[f'{split}_loss'] = sum(losses) / len(losses)
        result[f'{split}_accuracy'] = sum(accuracies) / len(accuracies)
    model.train()
    return result


def scientific_config(config):
    values = asdict(config)
    for field in ('device', 'batch_size', 'dataset_dir', 'dataset_root', 'max_iters',
                  'output_dir', 'save_checkpoint', 'checkpoint_interval', 'wandb_log',
                  'wandb_project', 'wandb_entity'):
        values.pop(field)
    return values


def save_checkpoint(path, model, optimizer, batches, config, step, last_eval, world_size, rank):
    state = {'training': batches.training.get_state(), 'validation': batches.validation.get_state(),
             'torch': torch.get_rng_state()}
    if config.device.startswith('cuda'):
        state['cuda'] = torch.cuda.get_rng_state(config.device)
    states = [state]
    if world_size > 1:
        states = [None] * world_size
        torch.distributed.all_gather_object(states, state)
    if rank == 0:
        payload = {'format': 'adam-scaling-v1', 'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'config': config_dict(config),
                   'scientific_config': scientific_config(config), 'step': step,
                   'last_eval': last_eval, 'rng_states': states}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)


def restore_checkpoint(path, model, optimizer, batches, config, world_size, rank):
    payload = torch.load(path, map_location=config.device, weights_only=False)
    if payload.get('format') != 'adam-scaling-v1':
        raise ValueError('Expected a checkpoint from this Adam scaling release.')
    if payload['scientific_config'] != scientific_config(config):
        raise ValueError('Checkpoint scientific settings differ from the requested scaling run.')
    if len(payload['rng_states']) != world_size:
        raise ValueError('Checkpoint restoration requires the same distributed world size.')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    state = payload['rng_states'][rank]
    batches.training.set_state(state['training'].cpu())
    batches.validation.set_state(state['validation'].cpu())
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu(), config.device)
    return payload['step'], payload['last_eval']


def train(config, resume=False):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = world_size > 1
    if distributed:
        config.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(config.device)
        torch.distributed.init_process_group(backend='nccl')
    config.validate(world_size)
    if config.mode == 'digital' and config.device.startswith('cuda') and not torch.cuda.is_bf16_supported():
        raise RuntimeError('The digital scaling baseline requires a GPU with BF16 support.')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.device.startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = Path(config.output_dir or f'outputs/{config.model_scale}-{config.mode}-adam')
    output.mkdir(parents=True, exist_ok=True)
    batches = TokenBatches(config, rank)
    raw_model = build_model(config)
    optimizer = build_optimizer(raw_model, config)
    step, last_eval = 0, -1
    if resume:
        step, last_eval = restore_checkpoint(output / 'ckpt.pt', raw_model, optimizer,
                                             batches, config, world_size, rank)
        print(f'Restored checkpoint at step {step}', flush=True)
    if distributed:
        if config.mode == 'analog':
            optimizer.prepare_for_ddp(raw_model)
        model = DDP(raw_model, device_ids=[int(os.environ['LOCAL_RANK'])])
    else:
        model = raw_model
    wandb_run = None
    if rank == 0:
        (output / 'config.json').write_text(json.dumps(config_dict(config), indent=2) + '\n')
        if config.wandb_log:
            import wandb
            wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity or None,
                                  name=f'{config.model_scale}-{config.mode}-adam', config=config_dict(config))
    stop = config.max_iters or config.planned_updates
    accumulation = config.global_batch_size // (config.batch_size * world_size)
    operations_per_token = (6 * config.budget_parameters +
                           12 * config.shape['n_layer'] * config.shape['n_embd'] * config.block_size)
    print(f'Adam scaling: mode={config.mode}, scale={config.model_scale}, '
          f'planned_updates={config.planned_updates}, tokens_per_update={config.tokens_per_update}', flush=True)
    try:
        while step <= stop:
            if should_evaluate(step, config) and step != last_eval:
                if distributed:
                    torch.distributed.barrier()
                if rank == 0:
                    metrics = evaluate(raw_model, batches, config)
                    record = {'step': step, 'tokens': step * config.tokens_per_update,
                              'compute': step * config.tokens_per_update * operations_per_token, **metrics}
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise RuntimeError('Nonfinite evaluation metric.')
                    with (output / 'metrics.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                    print(f"step {step}: train loss {metrics['train_loss']:.4f}, "
                          f"val loss {metrics['val_loss']:.4f}", flush=True)
                    if wandb_run:
                        wandb_run.log(record, step=step)
                last_eval = step
                if distributed:
                    torch.distributed.barrier()
            if step == stop:
                break
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group['lr'] = lr
            started = time.monotonic()
            for micro in range(accumulation):
                if distributed:
                    model.require_backward_grad_sync = micro == accumulation - 1
                inputs, targets = batches.get('train')
                with autocast(config):
                    _, loss = model(inputs, targets)
                (loss / accumulation).backward()
            if distributed and config.mode == 'analog':
                optimizer.ddp_synchronize_gradients(model)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip,
                                          error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if rank == 0:
                print(f'iter {step}: loss {loss.item():.4f}, lr {lr:.6g}, '
                      f'time {time.monotonic() - started:.3f}s', flush=True)
            if config.save_checkpoint and config.checkpoint_interval and step % config.checkpoint_interval == 0:
                save_checkpoint(output / 'ckpt.pt', raw_model, optimizer, batches,
                                config, step, last_eval, world_size, rank)
        if config.save_checkpoint:
            save_checkpoint(output / 'ckpt.pt', raw_model, optimizer, batches,
                            config, step, last_eval, world_size, rank)
    finally:
        if wandb_run:
            wandb_run.finish()
        if distributed:
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='config/digital_adam.py or config/analog_adam.py')
    parser.add_argument('--model-scale', '--model_scale', choices=('tiny', 'small', 'medium', 'large', 'xlarge'))
    parser.add_argument('--dataset-dir', '--dataset_dir')
    parser.add_argument('--device')
    parser.add_argument('--batch-size', '--batch_size', type=int)
    parser.add_argument('--max-iters', '--max_iters', type=int)
    parser.add_argument('--output-dir', '--output_dir')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for name in ('model_scale', 'dataset_dir', 'device', 'batch_size', 'max_iters', 'output_dir'):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    train(config, resume=args.resume)


if __name__ == '__main__':
    main()
