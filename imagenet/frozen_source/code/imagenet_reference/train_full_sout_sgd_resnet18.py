#!/usr/bin/env python3
"""Full ImageNet-1K ResNet-18 S-OUT SGD with resumable W&B logging.

Uses all 1,281,167 training and 50,000 validation examples and the same
TorchVision transforms and 30-epoch StepLR schedule as the digital baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from aihwkit.optim.context import AnalogContext

import train as reference
from train_analog_resnet18 import build_model_and_optimizer as build_sgd
from train_analog_resnet18 import measure
from sout_resnet_cifar import LogicalAnalogLayer, LogicalMpSGD


EXPECTED_TRAIN = 1_281_167
EXPECTED_VAL = 50_000
ENTITY = "analog-transformer"
PROJECT = "ImageNet"
KEY_FILE = Path("/home/zw868/.config/wandb/imagenet.key")


def source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "code").rglob("*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("s_out",), required=True)
    parser.add_argument("--data-path", type=Path, default=Path("/home/zw868/Desktop/imagenet_2012"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=3.0)
    parser.add_argument("--c-out", type=float, default=6.0)
    parser.add_argument("--c2-b", type=float, default=3.5)
    parser.add_argument("--dac-bits", type=int, default=16)
    parser.add_argument("--adc-bits", type=int, default=16)
    parser.add_argument("--write-profile", default="ecram")
    parser.add_argument("--rpu-construction-seed", type=int, default=1337)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-id", required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.workers < 0 or args.lr <= 0:
        parser.error("batch size, epochs, and learning rate must be positive; workers must be nonnegative")
    if not 0 <= args.momentum < 1 or args.weight_decay < 0:
        parser.error("momentum must be in [0,1); weight decay must be nonnegative")
    return args


def wandb_record(item: dict) -> dict:
    return {
        "epoch": item["epoch"],
        "train/loss": item["train"]["loss"],
        "train/top1_pct": 100 * item["train"]["top1"],
        "train/top5_pct": 100 * item["train"]["top5"],
        "train/seconds": item["train"]["seconds"],
        "val/loss": item["val"]["loss"],
        "val/top1_pct": 100 * item["val"]["top1"],
        "val/top5_pct": 100 * item["val"]["top5"],
        "val/seconds": item["val"]["seconds"],
        "lr_next_epoch": item["next_lr"],
    }


def analog_contexts(model: nn.Module) -> dict[str, AnalogContext]:
    result = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, LogicalAnalogLayer):
            continue
        analog_module = layer.analog_layer.analog_module
        contexts = (
            [analog_module.analog_ctx]
            if hasattr(analog_module, "analog_ctx")
            else [tile.analog_ctx for tile in analog_module.modules()
                  if hasattr(tile, "analog_ctx")]
        )
        if len(contexts) != 1:
            raise RuntimeError(f"S-OUT layer {name} requires one analog context")
        result[name] = contexts[0]
    if len(result) != 21:
        raise RuntimeError(f"Expected 21 analog layers, got {len(result)}")
    return result


def initial_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: context.logical_initial_weight.detach().cpu().clone()
        for name, context in analog_contexts(model).items()
    }


def rebuild_optimizer_after_model_load(
    model: nn.Module, args: argparse.Namespace, saved_initial: dict[str, torch.Tensor]
):
    """Rebind optimizer parameters after AIHWKit replaces tile contexts on load."""
    contexts = analog_contexts(model)
    if contexts.keys() != saved_initial.keys():
        raise RuntimeError("Checkpoint analog layers differ from restored model")
    for name, layer in model.named_modules():
        if name not in contexts:
            continue
        context = contexts[name]
        initial = saved_initial[name].to(device=context.device, dtype=context.dtype)
        if initial.shape != context.shape:
            raise RuntimeError(f"Initial weight shape mismatch for {name}")
        context.logical_weight_scale = float(layer.logical_scale.item())
        context.logical_gradient_units = True
        context.logical_native_gradient_scale = (
            context.logical_weight_scale if layer.backward_scale_position == "before_dac" else 1.0
        )
        context.logical_initial_weight = initial
    optimizer = LogicalMpSGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, disable_context_buffer=True,
    )
    optimizer.regroup_param_groups(model)
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    if model_ids != optimizer_ids:
        raise RuntimeError("Optimizer parameters do not match the restored model")
    return optimizer


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    data_args = reference.get_args_parser().parse_args(["--data-path", str(args.data_path)])
    data_args.distributed = False
    train_set, val_set, _, _ = reference.load_data(
        str(args.data_path / "train"), str(args.data_path / "val"), data_args
    )
    if len(train_set) != EXPECTED_TRAIN or len(val_set) != EXPECTED_VAL:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN}/{EXPECTED_VAL} images, got {len(train_set)}/{len(val_set)}")
    if len(train_set.classes) != 1000 or train_set.classes != val_set.classes:
        raise RuntimeError("ImageNet train and val must have matching 1000 classes")
    train_generator = torch.Generator()
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=train_generator,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model, optimizer = build_sgd(args, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    source_hash = source_digest()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({
        "dataset": "ImageNet-1K", "dataset_split": "full", "architecture": "resnet18",
        "train_samples": len(train_set), "val_samples": len(val_set),
        "train_batches": len(train_loader), "val_batches": len(val_loader),
        "source_sha256": source_hash, "torch": torch.__version__, "amp": False,
        "optimizer": "LogicalMpSGD", "analog_gradient_scale": "after_adc",
    })
    print(json.dumps({"event": "config", **config}, sort_keys=True), flush=True)

    start_epoch = 0
    if args.resume:
        # AIHWKit recreates each tile on the checkpoint tensor's device. A
        # CUDA load preserves its native tile layout; release the temporary
        # checkpoint tensors immediately after state restoration below.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["mode"] != args.mode or checkpoint["source_sha256"] != source_hash:
            raise RuntimeError("Resume mode or source version mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer = rebuild_optimizer_after_model_load(model, args, checkpoint["initial_weights"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        del checkpoint
        torch.cuda.empty_cache()
        print(json.dumps({"event": "resume", "next_epoch": start_epoch + 1,
                          "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20),
                          "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20)}), flush=True)

    run = None
    if args.wandb:
        import os
        import wandb

        os.environ["WANDB_API_KEY"] = KEY_FILE.read_text().strip()
        wandb_dir = args.output_dir / "wandb"
        wandb_dir.mkdir(exist_ok=True)
        run = wandb.init(
            entity=ENTITY, project=PROJECT, id=args.wandb_id,
            name=f"resnet18 {args.mode} full ImageNet 90ep",
            group="r18_full_imagenet_20261014_sout_sgd", job_type="training",
            tags=["full_imagenet", "resnet18", args.mode], config=config,
            dir=str(wandb_dir), resume="allow", settings=wandb.Settings(init_timeout=120),
        )
        (args.output_dir / "wandb_url.txt").write_text(run.url + "\n")
        print(json.dumps({"event": "wandb", "url": run.url}), flush=True)

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train_generator.manual_seed(args.seed + epoch)
            train_metrics = measure(model, train_loader, criterion, device, optimizer, log_interval=500)
            val_metrics = measure(model, val_loader, criterion, device)
            scheduler.step()
            item = {"event": "epoch", "mode": args.mode, "epoch": epoch,
                    "train": train_metrics, "val": val_metrics,
                    "next_lr": scheduler.get_last_lr()[0]}
            print(json.dumps(item, sort_keys=True), flush=True)
            checkpoint = {
                "mode": args.mode, "source_sha256": source_hash, "epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "initial_weights": initial_weights(model),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            temporary = args.output_dir / "checkpoint.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output_dir / "checkpoint.pth")
            if run is not None:
                run.log(wandb_record(item), step=epoch)
                run.summary["epochs_recorded"] = epoch
                run.summary["final_val_top1_pct"] = 100 * val_metrics["top1"]
                run.summary["best_val_top1_pct"] = max(
                    run.summary.get("best_val_top1_pct", 0), 100 * val_metrics["top1"]
                )
        (args.output_dir / "complete.txt").write_text(f"epochs={args.epochs}\n")
        if run is not None:
            run.summary["status"] = "complete"
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
