#!/usr/bin/env python3
"""Matched ImageNet ResNet-18 subset pilot: digital, native MP SGD, S-OUT SGD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50

import train as reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from sout_resnet_cifar import (
    LogicalMpSGD,
    a09_mp_config,
    apply_s_logical_parameterization,
    layer_a09_mp_config,
    layer_sout_config,
    sout_config,
)


def class_balanced_indices(dataset, per_class: int, seed: int) -> list[int]:
    by_class = [[] for _ in range(len(dataset.classes))]
    for index, target in enumerate(dataset.targets):
        by_class[target].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for target, indices in enumerate(by_class):
        if len(indices) < per_class:
            raise ValueError(f"Class {target} has {len(indices)} samples; requested {per_class}")
        selected.extend(rng.choice(indices, size=per_class, replace=False).tolist())
    return sorted(selected)


def logical_initial_stds(model: nn.Module) -> dict[str, float]:
    # TorchVision initializes ResNet convolutions with Kaiming fan-out normal;
    # its final Linear retains nn.Linear's default fan-in uniform initializer.
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            result[name] = math.sqrt(2.0 / (module.out_channels * math.prod(module.kernel_size)))
        elif isinstance(module, nn.Linear):
            result[name] = 1.0 / math.sqrt(3.0 * module.in_features)
    return result


def build_model_and_optimizer(args, device):
    model = resnet50(weights=None, num_classes=1000)
    expected_stds = logical_initial_stds(model)
    config_args = argparse.Namespace(
        tau=args.tau, omega=args.omega, c_out=args.c_out, c2_b=args.c2_b,
        dac_bits=args.dac_bits, adc_bits=args.adc_bits,
        write_profile=args.write_profile, perfect_io=False,
        rpu_construction_seed=args.rpu_construction_seed,
    )
    if args.mode == "s_out":
        model = convert_to_analog(
            model, sout_config(config_args),
            specific_rpu_config_fun=lambda n, m, c: layer_sout_config(n, m, c, config_args),
            inplace=True,
        ).to(device)
        apply_s_logical_parameterization(
            model, args.tau, args.omega, expected_stds, "after_adc"
        )
        optimizer = LogicalMpSGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, disable_context_buffer=True,
        )
        optimizer.regroup_param_groups(model)
    elif args.mode == "native_mp":
        model = convert_to_analog(
            model, a09_mp_config(config_args),
            specific_rpu_config_fun=lambda n, m, c: layer_a09_mp_config(n, m, c, config_args),
            inplace=True,
        ).to(device)
        optimizer = AnalogSGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        optimizer.regroup_param_groups(model)
    else:
        model = model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return model, optimizer


def measure(model, loader, criterion, device, optimizer=None, log_interval=0):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct1 = correct5 = count = 0
    start = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {step}: {loss.item()}")
            if training:
                loss.backward()
                optimizer.step()
            top5 = output.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == targets).sum().item()
            correct5 += (top5 == targets[:, None]).any(dim=1).sum().item()
            n = targets.numel()
            count += n
            loss_sum += loss.item() * n
            if log_interval and step % log_interval == 0:
                print(json.dumps({"event": "progress", "phase": "train" if training else "val",
                                  "batch": step, "batches": len(loader),
                                  "loss": loss_sum / count, "top1": correct1 / count,
                                  "elapsed_seconds": round(time.monotonic() - start, 1)}), flush=True)
    torch.cuda.synchronize()
    return {"loss": loss_sum / count, "top1": correct1 / count,
            "top5": correct5 / count, "samples": count,
            "seconds": round(time.monotonic() - start, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("digital", "native_mp", "s_out"), required=True)
    parser.add_argument("--data-path", default="/home/zw868/Desktop/imagenet_2012")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.train_per_class <= 0 or args.val_per_class <= 0:
        parser.error("batch size, epochs, and per-class sample counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    data_args = reference.get_args_parser().parse_args(["--data-path", args.data_path])
    data_args.distributed = False
    train_set, val_set, _, _ = reference.load_data(
        str(Path(args.data_path) / "train"), str(Path(args.data_path) / "val"), data_args
    )
    train_indices = class_balanced_indices(train_set, args.train_per_class, args.seed + 100)
    val_indices = class_balanced_indices(val_set, args.val_per_class, args.seed + 200)
    index_record = {"train_indices": train_indices, "val_indices": val_indices}
    index_file = args.output_dir.parent / "subset_indices.json"
    if index_file.exists():
        if json.loads(index_file.read_text()) != index_record:
            raise RuntimeError("Existing subset indices differ from this run")
    else:
        index_file.write_text(json.dumps(index_record) + "\n")
    train_generator = torch.Generator()
    train_loader = DataLoader(
        Subset(train_set, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=train_generator, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_set, val_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    model, optimizer = build_model_and_optimizer(args, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps({"event": "config", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      "source_sha256": source_hash, "torch": torch.__version__,
                      "train_samples": len(train_indices), "val_samples": len(val_indices),
                      "train_batches": len(train_loader), "val_batches": len(val_loader)}, sort_keys=True), flush=True)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["mode"] != args.mode or checkpoint["source_sha256"] != source_hash:
            raise RuntimeError("Resume mode or source version mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        print(json.dumps({"event": "resume", "next_epoch": start_epoch + 1}), flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_generator.manual_seed(args.seed + epoch)
        train_metrics = measure(model, train_loader, criterion, device, optimizer, log_interval=50)
        val_metrics = measure(model, val_loader, criterion, device, log_interval=0)
        scheduler.step()
        print(json.dumps({"event": "epoch", "mode": args.mode, "epoch": epoch,
                          "train": train_metrics, "val": val_metrics,
                          "next_lr": scheduler.get_last_lr()[0]}, sort_keys=True), flush=True)
        checkpoint = {"mode": args.mode, "source_sha256": source_hash, "epoch": epoch,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(),
                      "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        temporary = args.output_dir / "checkpoint.pth.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output_dir / "checkpoint.pth")
    (args.output_dir / "complete.txt").write_text(f"epochs={args.epochs}\n")


if __name__ == "__main__":
    main()
