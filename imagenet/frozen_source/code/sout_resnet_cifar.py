#!/usr/bin/env python3
"""Paired digital, S-OUT, and native-MP ResNet experiments on CIFAR-10/100.

S-OUT uses the paper's fixed logical reparameterization W=s*W_hat, logical
open-loop logical updates, Li-ECRAM tiles, 16-bit DAC/ADC, and static output
rails.  A convolution is interpreted as a matrix with K=C_in*k_h*k_w input
coordinates and M=C_out output coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, build_config
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, PulseType
from aihwkit.simulator.presets.devices import EcRamPresetDevice

ROOT = Path(__file__).resolve().parents[1]
NANOGPT_ROOT = ROOT / "code" / "E1-nanoGPT"
if str(NANOGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOGPT_ROOT))

from MixedPrecisionOptim.LogicalAdamW import LogicalMpAdamW
from MixedPrecisionOptim.LogicalOptimizers import LogicalMpSGD
from scaled_analog_linear import derive_rpu_construction_seed, set_rpu_construction_seed


class _ScaleInputGradient(torch.autograd.Function):
    """Keep the forward input unchanged; rescale only its returned gradient."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.scale, None


class _ScaleOutputValue(torch.autograd.Function):
    """Scale the logical output after the ADC, without scaling the DAC error."""

    @staticmethod
    def forward(ctx, output: torch.Tensor, scale: float) -> torch.Tensor:
        return output * scale

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


class LogicalAnalogLayer(nn.Module):
    """Apply S's fixed logical scale around one converted Conv2d/Linear tile."""
    def __init__(self, analog_layer: nn.Module, scale: float, backward_scale_position: str) -> None:
        super().__init__()
        self.analog_layer = analog_layer
        self.register_buffer("logical_scale", torch.tensor(float(scale)))
        self._scale_float = float(scale)
        self.backward_scale_position = backward_scale_position
        analog_module = analog_layer.analog_module
        contexts = [analog_module.analog_ctx] if hasattr(analog_module, "analog_ctx") else [
            tile.analog_ctx for tile in analog_module.modules() if hasattr(tile, "analog_ctx")
        ]
        if not contexts:
            raise RuntimeError("Converted analog layer exposes no tile contexts")
        for context in contexts:
            context.logical_weight_scale = float(scale)
            context.logical_gradient_units = True
            # The corrected read sends the raw logical error into the backward
            # DAC. The legacy path differentiates s*W_hat and needs division.
            context.logical_native_gradient_scale = (
                float(scale) if backward_scale_position == "before_dac" else 1.0
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self._scale_float
        if self.backward_scale_position == "after_adc":
            inputs = _ScaleInputGradient.apply(inputs, scale)
        output = self.analog_layer(inputs)
        bias = getattr(self.analog_layer.analog_module, "bias", None)
        if self.backward_scale_position == "after_adc":
            output = _ScaleOutputValue.apply(output, scale)
            if isinstance(bias, torch.Tensor):
                if isinstance(self.analog_layer, AnalogLinear):
                    # Linear features occupy the final axis, including for
                    # transformer tensors shaped [batch, token, feature].
                    bias_shape = (1,) * (output.ndim - 1) + (bias.numel(),)
                else:
                    bias_shape = (1, bias.numel()) + (1,) * (output.ndim - 2)
                output = output + (1.0 - scale) * bias.detach().view(bias_shape)
            return output
        if not isinstance(bias, torch.Tensor):
            return scale * output
        if isinstance(self.analog_layer, AnalogLinear):
            bias_shape = (1,) * (output.ndim - 1) + (bias.numel(),)
        else:
            bias_shape = (1, bias.numel()) + (1,) * (output.ndim - 2)
        return scale * output + (1.0 - scale) * bias.view(bias_shape)


def expected_logical_std(name: str, layer: nn.Module) -> float:
    """Torchvision ResNet's prescribed initialization, before sampling noise."""
    if isinstance(layer, nn.Linear) or name == "conv1":
        fan_in = layer.in_features if isinstance(layer, nn.Linear) else matrix_shape(layer)[0]
        return 1.0 / math.sqrt(3.0 * fan_in)
    if isinstance(layer, nn.Conv2d):
        kernel_h, kernel_w = layer.kernel_size
        fan_out = layer.out_channels * kernel_h * kernel_w / layer.groups
        return math.sqrt(2.0 / fan_out)
    raise TypeError(f"Unsupported initialization layer: {type(layer).__name__}")


@torch.no_grad()
def apply_s_logical_parameterization(
    model: nn.Module, tau: float, omega: float, expected_stds: dict[str, float],
    backward_scale_position: str, log_initialization: bool = False,
) -> None:
    """Initialize with predetermined s=omega*sigma_w/tau, independent of draws."""
    found = set()
    for full_name, child in list(model.named_modules()):
        if not isinstance(child, (AnalogConv2d, AnalogLinear)):
            continue
        name = full_name.removeprefix("module.")
        if name not in expected_stds:
            raise RuntimeError(f"No prescribed logical standard deviation for {name}")
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        logical_weight, logical_bias = child.get_weights(apply_weight_scaling=False)
        scale = omega * expected_stds[name] / tau
        child.set_weights(logical_weight / scale, logical_bias, apply_weight_scaling=False)
        # Logical SGD's coupled L2 penalty must use the logical target rather
        # than the AnalogContext residual.  Bind the converter-time logical
        # value as immutable optimizer initialization metadata.  AnalogConv2d
        # exposes its kernel in 4-D while its single tile/context is the
        # corresponding 2-D matrix, so reshape without changing element order.
        analog_module = child.analog_module
        contexts = (
            [analog_module.analog_ctx]
            if hasattr(analog_module, "analog_ctx")
            else [
                tile.analog_ctx
                for tile in analog_module.modules()
                if hasattr(tile, "analog_ctx")
            ]
        )
        if len(contexts) != 1:
            raise RuntimeError(
                "S-OUT ResNet expects one unsplit tile per Conv2d/Linear; "
                f"{name} exposes {len(contexts)} contexts"
            )
        context = contexts[0]
        if logical_weight.numel() != context.numel():
            raise RuntimeError(
                "Logical initial weight cannot be reshaped to its tile: "
                f"layer={name}, weight={tuple(logical_weight.shape)}, "
                f"context={tuple(context.shape)}"
            )
        context.logical_initial_weight = logical_weight.detach().reshape(
            context.shape
        ).clone()
        if log_initialization:
            physical_weight, _ = child.get_weights(apply_weight_scaling=False)
            print(json.dumps({
                "event": "initial_layer", "layer": name, "logical_scale": scale,
                "logical_std_target": expected_stds[name],
                "logical_std_sample": float(logical_weight.float().std().item()),
                "physical_std_realized": float(physical_weight.float().std().item()),
                "physical_std_target": tau / omega,
            }, sort_keys=True), flush=True)
        setattr(parent, child_name, LogicalAnalogLayer(child, scale, backward_scale_position))
        found.add(name)
    if found != set(expected_stds):
        raise RuntimeError(f"Missing converted analog layers: {sorted(set(expected_stds) - found)}")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def io_resolution(bits: int) -> float:
    if bits < 2:
        raise ValueError("DAC/ADC precision must be at least 2 bits")
    return 1.0 / (2**bits - 2)


def matrix_shape(module: nn.Module) -> tuple[int, int]:
    """Return (K, M) for the matrix represented by a Conv2d or Linear."""
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("S-OUT runner currently supports only groups=1 convolutions")
        kernel_h, kernel_w = module.kernel_size
        return int(module.in_channels * kernel_h * kernel_w), int(module.out_channels)
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features)
    raise TypeError(f"Unsupported analogized module type: {type(module).__name__}")


def write_device(args: argparse.Namespace):
    ecram_reference = EcRamPresetDevice()
    if args.write_profile == "ecram":
        return ecram_reference
    elif args.write_profile == "asym_ideal":
        ecram_reference.dw_min = 0.0
        return ecram_reference
    else:
        retain_ecram_bounds = args.write_profile == "flat_ecram_bounds"
        return ConstantStepDevice(
            dw_min=(ecram_reference.dw_min if args.write_profile in ("sym_dwmin", "flat_ecram_bounds") else 0.0),
            w_max=ecram_reference.w_max if retain_ecram_bounds else args.tau,
            w_min=ecram_reference.w_min if retain_ecram_bounds else -args.tau,
            w_max_dtod=0.0, w_min_dtod=0.0, up_down_dtod=0.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
        )


def configure_read_path(config, args: argparse.Namespace):
    """Apply the common current read/mapping profile to any training config."""
    config.mapping.learn_out_scaling = False
    config.mapping.out_scaling_columnwise = False
    config.mapping.weight_scaling_omega = 0.0
    config.mapping.max_input_size = 2**30
    config.mapping.max_output_size = 2**30
    config.forward.inp_res = io_resolution(args.dac_bits)
    config.forward.out_res = io_resolution(args.adc_bits)
    config.backward.inp_res = io_resolution(args.dac_bits)
    config.backward.out_res = io_resolution(args.adc_bits)
    config.forward.noise_management = NoiseManagementType.ABS_MAX
    config.backward.noise_management = NoiseManagementType.ABS_MAX
    config.forward.bound_management = BoundManagementType.ITERATIVE
    # AIHWKit does not implement meaningful bound management for the backward read.
    config.backward.bound_management = BoundManagementType.NONE
    if args.perfect_io:
        config.forward.is_perfect = True
        config.backward.is_perfect = True
    return config


def sout_config(args: argparse.Namespace) -> SingleRPUConfig:
    config = SingleRPUConfig(device=write_device(args))
    configure_read_path(config, args)
    # LogicalMp* performs explicit full-rank physical pulse transfers.
    config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
    return config


def a09_mp_config(args: argparse.Namespace):
    """A09-style native mixed precision, matched to the current device/read profile."""
    config = build_config(
        "mp", device=write_device(args), construction_seed=args.rpu_construction_seed
    )
    return configure_read_path(config, args)


def layer_sout_config(name: str, module: nn.Module, config: SingleRPUConfig, args: argparse.Namespace) -> SingleRPUConfig:
    # The AIHWKit converter calls the hook while descending through containers
    # as well as on modules that become tiles. Only matrix modules receive a
    # matrix-specific output rail.
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return config
    layer = deepcopy(config)
    set_rpu_construction_seed(layer, args.rpu_construction_seed, name)
    k, m = matrix_shape(module)
    physical_std = args.tau / args.omega
    layer.forward.out_bound = args.c_out * physical_std * math.sqrt(k)
    layer.backward.out_bound = args.c2_b * physical_std * math.sqrt(m)
    return layer


def layer_a09_mp_config(name: str, module: nn.Module, config, args: argparse.Namespace):
    """Set deterministic per-layer seeds and the same static output rails as S-OUT."""
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return config
    layer = deepcopy(config)
    seed = derive_rpu_construction_seed(args.rpu_construction_seed, name)
    if seed:
        layer.device.construction_seed = seed
        layer.device.device.construction_seed = seed
    k, m = matrix_shape(module)
    physical_std = args.tau / args.omega
    layer.forward.out_bound = args.c_out * physical_std * math.sqrt(k)
    layer.backward.out_bound = args.c2_b * physical_std * math.sqrt(m)
    return layer


def make_resnet(architecture: str, num_classes: int) -> nn.Module:
    constructor = getattr(models, architecture.lower(), None)
    if constructor is None:
        raise ValueError(f"Unknown torchvision ResNet architecture: {architecture}")
    model = constructor(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    policy = transforms.AutoAugmentPolicy.CIFAR10 if args.dataset == "CIFAR10" else transforms.AutoAugmentPolicy.CIFAR10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.4, 2.5), value=0),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset_type = datasets.CIFAR10 if args.dataset == "CIFAR10" else datasets.CIFAR100
    root = Path(args.data_root) / args.dataset
    train_set = dataset_type(root=root, train=True, download=False, transform=train_transform)
    test_set = dataset_type(root=root, train=False, download=False, transform=test_transform)
    generator = torch.Generator().manual_seed(args.seed + 100)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 0) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += criterion(logits, targets).item() * targets.numel()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    if total == 0:
        raise RuntimeError("Evaluation loader produced no examples")
    return loss_sum / total, correct / total


@torch.no_grad()
def diagnostic_snapshot(model: nn.Module, optimizer, epoch: int) -> None:
    """Observe simulator and BN state without feeding any value into updates."""
    for name, layer in model.named_modules():
        if not isinstance(layer, LogicalAnalogLayer):
            continue
        tile = layer.analog_layer.analog_module
        physical, _ = layer.analog_layer.get_weights(apply_weight_scaling=False)
        physical = physical.float()
        bounds = tile.rpu_config.device
        context = tile.analog_ctx
        state = optimizer.state.get(context, {})
        residual = state.get("logical_residual")
        granularity = layer._scale_float * float(bounds.dw_min)
        max_pulses = int(tile.rpu_config.update.desired_bl)
        print(json.dumps({
            "event": "tile_diagnostic", "epoch": epoch, "layer": name,
            "physical_mean": physical.mean().item(),
            "physical_std": physical.std().item(),
            "near_lower_bound": (physical <= bounds.w_min + 0.02).float().mean().item(),
            "near_upper_bound": (physical >= bounds.w_max - 0.02).float().mean().item(),
            "logical_residual_rms": (
                residual.float().square().mean().sqrt().item()
                if isinstance(residual, torch.Tensor) else None
            ),
            "residual_over_one_pulse": (
                (residual.abs() >= granularity).float().mean().item()
                if isinstance(residual, torch.Tensor) and granularity > 0 else None
            ),
            "residual_over_pulse_limit": (
                (residual.abs() >= granularity * max_pulses).float().mean().item()
                if isinstance(residual, torch.Tensor) and granularity > 0 else None
            ),
            "logical_moment_rms": (
                state["exp_avg"].float().square().mean().sqrt().item()
                if "exp_avg" in state else None
            ),
        }, sort_keys=True), flush=True)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            print(json.dumps({
                "event": "bn_diagnostic", "epoch": epoch, "layer": name,
                "running_mean_rms": layer.running_mean.float().square().mean().sqrt().item(),
                "running_var_min": layer.running_var.min().item(),
                "running_var_max": layer.running_var.max().item(),
                "gamma_rms": layer.weight.float().square().mean().sqrt().item(),
            }, sort_keys=True), flush=True)


def train(args: argparse.Namespace) -> dict[str, object]:
    seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    num_classes = 10 if args.dataset == "CIFAR10" else 100
    train_loader, test_loader = make_loaders(args)
    model = make_resnet(args.architecture, num_classes)
    expected_stds = {
        name: expected_logical_std(name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    }

    print(json.dumps({
        "event": "run_config", "arguments": vars(args),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
    }, sort_keys=True), flush=True)

    if args.mode == "s_out":
        model = convert_to_analog(model, sout_config(args), specific_rpu_config_fun=lambda n, m, c: layer_sout_config(n, m, c, args), inplace=True)
        model = model.to(device)
        apply_s_logical_parameterization(model, args.tau, args.omega, expected_stds,
                                         args.backward_scale_position, args.log_initialization)
        if args.logical_optimizer == "adamw":
            optimizer_name = "LogicalMpAdamW"
            optimizer = LogicalMpAdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                       weight_decay=args.weight_decay, disable_context_buffer=True)
        else:
            optimizer_name = "LogicalMpSGD"
            optimizer = LogicalMpSGD(model.parameters(), lr=args.lr, momentum=0.0,
                                     weight_decay=args.weight_decay, disable_context_buffer=True)
        optimizer.regroup_param_groups(model)
    elif args.mode == "a09_mp":
        optimizer_name = "A09MixedPrecisionCompound+AnalogSGD"
        model = convert_to_analog(
            model, a09_mp_config(args),
            specific_rpu_config_fun=lambda n, m, c: layer_a09_mp_config(n, m, c, args),
            inplace=True,
        )
        model = model.to(device)
        optimizer = AnalogSGD(
            model.parameters(), lr=args.lr, momentum=0.0,
            weight_decay=args.weight_decay,
        )
        optimizer.regroup_param_groups(model)
    else:
        optimizer_name = "torch.optim.AdamW"
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.lr_tmax or args.epochs,
        eta_min=(args.lr * 0.1 if args.lr_eta_min is None else args.lr_eta_min),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    initial_loss, initial_acc = evaluate(model, test_loader, device, args.max_test_batches)
    print(json.dumps({"event": "initial_eval", "loss": initial_loss, "accuracy": initial_acc}, sort_keys=True), flush=True)
    if args.diagnostic_every:
        diagnostic_snapshot(model, optimizer, 0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = total = 0
        optimizer.zero_grad(set_to_none=True)
        pending_microbatches = 0
        train_batches = min(len(train_loader), args.max_train_batches or len(train_loader))
        for batch_index, (inputs, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch}, batch={batch_index}: {loss.item()}")
            group_start = batch_index - pending_microbatches
            group_end = min(group_start + args.grad_accum, train_batches)
            group_size = group_end - group_start
            group_examples = min(len(train_loader.dataset), group_end * args.batch_size) - group_start * args.batch_size
            (loss * targets.numel() / group_examples).backward()
            pending_microbatches += 1
            if pending_microbatches == group_size:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_microbatches = 0
            loss_sum += loss.item() * targets.numel()
            total += targets.numel()
        if total == 0:
            raise RuntimeError("Training loader produced no examples")
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, device, args.max_test_batches)
        record = {"event": "epoch", "epoch": epoch, "train_loss": loss_sum / total,
                  "test_loss": test_loss, "test_accuracy": test_acc,
                  "lr": scheduler.get_last_lr()[0]}
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.diagnostic_every and epoch % args.diagnostic_every == 0:
            diagnostic_snapshot(model, optimizer, epoch)

    result = {"event": "final", "mode": args.mode, "dataset": args.dataset,
              "architecture": args.architecture, "epochs": args.epochs,
              "test_accuracy": test_acc, "test_loss": test_loss,
              "batch_size": args.batch_size, "grad_accum": args.grad_accum,
              "weight_decay": args.weight_decay,
              "lr_eta_min": (args.lr * 0.1 if args.lr_eta_min is None else args.lr_eta_min),
              "tau": args.tau, "omega": args.omega, "c_out": args.c_out,
              "c2_b": args.c2_b, "dac_bits": args.dac_bits, "adc_bits": args.adc_bits,
              "seed": args.seed, "backward_scale_position": args.backward_scale_position,
              "rpu_construction_seed": args.rpu_construction_seed, "perfect_io": args.perfect_io,
              "write_profile": args.write_profile, "optimizer": optimizer_name}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("digital", "s_out", "a09_mp"), required=True)
    parser.add_argument("--dataset", choices=("CIFAR10", "CIFAR100"), required=True)
    parser.add_argument(
        "--architecture",
        choices=("ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"),
        required=True,
    )
    parser.add_argument("--data-root", default="/home/zw868/Desktop/2023.09.05-analog/data")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-eta-min", type=float)
    parser.add_argument("--logical-optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=3.0)
    parser.add_argument("--rpu-construction-seed", type=int, default=1337)
    parser.add_argument("--backward-scale-position", choices=("after_adc", "before_dac"), default="after_adc")
    parser.add_argument("--perfect-io", action="store_true")
    parser.add_argument("--write-profile", choices=("ecram", "asym_ideal", "sym_dwmin", "sym_ideal", "flat_ecram_bounds"), default="ecram")
    parser.add_argument("--lr-tmax", type=int, default=0)
    parser.add_argument("--log-initialization", action="store_true")
    parser.add_argument("--diagnostic-every", type=int, default=0)
    parser.add_argument("--c-out", type=float, default=6.0)
    parser.add_argument("--c2-b", type=float, default=3.5)
    parser.add_argument("--dac-bits", type=int, default=16)
    parser.add_argument("--adc-bits", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args()
    if args.grad_accum <= 0:
        parser.error("--grad-accum must be positive")
    if args.omega <= 1.0:
        parser.error("--omega must exceed 1 for Std(W_hat)=tau/omega")
    if args.rpu_construction_seed < 0 or args.lr_tmax < 0 or args.diagnostic_every < 0:
        parser.error("construction seed, lr-tmax, and diagnostic-every must be nonnegative")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be nonnegative")
    if args.lr_eta_min is not None and not 0.0 <= args.lr_eta_min <= args.lr:
        parser.error("--lr-eta-min must be between zero and --lr")
    train(args)


if __name__ == "__main__":
    main()
