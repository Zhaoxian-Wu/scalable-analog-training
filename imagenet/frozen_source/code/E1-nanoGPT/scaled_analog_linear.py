"""Logical-weight scaled analog MVMs for nanoGPT.

The array stores a conductance matrix ``hat W`` while the network uses the
logical matrix ``W = s * hat W``.  Forward activations and backward errors are
normalized digitally before the physical MVM and restored digitally after it.
Weight gradients are accumulated directly in logical-weight units, so the
normalized read signals never leak into the optimizer update.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.nn import functional as F

from aihwkit.nn import AnalogLinear


GPT2_REFERENCE_WIDTH = 768
GPT2_MATRIX_STD = 0.02
GELU_ARGMIN = -0.7517915246935645
GELU_OUTPUT_MIN = -0.1699712074799037

FORWARD_RMS = "rms"
FORWARD_GELU = "gelu_midpoint"
FORWARD_IDENTITY = "identity"
BACKWARD_RMS = 0
BACKWARD_L2 = 1
BACKWARD_SPLIT_L2 = 2
BACKWARD_IDENTITY = 3
BACKWARD_SPLIT_L2_EXACT = 4

LOGICAL_INIT_WIDTH_STABLE = "width_stable"
LOGICAL_INIT_NANOGPT = "nanogpt"
LM_HEAD_MODE_SIGNED_L2 = "signed_l2"
LM_HEAD_MODE_SPLIT_L2 = "split_l2"
LM_HEAD_MODE_SIGMA_WINDOW = "sigma_window"

READ_RANGE_NONE = "none"
READ_RANGE_GAUSSIAN = "gaussian"
READ_RANGE_LAPLACE = "laplace"
READ_RANGE_ADAPTIVE = "adaptive"
READ_RANGE_LP_ADAPTIVE = "lp_adaptive"
READ_RANGE_MODES = {
    READ_RANGE_NONE,
    READ_RANGE_GAUSSIAN,
    READ_RANGE_LAPLACE,
    READ_RANGE_ADAPTIVE,
    READ_RANGE_LP_ADAPTIVE,
}
# ACIQ assumes a parametric input distribution and analytically balances the
# quantizer's granular error against clipping error. Keep these names equal to
# the historical range modes so existing ``s_opt_nm`` checkpoints remain
# loadable while formal ACIQ experiments have an explicit interface.
ACIQ_DISTRIBUTION_GAUSSIAN = READ_RANGE_GAUSSIAN
ACIQ_DISTRIBUTION_LAPLACE = READ_RANGE_LAPLACE
ACIQ_DISTRIBUTIONS = {
    ACIQ_DISTRIBUTION_GAUSSIAN,
    ACIQ_DISTRIBUTION_LAPLACE,
}

# Above this order the finite-p norm is both numerically fragile and already
# close to its infinity-norm limit for the small transformer vectors studied
# here.  The threshold is deliberately much larger than the finite-p values
# used at 3--6 bits for D=48.
LP_ADAPTIVE_MAX_FINITE_P = 128.0


def derive_rpu_construction_seed(base_seed: int, module_name: str) -> int:
    """Derive a stable nonzero uint32 device seed for one analog module.

    ``base_seed == 0`` is the opt-out used by historical experiments.  A
    cryptographic digest is used instead of Python's process-randomized
    ``hash()`` so paired realizations are reproducible across processes and
    machines.
    """

    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base construction seed must be an integer")
    if base_seed < 0:
        raise ValueError("base construction seed must be nonnegative")
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("module_name must be a nonempty string")
    if base_seed == 0:
        return 0

    payload = f"aimc-rpu-v1\0{base_seed}\0{module_name}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    # AIHWKit reserves zero for its historical nondeterministic construction.
    return seed if seed != 0 else 1


def set_rpu_construction_seed(
    rpu_config: Any,
    base_seed: int,
    module_name: str,
) -> int:
    """Apply the derived seed to a simple RPU device, or fail explicitly.

    Composite/device types without a writable top-level ``construction_seed``
    are deliberately rejected when deterministic construction is requested;
    silently setting an unrelated Python attribute would leave the physical
    realization unpaired.
    """

    seed = derive_rpu_construction_seed(base_seed, module_name)
    if seed == 0:
        return 0

    device = getattr(rpu_config, "device", None)
    if device is None or not hasattr(device, "construction_seed"):
        device_name = type(device).__name__ if device is not None else "None"
        raise TypeError(
            "RPU_CONSTRUCTION_SEED requires a device with a writable "
            f"construction_seed field; module {module_name!r} uses {device_name}"
        )
    try:
        device.construction_seed = seed
    except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise TypeError(
            "RPU_CONSTRUCTION_SEED could not set construction_seed for "
            f"module {module_name!r} on {type(device).__name__}"
        ) from exc
    if int(device.construction_seed) != seed:
        raise RuntimeError(
            f"construction_seed assignment was not retained for module {module_name!r}"
        )
    return seed


def _exact_nonzero_scale(energy: Tensor, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
    """Return ``sqrt(energy)`` for active rows and a safe unit scale otherwise.

    Backward errors already include the loss-reduction factor and can be much
    smaller than ``sqrt(eps)``. Adding epsilon to a nonzero row's energy would
    therefore suppress the normalized physical-read signal and waste DAC
    range. Exact zero rows use a unit placeholder scale and are masked out.
    """

    active = energy > 0.0
    scale = torch.where(active, torch.sqrt(energy), torch.ones_like(energy))
    return scale.to(dtype=dtype), active


def logical_weight_std(
    input_width: int,
    n_layer: int,
    residual_projection: bool = False,
    init_mode: str = LOGICAL_INIT_WIDTH_STABLE,
) -> float:
    """Return the width-stable nanoGPT-reference logical-weight std.

    Dense matrices use ``0.02 * sqrt(768 / D)`` so unit-RMS MVM outputs retain
    a width-independent initialization scale and recover nanoGPT at ``D=768``.
    Residual ``c_proj`` matrices additionally carry nanoGPT's ``1/sqrt(2L)``.
    """

    if input_width <= 0:
        raise ValueError("input_width must be positive")
    if n_layer <= 0:
        raise ValueError("n_layer must be positive")
    if init_mode == LOGICAL_INIT_WIDTH_STABLE:
        std = GPT2_MATRIX_STD * math.sqrt(GPT2_REFERENCE_WIDTH / input_width)
    elif init_mode == LOGICAL_INIT_NANOGPT:
        std = GPT2_MATRIX_STD
    else:
        raise ValueError(f"Unknown logical initialization mode: {init_mode}")
    if residual_projection:
        std /= math.sqrt(2.0 * n_layer)
    return std


def fixed_output_scale(
    n_layer: int,
    input_width: int,
    residual_projection: bool = False,
    omega: float = 1.0,
    conductance_bound: float = 1.0,
    init_mode: str = LOGICAL_INIT_WIDTH_STABLE,
) -> float:
    """Return ``s`` for ``W = s * hat W``.

    ``Std(hat W) = conductance_bound / omega`` and therefore
    ``s = omega * Std(W) / conductance_bound``.
    """

    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if conductance_bound <= 0.0:
        raise ValueError("conductance_bound must be positive")
    return (
        omega
        * logical_weight_std(
            input_width, n_layer, residual_projection, init_mode
        )
        / conductance_bound
    )


def position_embedding_output_scale(
    omega: float,
    conductance_bound: float = 1.0,
) -> float:
    """Preserve nanoGPT's N(0, 0.02^2) position-embedding distribution."""

    if omega <= 0.0 or conductance_bound <= 0.0:
        raise ValueError("omega and conductance_bound must be positive")
    return omega * GPT2_MATRIX_STD / conductance_bound


def _bind_logical_scale(analog_ctx: Tensor, scale: float) -> None:
    """Bind one canonical logical scale to an analog context."""

    existing = getattr(analog_ctx, "logical_weight_scale", None)
    if existing is not None and not math.isclose(
        float(existing), float(scale), rel_tol=1e-6, abs_tol=1e-12
    ):
        raise ValueError(
            "One analog tile was assigned inconsistent logical scales: "
            f"{existing} and {scale}. Strict tied weights require one scale."
        )
    analog_ctx.logical_weight_scale = float(scale)
    # A time-varying virtual-map multiplier changes both the logical read
    # interpretation and the logical size of one physical update pulse.  It
    # is kept separate from the immutable base scale so checkpoints can
    # validate the coordinate system while the training loop restores the
    # current schedule phase.
    if not hasattr(analog_ctx, "logical_mapping_multiplier"):
        analog_ctx.logical_mapping_multiplier = 1.0
    analog_ctx.logical_gradient_units = True


def rms_normalize(x_input: Tensor, eps: float) -> Tuple[Tensor, Tensor, Tensor]:
    """Return detached per-vector unit-RMS input, inverse scale and active mask."""

    energy = x_input.detach().float().square().mean(dim=-1, keepdim=True)
    active = energy > 0.0
    scale = torch.sqrt(energy + eps).to(dtype=x_input.dtype)
    bare = torch.where(active, x_input.detach() / scale, torch.zeros_like(x_input))
    return bare, scale.detach(), active


def _lambert_w_positive(value: float) -> float:
    """Return the principal Lambert-W branch for a positive scalar."""

    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Lambert-W input must be finite and positive")
    if value < math.e:
        estimate = math.log1p(value)
    else:
        log_value = math.log(value)
        estimate = log_value - math.log(log_value)
    for _ in range(20):
        exponential = math.exp(estimate)
        residual = estimate * exponential - value
        correction = residual / (exponential * (estimate + 1.0))
        estimate -= correction
        if abs(correction) <= 2e-15 * max(1.0, abs(estimate)):
            break
    return estimate


@lru_cache(maxsize=None)
def distribution_optimal_rms_rail(distribution: str, bits: int) -> float:
    """Return a high-resolution MSE-optimal rail in RMS units.

    Gaussian uses the tail-corrected granular/clipping stationarity equation.
    Unit-variance Laplace uses the leading-order closed form
    ``W(3 K^2) / sqrt(2)``. These are distribution-level thresholds; unlike
    :func:`mse_range_normalize`'s adaptive mode they do not use order
    statistics of the observed vector.
    """

    if bits < 2:
        raise ValueError("converter bits must be at least two")
    code_intervals = float(2**int(bits) - 2)
    if distribution == READ_RANGE_LAPLACE:
        return _lambert_w_positive(3.0 * code_intervals**2) / math.sqrt(2.0)
    if distribution != READ_RANGE_GAUSSIAN:
        raise ValueError(f"unsupported distributional range mode: {distribution}")

    def stationarity(rail: float) -> float:
        density = math.exp(-0.5 * rail * rail) / math.sqrt(2.0 * math.pi)
        upper_tail = 0.5 * math.erfc(rail / math.sqrt(2.0))
        clipping_benefit = 2.0 * (density - rail * upper_tail)
        granular_cost = (
            rail
            * (1.0 - 2.0 * upper_tail + rail * density)
            / (3.0 * code_intervals**2)
        )
        return clipping_benefit - granular_cost

    lower = 0.0
    upper = 2.0
    while stationarity(upper) > 0.0:
        upper *= 2.0
        if upper > 64.0:
            raise RuntimeError("could not bracket Gaussian optimal range")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if stationarity(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def aciq_optimal_rms_rail(distribution: str, bits: int) -> float:
    """Return the ACIQ clipping rail in RMS units.

    The Gaussian and Laplace stationary-point rules above are ACIQ's two
    analytical distribution assumptions. This named interface prevents formal
    ACIQ runs from accidentally selecting an observed-vector heuristic.
    """

    if distribution not in ACIQ_DISTRIBUTIONS:
        raise ValueError(
            "ACIQ distribution must be one of "
            f"{sorted(ACIQ_DISTRIBUTIONS)}, got {distribution!r}"
        )
    return distribution_optimal_rms_rail(distribution, bits)


@lru_cache(maxsize=None)
def lp_adaptive_range_parameters(feature_width: int, bits: int) -> Tuple[float, float]:
    """Return the simple bit-adaptive ``(A_b, p_b)`` range approximation.

    The raw-signal rail is ``A_b ||x||_{p_b}``, where

    ``A_b = [1 + (D - 1)/(3 K_b^2)]^{-1}``,
    ``p_b = log(D)/log(1 + 3/K_b)``, and ``K_b = 2^b - 2``.

    This is the closed-form interpolation between finite-p low-resolution
    control and AbsMax at high resolution.  Returning ``math.inf`` represents
    the latter limit and avoids evaluating an unnecessarily large power norm.
    """

    if feature_width < 1:
        raise ValueError("feature width must be positive")
    if bits < 2:
        raise ValueError("converter bits must be at least two")
    code_intervals = float(2**int(bits) - 2)
    shrink = 1.0 / (1.0 + (feature_width - 1.0) / (3.0 * code_intervals**2))
    if feature_width == 1:
        return shrink, math.inf
    order = math.log(float(feature_width)) / math.log1p(3.0 / code_intervals)
    if order > LP_ADAPTIVE_MAX_FINITE_P:
        return shrink, math.inf
    return shrink, order


def mse_range_normalize(
    x_input: Tensor,
    mode: str,
    bits: int,
    input_bound: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Normalize reads using an MSE-oriented effective raw-signal rail.

    ``gaussian`` and ``laplace`` choose a distribution-level RMS threshold.
    ``adaptive`` uses the one-overload high-resolution per-vector optimum

    ``A = absmax / (1 + (D - 1) / (3 K_b^2))``.

    ``lp_adaptive`` uses the closed-form bit-adaptive interpolation

    ``A = A_b ||x||_{p_b}``,
    ``A_b = [1 + (D - 1)/(3 K_b^2)]^{-1}``,
    ``p_b = log(D)/log(1 + 3/K_b)``.

    The inverse scale maps raw rail ``A`` to the tile's existing input bound.
    Values outside ``A`` are deliberately left outside that physical bound
    and are clipped by the native quantizer.
    """

    if mode not in READ_RANGE_MODES - {READ_RANGE_NONE}:
        raise ValueError(f"unsupported MSE range mode: {mode}")
    if bits < 2:
        raise ValueError("converter bits must be at least two")
    if not math.isfinite(input_bound) or input_bound <= 0.0:
        raise ValueError("input_bound must be finite and positive")
    detached = x_input.detach()
    value_float = detached.float()
    absmax = value_float.abs().amax(dim=-1, keepdim=True)
    active = absmax > 0.0
    if mode == READ_RANGE_ADAPTIVE:
        feature_width = int(x_input.size(-1))
        code_intervals = float(2**int(bits) - 2)
        shrink_denominator = 1.0 + (feature_width - 1.0) / (
            3.0 * code_intervals**2
        )
        effective_rail = absmax / shrink_denominator
    elif mode == READ_RANGE_LP_ADAPTIVE:
        shrink, order = lp_adaptive_range_parameters(int(x_input.size(-1)), bits)
        if math.isinf(order):
            norm = absmax
        else:
            # Evaluate the finite-p norm relative to the vector maximum.  The
            # direct float32 expression can overflow at p~80 even for entirely
            # ordinary activation magnitudes (for example 3**82), despite the
            # final norm being finite.  This algebraically identical form has
            # a powered base in [0, 1] and is stable for every finite p used.
            safe_absmax = torch.where(active, absmax, torch.ones_like(absmax))
            normalized = value_float / safe_absmax
            norm = absmax * torch.linalg.vector_norm(
                normalized, ord=order, dim=-1, keepdim=True
            )
        effective_rail = norm * shrink
    else:
        rms = torch.sqrt(
            value_float.square().mean(dim=-1, keepdim=True) + eps
        )
        effective_rail = rms * aciq_optimal_rms_rail(mode, bits)
    safe_rail = torch.where(active, effective_rail, torch.ones_like(effective_rail))
    inverse_scale = (safe_rail / float(input_bound)).to(dtype=x_input.dtype)
    bare = torch.where(active, detached / inverse_scale, torch.zeros_like(detached))
    return bare, inverse_scale.detach(), active, effective_rail.detach()


def interpolated_read_normalize(
    x_input: Tensor,
    interpolation_lambda: float,
    eps: float,
    *,
    l2_endpoint: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Interpolate AbsMax management and explicit FB normalization.

    ``interpolation_lambda=0`` uses the per-vector absolute maximum, matching
    AbsMax noise management once native noise management is disabled.
    ``interpolation_lambda=1`` uses the explicit RMS endpoint, or L2 for the
    language-head backward read. Positive scales are interpolated in log space
    so both endpoints are exact and intermediate scales remain positive.
    """

    lam = float(interpolation_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("interpolation_lambda must lie in [0, 1]")
    detached = x_input.detach()
    value_float = detached.float()
    managed_scale = value_float.abs().amax(dim=-1, keepdim=True)
    active = managed_scale > 0.0
    if l2_endpoint:
        endpoint_scale = torch.sqrt(
            value_float.square().sum(dim=-1, keepdim=True)
        )
    else:
        endpoint_scale = torch.sqrt(
            value_float.square().mean(dim=-1, keepdim=True) + eps
        )
    safe_managed = torch.where(active, managed_scale, torch.ones_like(managed_scale))
    safe_endpoint = torch.where(
        active, endpoint_scale, torch.ones_like(endpoint_scale)
    )
    if lam == 0.0:
        scale = safe_managed.to(dtype=x_input.dtype)
    elif lam == 1.0:
        scale = safe_endpoint.to(dtype=x_input.dtype)
    else:
        scale = torch.exp(
            (1.0 - lam) * torch.log(safe_managed)
            + lam * torch.log(safe_endpoint)
        ).to(dtype=x_input.dtype)
    bare = torch.where(active, detached / scale, torch.zeros_like(detached))
    return bare, scale.detach(), active


def gelu_midpoint_normalize(
    gelu_output: Tensor,
    pre_gelu: Tensor,
    eps: float,
    tail_sigma: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Prepare the static asymmetric GeLU path with unit-RMS bare inputs.

    The raw target interval is ``[GELU(argmin), GELU(tail_sigma*sigma)]``.
    Its midpoint is the zero point, while the scale is chosen from the actual
    centered vector RMS.  Thus the bare input has unit second moment and the
    exact inverse is ``h = scale * bare + midpoint``.
    """

    sigma = pre_gelu.detach().float().std(dim=-1, unbiased=False, keepdim=True)
    upper = F.gelu(tail_sigma * sigma, approximate="none")
    lower = torch.full_like(upper, GELU_OUTPUT_MIN)
    midpoint = ((upper + lower) * 0.5).to(dtype=gelu_output.dtype).detach()
    centered = gelu_output.detach() - midpoint
    energy = centered.float().square().mean(dim=-1, keepdim=True)
    inverse_scale = torch.sqrt(energy + eps).to(dtype=gelu_output.dtype).detach()
    bare = centered / inverse_scale
    active = torch.ones_like(energy, dtype=torch.bool)
    return bare, inverse_scale, midpoint, sigma.detach(), active


class _LogicalAnalogMVMFunction(Function):
    """Custom dense MVM with separate physical-read and logical-update signals."""

    @staticmethod
    @torch.no_grad()
    def forward(
        ctx,
        analog_ctx: Tensor,
        raw_input: Tensor,
        bare_input: Tensor,
        inverse_scale: Tensor,
        zero_point: Tensor,
        active: Tensor,
        output_scale: Tensor,
        learned_output_scale: bool,
        analog_tile,
        use_correction: bool,
        backward_norm: int,
        eps: float,
        owner,
    ) -> Tensor:
        ctx.analog_ctx = analog_ctx
        ctx.analog_tile = analog_tile
        ctx.backward_norm = int(backward_norm)
        ctx.eps = float(eps)
        ctx.owner = owner
        ctx.learned_output_scale = bool(learned_output_scale)

        is_test = not analog_tile.training
        if use_correction:
            feature_width = bare_input.size(-1)
            flat_bare = bare_input.reshape(-1, feature_width)
            ones = torch.ones(
                1,
                feature_width,
                device=bare_input.device,
                dtype=bare_input.dtype,
            )
            combined_input = torch.cat((flat_bare, ones), dim=0)
            combined_output = owner._forward_read(combined_input)
            main_adc = combined_output[:-1].reshape(
                *bare_input.shape[:-1], combined_output.size(-1)
            )
            row_sum_adc = combined_output[-1].reshape(
                *((1,) * (bare_input.dim() - 1)), combined_output.size(-1)
            )
            if owner.exact_forward_row_sum:
                physical_weight, _ = analog_tile.get_weights(
                    apply_weight_scaling=False
                )
                row_sum_adc = physical_weight.sum(dim=1).to(
                    device=bare_input.device,
                    dtype=combined_output.dtype,
                ).reshape(
                    *((1,) * (bare_input.dim() - 1)), combined_output.size(-1)
                )
            processed = inverse_scale * main_adc + zero_point * row_sum_adc
            owner._capture_tensor("row_sum_adc", row_sum_adc)
        else:
            main_adc = owner._forward_read(bare_input)
            processed = inverse_scale * main_adc

        logical_output = output_scale * processed
        logical_output = torch.where(active, logical_output, torch.zeros_like(logical_output))
        # The logical path's scale is a fixed buffer.  The original AIMC path
        # uses AIHWKit's trainable columnwise alpha, whose gradient must remain
        # a normal digital optimizer gradient.
        if ctx.learned_output_scale:
            ctx.save_for_backward(raw_input, output_scale, processed)
        else:
            ctx.save_for_backward(raw_input, output_scale)
        owner._capture_tensor("forward_adc_output", main_adc)
        owner._capture_tensor("processed_output", processed)
        return logical_output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, raw_error: Tensor):
        saved = ctx.saved_tensors
        raw_input, output_scale = saved[:2]
        processed = saved[2] if ctx.learned_output_scale else None

        # The optimizer is defined in logical W units.  Keep the raw logical
        # pair here; the normalized error below is exclusively for the physical
        # transpose read.
        # With fixed s the context stores logical-W gradients. With learned
        # alpha it must instead receive the physical conductance gradient
        # (alpha * e) x^T, matching native AnalogLinear semantics.
        physical_error = (
            raw_error * output_scale if ctx.learned_output_scale else raw_error
        )
        ctx.analog_ctx.accumulate_digital_gradient(raw_input, physical_error)

        detached_error = physical_error.detach()
        if ctx.backward_norm in {BACKWARD_SPLIT_L2, BACKWARD_SPLIT_L2_EXACT}:
            positive_error = detached_error.clamp_min(0.0)
            negative_error = (-detached_error).clamp_min(0.0)
            positive_energy = positive_error.float().square().sum(
                dim=-1, keepdim=True
            )
            negative_energy = negative_error.float().square().sum(
                dim=-1, keepdim=True
            )
            positive_scale, positive_active = _exact_nonzero_scale(
                positive_energy, raw_error.dtype
            )
            negative_scale, negative_active = _exact_nonzero_scale(
                negative_energy, raw_error.dtype
            )
            positive_bare_error = torch.where(
                positive_active,
                positive_error / positive_scale,
                torch.zeros_like(positive_error),
            )
            negative_bare_error = torch.where(
                negative_active,
                negative_error / negative_scale,
                torch.zeros_like(negative_error),
            )

            if ctx.backward_norm == BACKWARD_SPLIT_L2_EXACT:
                # H1: split the signed error into two exact L2-normalized
                # physical reads, but deliberately leave both non-negative
                # branches in the DAC's positive half-range.  There is no
                # affine range remapping and no row-sum correction in this
                # control; it isolates splitting from the sigma-window method.
                feature_width = raw_error.size(-1)
                flat_positive = positive_bare_error.reshape(-1, feature_width)
                flat_negative = negative_bare_error.reshape(-1, feature_width)
                physical_reads = torch.cat((flat_positive, flat_negative), dim=0)
                physical_output = ctx.analog_tile.backward(physical_reads, ctx=None)
                vector_count = flat_positive.size(0)
                output_width = physical_output.size(-1)
                positive_output = physical_output[:vector_count].reshape(
                    *raw_error.shape[:-1], output_width
                )
                negative_output = physical_output[vector_count:].reshape(
                    *raw_error.shape[:-1], output_width
                )
                active = positive_active | negative_active
                grad_input = positive_scale * positive_output - negative_scale * negative_output
                if not ctx.learned_output_scale:
                    grad_input = output_scale * grad_input
                grad_input = torch.where(
                    active, grad_input, torch.zeros_like(grad_input)
                )
                ctx.owner._capture_tensor(
                    "normalized_backward_error_positive", positive_bare_error
                )
                ctx.owner._capture_tensor(
                    "normalized_backward_error_negative", negative_bare_error
                )
                ctx.owner._capture_tensor(
                    "bare_backward_error_positive", positive_bare_error
                )
                ctx.owner._capture_tensor(
                    "bare_backward_error_negative", negative_bare_error
                )
                ctx.owner._capture_tensor("backward_adc_output_positive", positive_output)
                ctx.owner._capture_tensor("backward_adc_output_negative", negative_output)
                ctx.owner._capture_tensor("raw_backward_error", raw_error)
                grad_output_scale = None
                if ctx.learned_output_scale:
                    assert processed is not None
                    grad_output_scale = (raw_error * processed).reshape(
                        -1, processed.size(-1)
                    ).sum(dim=0).view_as(output_scale)
                return (
                    None,
                    grad_input,
                    None,
                    None,
                    None,
                    None,
                    grad_output_scale,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            # The normalized positive branch is concentrated near zero rather
            # than uniformly filling [0, 1].  Match its useful c3-sigma window
            # [0, U+] to the signed DAC range, where U+=min(1,c3*sigma+).
            # Values above U+ are intentionally clipped.  The affine inverse is
            #
            #   W_hat^T z+ ~= U+/2 * (W_hat^T q+ + W_hat^T 1).
            #
            # The negative CE branch is a sparse target coordinate near one.
            # Its calibrated window is [L-,1], L-=max(0,1-c3*sigma-), but
            # structural zeros must stay zero; mapping those zeros to -1 would
            # recreate the dense common-mode read that this split is meant to
            # avoid.  For the one-target CE branch the sole nonzero maps to +1
            # and needs no dense row-sum correction.
            c3 = ctx.owner.backward_window_sigma_value
            positive_sigma = positive_bare_error.float().std(
                dim=-1, unbiased=False, keepdim=True
            ).detach()
            negative_sigma = negative_bare_error.float().std(
                dim=-1, unbiased=False, keepdim=True
            ).detach()
            positive_upper = torch.clamp(c3 * positive_sigma, max=1.0).to(
                dtype=raw_error.dtype
            )
            negative_lower = torch.clamp(1.0 - c3 * negative_sigma, min=0.0).to(
                dtype=raw_error.dtype
            )
            positive_upper_safe = torch.where(
                positive_active,
                positive_upper.clamp_min(torch.finfo(raw_error.dtype).tiny),
                torch.ones_like(positive_upper),
            )
            negative_width_safe = torch.where(
                negative_active,
                (1.0 - negative_lower).clamp_min(
                    torch.finfo(raw_error.dtype).tiny
                ),
                torch.ones_like(negative_lower),
            )
            positive_windowed_error = torch.minimum(
                positive_bare_error, positive_upper_safe
            )
            negative_windowed_error = torch.where(
                negative_bare_error > 0.0,
                torch.maximum(
                    torch.minimum(
                        negative_bare_error,
                        torch.ones_like(negative_bare_error),
                    ),
                    negative_lower,
                ),
                torch.zeros_like(negative_bare_error),
            )
            if ctx.owner.positive_unipolar:
                positive_dac_input = torch.where(
                    positive_active,
                    positive_windowed_error / positive_upper_safe,
                    torch.zeros_like(positive_bare_error),
                )
            else:
                positive_dac_input = torch.where(
                    positive_active,
                    2.0 * positive_windowed_error / positive_upper_safe - 1.0,
                    torch.zeros_like(positive_bare_error),
                )
            negative_support = negative_bare_error > 0.0
            negative_dac_input = torch.where(
                negative_support,
                2.0
                * (negative_windowed_error - negative_lower)
                / negative_width_safe
                - 1.0,
                torch.zeros_like(negative_bare_error),
            )
            positive_gain = torch.where(
                positive_active, positive_scale, torch.zeros_like(positive_scale)
            )
            negative_gain = torch.where(
                negative_active, negative_scale, torch.zeros_like(negative_scale)
            )

            feature_width = raw_error.size(-1)
            flat_positive = positive_dac_input.reshape(-1, feature_width)
            flat_negative = negative_dac_input.reshape(-1, feature_width)
            vector_count = flat_positive.size(0)
            narrow_tile = ctx.owner.narrow_backward_tile
            if ctx.owner.positive_unipolar:
                positive_backward_adc = ctx.analog_tile.backward(
                    flat_positive, ctx=None
                ).reshape(*raw_error.shape[:-1], -1)
                negative_tile = (
                    narrow_tile if narrow_tile is not None else ctx.analog_tile
                )
                negative_backward_adc = negative_tile.backward(
                    flat_negative, ctx=None
                ).reshape(*raw_error.shape[:-1], -1)
                correction_input = None
                correction_backward_adc = None
            elif narrow_tile is None:
                correction_input = torch.ones(
                    1,
                    feature_width,
                    device=raw_error.device,
                    dtype=raw_error.dtype,
                )
                combined_bare_error = torch.cat(
                    (flat_positive, flat_negative, correction_input), dim=0
                )
                combined_backward_adc = ctx.analog_tile.backward(
                    combined_bare_error, ctx=None
                )
                adc_width = combined_backward_adc.size(-1)
                positive_backward_adc = combined_backward_adc[:vector_count].reshape(
                    *raw_error.shape[:-1], adc_width
                )
                negative_backward_adc = combined_backward_adc[
                    vector_count : 2 * vector_count
                ].reshape(
                    *raw_error.shape[:-1], adc_width
                )
                correction_backward_adc = combined_backward_adc[-1].reshape(
                    *((1,) * (raw_error.dim() - 1)), adc_width
                )
            else:
                # The positive affine read and its dense W^T 1 correction need
                # the O(sqrt(V)) ADC range.  The target branch is one-hot and
                # therefore uses the synchronized O(1)-range read replica.
                correction_input = torch.ones(
                    1,
                    feature_width,
                    device=raw_error.device,
                    dtype=raw_error.dtype,
                )
                wide_bare_error = torch.cat(
                    (flat_positive, correction_input), dim=0
                )
                wide_backward_adc = ctx.analog_tile.backward(
                    wide_bare_error, ctx=None
                )
                narrow_negative_adc = narrow_tile.backward(
                    flat_negative, ctx=None
                )
                adc_width = wide_backward_adc.size(-1)
                positive_backward_adc = wide_backward_adc[:vector_count].reshape(
                    *raw_error.shape[:-1], adc_width
                )
                negative_backward_adc = narrow_negative_adc.reshape(
                    *raw_error.shape[:-1], adc_width
                )
                correction_backward_adc = wide_backward_adc[-1].reshape(
                    *((1,) * (raw_error.dim() - 1)), adc_width
                )
            if ctx.owner.positive_unipolar:
                grad_input = (
                    positive_gain
                    * positive_upper_safe
                    * positive_backward_adc
                    - negative_gain * negative_backward_adc
                )
            else:
                positive_half_range = 0.5 * positive_upper_safe
                grad_input = (
                    positive_gain
                    * positive_half_range
                    * (positive_backward_adc + correction_backward_adc)
                    - negative_gain * negative_backward_adc
                )
            if not ctx.learned_output_scale:
                grad_input = output_scale * grad_input
            active = positive_active | negative_active
            grad_input = torch.where(active, grad_input, torch.zeros_like(grad_input))

            ctx.owner._capture_tensor(
                "normalized_backward_error_positive", positive_bare_error
            )
            ctx.owner._capture_tensor(
                "normalized_backward_error_negative", negative_bare_error
            )
            ctx.owner._capture_tensor(
                "backward_error_sigma_positive", positive_sigma
            )
            ctx.owner._capture_tensor(
                "backward_error_sigma_negative", negative_sigma
            )
            ctx.owner._capture_tensor(
                "backward_window_upper_positive", positive_upper
            )
            ctx.owner._capture_tensor(
                "backward_window_lower_negative", negative_lower
            )
            ctx.owner._capture_tensor(
                "windowed_backward_error_positive", positive_windowed_error
            )
            ctx.owner._capture_tensor(
                "windowed_backward_error_negative", negative_windowed_error
            )
            ctx.owner._capture_tensor(
                "bare_backward_error_positive", positive_dac_input
            )
            ctx.owner._capture_tensor(
                "bare_backward_error_negative", negative_dac_input
            )
            ctx.owner._capture_tensor("bare_backward_error_correction", correction_input)
            ctx.owner._capture_tensor(
                "backward_adc_output_positive", positive_backward_adc
            )
            ctx.owner._capture_tensor(
                "backward_adc_output_negative", negative_backward_adc
            )
            ctx.owner._capture_tensor(
                "backward_adc_output_correction", correction_backward_adc
            )
        else:
            error_float = detached_error.float()
            effective_rail = None
            if ctx.owner.read_range_enabled:
                bare_error, error_scale, active, effective_rail = mse_range_normalize(
                    detached_error,
                    ctx.owner.read_range_mode,
                    ctx.owner.backward_dac_bits,
                    ctx.owner.backward_input_bound_value,
                    ctx.eps,
                )
            elif ctx.owner.read_interpolation_enabled:
                bare_error, error_scale, active = interpolated_read_normalize(
                    detached_error,
                    ctx.owner.read_interpolation_lambda_value,
                    ctx.eps,
                    l2_endpoint=ctx.backward_norm == BACKWARD_L2,
                )
            else:
                if ctx.backward_norm == BACKWARD_IDENTITY:
                    energy = torch.ones_like(error_float[..., :1])
                elif ctx.backward_norm == BACKWARD_L2:
                    energy = error_float.square().sum(dim=-1, keepdim=True)
                else:
                    energy = error_float.square().mean(dim=-1, keepdim=True)
                error_scale, active = _exact_nonzero_scale(energy, raw_error.dtype)
                bare_error = torch.where(
                    active,
                    detached_error / error_scale,
                    torch.zeros_like(raw_error),
                )
            backward_adc = ctx.analog_tile.backward(bare_error, ctx=None)
            grad_input = error_scale * backward_adc
            if not ctx.learned_output_scale:
                grad_input = output_scale * grad_input
            grad_input = torch.where(active, grad_input, torch.zeros_like(grad_input))

            ctx.owner._capture_tensor("bare_backward_error", bare_error)
            ctx.owner._capture_tensor("backward_adc_output", backward_adc)
            ctx.owner._capture_tensor("effective_backward_input_rail", effective_rail)

        ctx.owner._capture_tensor("raw_backward_error", raw_error)
        grad_output_scale = None
        if ctx.learned_output_scale:
            assert processed is not None
            grad_output_scale = (raw_error * processed).reshape(
                -1, processed.size(-1)
            ).sum(dim=0).view_as(output_scale)
        return (
            None,
            grad_input,
            None,
            None,
            None,
            None,
            grad_output_scale,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _LogicalAnalogEmbeddingFunction(Function):
    """One-hot forward MVM with a logical-gradient-only update trace."""

    @staticmethod
    @torch.no_grad()
    def forward(ctx, analog_ctx, one_hot, output_scale, analog_tile, owner):
        ctx.analog_ctx = analog_ctx
        ctx.analog_tile = analog_tile
        ctx.owner = owner
        ctx.save_for_backward(one_hot)
        adc_output = analog_tile.joint_forward(
            one_hot, is_test=not analog_tile.training, ctx=None
        )
        owner._capture_tensor("bare_input", one_hot)
        owner._capture_tensor("forward_adc_output", adc_output)
        return output_scale * adc_output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, raw_error):
        (one_hot,) = ctx.saved_tensors
        ctx.analog_ctx.accumulate_digital_gradient(one_hot, raw_error)
        ctx.owner._capture_tensor("raw_backward_error", raw_error)
        return None, None, None, None, None


class _DiagnosticMixin:
    """Small opt-in capture used by calibration and plotting scripts."""

    def _init_diagnostics(self) -> None:
        self._diagnostics_enabled = False
        self._diagnostic_max_vectors = 128
        self._diagnostics: Dict[str, Tensor] = {}

    def enable_diagnostics(self, max_vectors: int = 128) -> None:
        self._diagnostics_enabled = True
        self._diagnostic_max_vectors = int(max_vectors)
        self._diagnostics = {}

    def disable_diagnostics(self) -> None:
        self._diagnostics_enabled = False

    @property
    def diagnostics(self) -> Dict[str, Tensor]:
        return dict(self._diagnostics)

    @torch.no_grad()
    def _capture_tensor(self, name: str, value: Optional[Tensor]) -> None:
        if not self._diagnostics_enabled or value is None:
            return
        tensor = value.detach().float()
        if tensor.dim() == 0:
            sample = tensor.reshape(1)
        elif tensor.dim() == 1:
            sample = tensor[: self._diagnostic_max_vectors]
        else:
            sample = tensor.reshape(-1, tensor.size(-1))[: self._diagnostic_max_vectors]
        self._diagnostics[name] = sample.cpu()


class LogicalAnalogLinear(_DiagnosticMixin, nn.Module):
    """AnalogLinear wrapper implementing logical scaling and normalized reads."""

    def __init__(
        self,
        analog_linear: AnalogLinear,
        output_scale: float,
        omega: float,
        forward_mode: str = FORWARD_RMS,
        backward_mode: int = BACKWARD_RMS,
        eps: float = 1e-8,
        tail_sigma: float = 3.0,
        backward_window_sigma: float = 3.0,
        forward_input_bound: float = 3.0,
        forward_output_bound: float = 3.0,
        backward_input_bound: float = 3.0,
        backward_output_bound: float = 3.0,
        layer_name: str = "",
        learned_output_scale: bool = False,
        learnable_scale_gain: bool = False,
        narrow_backward_output_bound: Optional[float] = None,
        positive_unipolar: bool = False,
        read_interpolation_lambda: Optional[float] = None,
        read_range_mode: str = READ_RANGE_NONE,
        forward_dac_bits: int = 7,
        backward_dac_bits: int = 7,
    ):
        super().__init__()
        if not isinstance(analog_linear, AnalogLinear):
            raise TypeError("LogicalAnalogLinear requires an AnalogLinear child")
        if forward_mode not in {FORWARD_RMS, FORWARD_GELU, FORWARD_IDENTITY}:
            raise ValueError(f"Unknown forward mode: {forward_mode}")
        if backward_mode not in {
            BACKWARD_RMS, BACKWARD_L2, BACKWARD_SPLIT_L2,
            BACKWARD_IDENTITY, BACKWARD_SPLIT_L2_EXACT,
        }:
            raise ValueError(f"Unknown backward mode: {backward_mode}")

        analog_tile = analog_linear.analog_module
        if analog_tile.analog_bias:
            raise ValueError("LogicalAnalogLinear requires a digital, not analog, bias")

        self.analog_linear = analog_linear
        self.forward_mode = forward_mode
        self.backward_mode = int(backward_mode)
        self.eps = float(eps)
        self.tail_sigma = float(tail_sigma)
        if backward_window_sigma <= 0.0:
            raise ValueError("backward_window_sigma must be positive")
        self.backward_window_sigma_value = float(backward_window_sigma)
        self.layer_name = layer_name
        self.learned_output_scale = bool(learned_output_scale)
        self.learnable_scale_gain = bool(learnable_scale_gain)
        if self.learned_output_scale and self.learnable_scale_gain:
            raise ValueError(
                "AIHWKit learned output scaling and a logical learnable scale "
                "gain are mutually exclusive"
            )
        self.positive_unipolar = bool(positive_unipolar)
        self.read_interpolation_enabled = read_interpolation_lambda is not None
        self.read_interpolation_lambda_value = (
            float(read_interpolation_lambda)
            if read_interpolation_lambda is not None
            else math.nan
        )
        if self.read_interpolation_enabled and not (
            0.0 <= self.read_interpolation_lambda_value <= 1.0
        ):
            raise ValueError("read_interpolation_lambda must lie in [0, 1]")
        if self.read_interpolation_enabled and (
            self.forward_mode == FORWARD_GELU
            or self.backward_mode in {
                BACKWARD_SPLIT_L2,
                BACKWARD_SPLIT_L2_EXACT,
            }
        ):
            raise ValueError(
                "read interpolation supports ordinary/RMS forward and "
                "ordinary RMS/L2 backward reads only"
            )
        if read_range_mode not in READ_RANGE_MODES:
            raise ValueError(f"Unknown read range mode: {read_range_mode}")
        if self.read_interpolation_enabled and read_range_mode != READ_RANGE_NONE:
            raise ValueError("read interpolation and MSE range management are exclusive")
        if forward_dac_bits < 2 or backward_dac_bits < 2:
            raise ValueError("forward/backward DAC bits must be at least two")
        self.read_range_mode = str(read_range_mode)
        self.read_range_enabled = self.read_range_mode != READ_RANGE_NONE
        self.forward_dac_bits = int(forward_dac_bits)
        self.backward_dac_bits = int(backward_dac_bits)
        self.exact_forward_row_sum = False
        if self.positive_unipolar and self.backward_mode != BACKWARD_SPLIT_L2:
            raise ValueError(
                "Unipolar positive mapping requires sigma-window split backward"
            )
        self.narrow_read_analog_linear: Optional[AnalogLinear] = None
        if self.learned_output_scale:
            if analog_tile.out_scaling_alpha is None:
                raise ValueError(
                    "ExplicitPeripheryAnalogLinear requires learned output scaling"
                )
        else:
            self.register_buffer("output_scale", torch.tensor(float(output_scale)))
        # Nandakumar et al.'s PCM experiment initially interpreted one fixed
        # conductance interval as a narrower logical weight interval, then
        # relaxed that interval over epochs 1--3.  Keep this factor separate
        # from ``output_scale``: the latter defines the fixed logical-W
        # coordinate, optimizer moments, and pulse residual, whereas this
        # buffer changes only the read interpretation.
        self.register_buffer("mapping_warmup_multiplier", torch.ones(1))
        if self.learnable_scale_gain:
            # Parameterize the multiplicative gain in log space so that the
            # effective scale s_eff = s_0 * exp(theta) stays positive.  The
            # fixed base scale s_0 continues to define LogicalMpAdamW's pulse
            # quantum and pending-residual units.
            self.logical_log_scale_gain = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("logical_log_scale_gain", None)
        self.register_buffer("omega", torch.tensor(float(omega)))
        self.register_buffer(
            "backward_window_sigma", torch.tensor(float(backward_window_sigma))
        )
        self.register_buffer("forward_input_bound", torch.tensor(float(forward_input_bound)))
        self.register_buffer("forward_output_bound", torch.tensor(float(forward_output_bound)))
        self.register_buffer("backward_input_bound", torch.tensor(float(backward_input_bound)))
        self.register_buffer("backward_output_bound", torch.tensor(float(backward_output_bound)))
        # The registered bounds move to CUDA with the module. Converting those
        # device scalars to float on every read would force a host sync, so the
        # range normalizer uses immutable Python copies instead.
        self.forward_input_bound_value = float(forward_input_bound)
        self.backward_input_bound_value = float(backward_input_bound)
        if not self.learned_output_scale:
            _bind_logical_scale(analog_tile.analog_ctx, output_scale)

        if narrow_backward_output_bound is not None:
            if self.learned_output_scale:
                raise ValueError("A dual-range read tile requires fixed logical scaling")
            if self.backward_mode != BACKWARD_SPLIT_L2:
                raise ValueError(
                    "A dual-range read tile is only valid for sigma-window split backward"
                )
            if narrow_backward_output_bound <= 0.0:
                raise ValueError("narrow_backward_output_bound must be positive")
            narrow_config = deepcopy(analog_tile.rpu_config)
            narrow_config.backward.inp_bound = 1.0
            narrow_config.backward.out_bound = float(narrow_backward_output_bound)
            narrow_linear = AnalogLinear(
                analog_linear.in_features,
                analog_linear.out_features,
                bias=False,
                rpu_config=narrow_config,
            )
            # This is a synchronized read-only replica.  It must never be
            # regrouped into LogicalMpAdamW or receive an independent update.
            narrow_linear.analog_module.analog_ctx.requires_grad_(False)
            self.narrow_read_analog_linear = narrow_linear
        self._init_diagnostics()

    @property
    def narrow_backward_tile(self):
        if self.narrow_read_analog_linear is None:
            return None
        return self.narrow_read_analog_linear.analog_module

    @property
    def in_features(self) -> int:
        return self.analog_linear.in_features

    @property
    def out_features(self) -> int:
        return self.analog_linear.out_features

    @property
    def bias(self):
        tile = self.analog_linear.analog_module
        return tile.bias if tile.digital_bias else None

    def _logical_mvm(
        self,
        raw_input: Tensor,
        bare_input: Tensor,
        inverse_scale: Tensor,
        zero_point: Tensor,
        active: Tensor,
        use_correction: bool,
    ) -> Tensor:
        analog_tile = self.analog_linear.analog_module
        output_scale = (
            analog_tile.out_scaling_alpha
            if self.learned_output_scale
            else self.output_scale * self.mapping_warmup_multiplier
        )
        output = _LogicalAnalogMVMFunction.apply(
            analog_tile.analog_ctx,
            raw_input,
            bare_input,
            inverse_scale,
            zero_point,
            active,
            output_scale,
            self.learned_output_scale,
            analog_tile,
            use_correction,
            self.backward_mode,
            self.eps,
            self,
        )
        if self.logical_log_scale_gain is not None:
            output = output * self.logical_log_scale_gain.exp()
        if self.bias is not None:
            output = output + self.bias.view(*((1,) * (output.dim() - 1)), -1)
        return output

    def _forward_read(self, x_input: Tensor) -> Tensor:
        tile = self.analog_linear.analog_module
        mode = getattr(self, "channel_recompute_mode", "native")
        if mode == "native":
            return tile.joint_forward(x_input, is_test=not tile.training, ctx=None)
        from channel_recompute import managed_forward
        return managed_forward(tile, x_input, mode,
                               chunk_rows=self.channel_recompute_chunk_rows)

    def forward(self, x_input: Tensor) -> Tensor:
        if self.forward_mode == FORWARD_GELU:
            raise RuntimeError(
                f"{self.layer_name or 'GeLU projection'} requires forward_from_gelu(h, a)"
            )
        effective_rail = None
        if self.read_range_enabled:
            bare, inverse_scale, active, effective_rail = mse_range_normalize(
                x_input,
                self.read_range_mode,
                self.forward_dac_bits,
                self.forward_input_bound_value,
                self.eps,
            )
        elif self.read_interpolation_enabled:
            bare, inverse_scale, active = interpolated_read_normalize(
                x_input,
                self.read_interpolation_lambda_value,
                self.eps,
            )
        elif self.forward_mode == FORWARD_IDENTITY:
            bare = x_input.detach()
            inverse_scale = torch.ones_like(x_input[..., :1])
            active = torch.ones_like(x_input[..., :1], dtype=torch.bool)
        else:
            bare, inverse_scale, active = rms_normalize(x_input, self.eps)
        zero_point = torch.zeros_like(inverse_scale)
        self._capture_tensor("raw_input", x_input)
        self._capture_tensor("bare_input", bare)
        self._capture_tensor("inverse_scale", inverse_scale)
        self._capture_tensor("effective_input_rail", effective_rail)
        return self._logical_mvm(
            x_input, bare, inverse_scale, zero_point, active, use_correction=False
        )

    def forward_from_gelu(self, gelu_output: Tensor, pre_gelu: Tensor) -> Tensor:
        if self.forward_mode != FORWARD_GELU:
            return self.forward(gelu_output)
        bare, inverse_scale, midpoint, sigma, active = gelu_midpoint_normalize(
            gelu_output,
            pre_gelu,
            self.eps,
            self.tail_sigma,
        )
        self._capture_tensor("raw_input", gelu_output)
        self._capture_tensor("bare_input", bare)
        self._capture_tensor("inverse_scale", inverse_scale)
        self._capture_tensor("zero_point", midpoint)
        self._capture_tensor("pre_gelu_sigma", sigma)
        return self._logical_mvm(
            gelu_output,
            bare,
            inverse_scale,
            midpoint,
            active,
            use_correction=True,
        )


class ExplicitPeripheryAnalogLinear(LogicalAnalogLinear):
    """F/B/G wrapper retaining native MpAdamW and learned AIHWKit alpha.

    Unlike :class:`LogicalAnalogLinear`, this wrapper does not introduce a
    logical reparameterization.  Conductance updates remain native MpAdamW
    updates, while the trainable columnwise ``out_scaling_alpha`` is applied
    explicitly and receives its normal digital gradient.
    """

    def __init__(self, analog_linear: AnalogLinear, **kwargs):
        super().__init__(
            analog_linear,
            output_scale=1.0,
            omega=1.0,
            learned_output_scale=True,
            **kwargs,
        )


class AnalogEmbedding(_DiagnosticMixin, nn.Module):
    """Embedding lookup implemented as a one-hot analog forward MVM."""

    def __init__(
        self,
        analog_linear: AnalogLinear,
        num_embeddings: int,
        output_scale: float = 1.0,
        omega: float = 1.0,
        layer_name: str = "transformer.wpe",
    ):
        super().__init__()
        if not isinstance(analog_linear, AnalogLinear):
            raise TypeError("AnalogEmbedding requires an AnalogLinear child")
        if analog_linear.in_features != num_embeddings:
            raise ValueError("AnalogEmbedding input width must equal num_embeddings")
        if analog_linear.bias not in (False, None):
            raise ValueError("AnalogEmbedding requires a bias-free AnalogLinear")
        self.analog_linear = analog_linear
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = analog_linear.out_features
        self.layer_name = layer_name
        self.register_buffer("output_scale", torch.tensor(float(output_scale)))
        self.register_buffer("omega", torch.tensor(float(omega)))
        _bind_logical_scale(analog_linear.analog_module.analog_ctx, output_scale)
        self._init_diagnostics()

    def forward(self, token_ids: Tensor) -> Tensor:
        one_hot = F.one_hot(token_ids, num_classes=self.num_embeddings).to(
            dtype=self.analog_linear.analog_module.analog_ctx.dtype
        )
        return _LogicalAnalogEmbeddingFunction.apply(
            self.analog_linear.analog_module.analog_ctx,
            one_hot,
            self.output_scale,
            self.analog_linear.analog_module,
            self,
        )


def wrap_scaled_analog_linears(
    module: nn.Module,
    n_layer: int,
    omega: float,
    conductance_bound: float,
    eps: float,
    tail_sigma: float,
    backward_window_sigma: float = 3.0,
    prefix: str = "",
    lm_head_backward_split: bool = False,
    lm_head_exact_split: bool = False,
    lm_head_dual_adc_ranges: bool = False,
    lm_head_narrow_backward_output_bound: Optional[float] = None,
    lm_head_positive_unipolar: bool = False,
    logical_init_mode: str = LOGICAL_INIT_WIDTH_STABLE,
    ordinary_forward_mode: str = FORWARD_RMS,
    gelu_forward_mode: str = FORWARD_GELU,
    ordinary_backward_mode: int = BACKWARD_RMS,
    read_interpolation_lambda: Optional[float] = None,
    read_range_mode: str = READ_RANGE_NONE,
    forward_dac_bits: int = 7,
    backward_dac_bits: int = 7,
    learnable_scale_gain: bool = False,
    scale_rule: str = "variance",
    fixed_logical_scale: Optional[float] = None,
) -> nn.Module:
    """Recursively wrap every existing AnalogLinear with the v0 algorithm."""

    if scale_rule not in {"variance", "rasch_absmax", "fixed"}:
        raise ValueError("scale_rule must be variance, rasch_absmax, or fixed")
    if scale_rule == "fixed" and (
        fixed_logical_scale is None or fixed_logical_scale <= 0.0
    ):
        raise ValueError("fixed scale_rule requires a positive fixed_logical_scale")

    if lm_head_dual_adc_ranges:
        if not lm_head_backward_split or lm_head_exact_split:
            raise ValueError(
                "Dual lm-head ADC ranges require the non-exact sigma-window split"
            )
        if (
            lm_head_narrow_backward_output_bound is None
            or lm_head_narrow_backward_output_bound <= 0.0
        ):
            raise ValueError(
                "Dual lm-head ADC ranges require a positive narrow output bound"
            )
    if lm_head_positive_unipolar and (
        not lm_head_backward_split or lm_head_exact_split
    ):
        raise ValueError(
            "Unipolar lm-head positive mapping requires the non-exact "
            "sigma-window split"
        )

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, AnalogLinear):
            is_residual = full_name.endswith("c_proj")
            is_gelu_projection = full_name.endswith("mlp.c_proj")
            is_lm_head = full_name == "lm_head"
            if is_lm_head:
                backward_mode = BACKWARD_SPLIT_L2_EXACT if lm_head_exact_split else (
                    BACKWARD_SPLIT_L2 if lm_head_backward_split else BACKWARD_L2
                )
            else:
                backward_mode = ordinary_backward_mode
            scale = (
                float(fixed_logical_scale)
                if scale_rule == "fixed"
                else fixed_output_scale(
                    n_layer,
                    child.in_features,
                    is_residual,
                    omega,
                    conductance_bound,
                    logical_init_mode,
                )
            )
            if scale_rule == "rasch_absmax":
                # Preserve precisely the logical initialization used by S,
                # then remap that same W to Rasch's fixed virtual range:
                # W = a W_hat, a=max(abs(W))/conductance_bound.  Thereafter
                # a stays fixed, so the comparison changes only the scale
                # rule, not logical initialization, AdamW, or read handling.
                physical_weight, physical_bias = child.get_weights(
                    apply_weight_scaling=False
                )
                logical_weight = physical_weight * scale
                absmax = float(logical_weight.abs().max().item())
                if not math.isfinite(absmax) or absmax <= 0.0:
                    raise RuntimeError(
                        f"{full_name}: invalid logical AbsMax for Rasch mapping"
                    )
                scale = absmax / conductance_bound
                child.set_weights(
                    logical_weight / scale,
                    physical_bias,
                    apply_weight_scaling=False,
                )
            rpu_config = child.analog_module.rpu_config
            wrapped = LogicalAnalogLinear(
                child,
                output_scale=scale,
                omega=omega,
                forward_mode=gelu_forward_mode if is_gelu_projection else ordinary_forward_mode,
                backward_mode=backward_mode,
                eps=eps,
                tail_sigma=tail_sigma,
                backward_window_sigma=backward_window_sigma,
                forward_input_bound=rpu_config.forward.inp_bound,
                forward_output_bound=rpu_config.forward.out_bound,
                backward_input_bound=rpu_config.backward.inp_bound,
                backward_output_bound=rpu_config.backward.out_bound,
                layer_name=full_name,
                narrow_backward_output_bound=(
                    lm_head_narrow_backward_output_bound
                    if is_lm_head and lm_head_dual_adc_ranges
                    else None
                ),
                positive_unipolar=(
                    lm_head_positive_unipolar if is_lm_head else False
                ),
                read_interpolation_lambda=read_interpolation_lambda,
                read_range_mode=read_range_mode,
                forward_dac_bits=forward_dac_bits,
                backward_dac_bits=backward_dac_bits,
                learnable_scale_gain=learnable_scale_gain,
            )
            wrapped.scale_rule = scale_rule
            setattr(module, child_name, wrapped)
        else:
            wrap_scaled_analog_linears(
                child,
                n_layer,
                omega,
                conductance_bound,
                eps,
                tail_sigma,
                backward_window_sigma,
                prefix=full_name,
                lm_head_backward_split=lm_head_backward_split,
                lm_head_exact_split=lm_head_exact_split,
                lm_head_dual_adc_ranges=lm_head_dual_adc_ranges,
                lm_head_narrow_backward_output_bound=(
                    lm_head_narrow_backward_output_bound
                ),
                lm_head_positive_unipolar=lm_head_positive_unipolar,
                logical_init_mode=logical_init_mode,
                ordinary_forward_mode=ordinary_forward_mode,
                gelu_forward_mode=gelu_forward_mode,
                ordinary_backward_mode=ordinary_backward_mode,
                read_interpolation_lambda=read_interpolation_lambda,
                read_range_mode=read_range_mode,
                forward_dac_bits=forward_dac_bits,
                backward_dac_bits=backward_dac_bits,
                learnable_scale_gain=learnable_scale_gain,
                scale_rule=scale_rule,
                fixed_logical_scale=fixed_logical_scale,
            )
    return module


@torch.no_grad()
def synchronize_dual_range_read_tiles(module: nn.Module) -> int:
    """Copy each updated main conductance matrix to its narrow read replica.

    The replica is deliberately read-only and excluded from the optimizer.  An
    exact copy models a second read array whose conductances are kept in sync
    with the trainable head array; it does not claim a zero-cost hardware
    implementation of that synchronization.
    """

    synchronized = 0
    for child in module.modules():
        if not isinstance(child, LogicalAnalogLinear):
            continue
        narrow = child.narrow_read_analog_linear
        if narrow is None:
            continue
        main_tile = child.analog_linear.analog_module
        narrow_tile = narrow.analog_module
        physical_weight, _ = child.analog_linear.get_weights(
            apply_weight_scaling=False
        )
        narrow_device = narrow_tile.analog_ctx.device
        # Device-to-device bound variation can otherwise clip an exact copied
        # conductance differently on the replica.  Copy the latent device
        # parameters first so the controlled comparison changes only ADC IO.
        hidden_parameters = {
            name: value.to(device=narrow_device)
            for name, value in main_tile.get_hidden_parameters().items()
        }
        narrow_tile.set_hidden_parameters(hidden_parameters)
        narrow_tile.set_hidden_update_index(main_tile.get_hidden_update_index())
        narrow.set_weights(
            physical_weight.to(device=narrow_device),
            apply_weight_scaling=False,
        )
        synchronized += 1
    return synchronized


def wrap_explicit_periphery_analog_linears(
    module: nn.Module,
    eps: float,
    tail_sigma: float,
    prefix: str = "",
    ordinary_forward_mode: str = FORWARD_RMS,
    gelu_forward_mode: str = FORWARD_GELU,
    ordinary_backward_mode: int = BACKWARD_RMS,
    gelu_only: bool = False,
) -> nn.Module:
    """Wrap analog linears for the no-S F/FB/FBG/G controls.

    The wrapper changes only selected physical read paths.  It deliberately
    retains native conductance initialization, MpAdamW, and AIHWKit's learned
    columnwise output scaling, so these profiles identify F/B/G effects that
    are not conditional on the composite S component.  ``gelu_only`` leaves
    every matrix except ``mlp.c_proj`` untouched for the isolated G profile.
    """

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, AnalogLinear):
            is_gelu_projection = full_name.endswith("mlp.c_proj")
            if gelu_only and not is_gelu_projection:
                continue
            rpu_config = child.analog_module.rpu_config
            wrapped = ExplicitPeripheryAnalogLinear(
                child,
                forward_mode=(
                    gelu_forward_mode if is_gelu_projection else ordinary_forward_mode
                ),
                backward_mode=(
                    BACKWARD_L2 if full_name == "lm_head" else ordinary_backward_mode
                ),
                eps=eps,
                tail_sigma=tail_sigma,
                forward_input_bound=rpu_config.forward.inp_bound,
                forward_output_bound=rpu_config.forward.out_bound,
                backward_input_bound=rpu_config.backward.inp_bound,
                backward_output_bound=rpu_config.backward.out_bound,
                layer_name=full_name,
            )
            setattr(module, child_name, wrapped)
        else:
            wrap_explicit_periphery_analog_linears(
                child,
                eps=eps,
                tail_sigma=tail_sigma,
                prefix=full_name,
                ordinary_forward_mode=ordinary_forward_mode,
                gelu_forward_mode=gelu_forward_mode,
                ordinary_backward_mode=ordinary_backward_mode,
                gelu_only=gelu_only,
            )
    return module


def bind_model_logical_scales(module: nn.Module) -> int:
    """Rebind checkpointed wrapper scales to the live AnalogContexts.

    AIHWKit can recreate an ``AnalogContext`` while loading tile state.  The
    scale is a registered wrapper buffer and survives normally, whereas the
    context marker is intentionally Python-only.  Call this after model state
    loading/device placement and before constructing ``LogicalMpAdamW``.
    """

    bound_contexts = set()
    for child in module.modules():
        if isinstance(child, LogicalAnalogLinear):
            analog_ctx = child.analog_linear.analog_module.analog_ctx
            scale = float(child.output_scale.detach().cpu())
        elif isinstance(child, AnalogEmbedding):
            analog_ctx = child.analog_linear.analog_module.analog_ctx
            scale = float(child.output_scale.detach().cpu())
        else:
            continue
        _bind_logical_scale(analog_ctx, scale)
        bound_contexts.add(id(analog_ctx))
    return len(bound_contexts)


@torch.no_grad()
def set_mapping_warmup_multiplier(module: nn.Module, multiplier: float) -> int:
    """Set a common logical-map multiplier on reads and physical transfers.

    The physical tile and optimizer moments/residual are deliberately
    untouched.  The tied token lookup obtains its scale from the shared
    ``lm_head`` wrapper, while LogicalMpAdamW reads the mirrored context
    attribute when converting logical updates to physical pulses.
    """
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("mapping warm-up multiplier must be finite and positive")
    wrappers = [child for child in module.modules() if isinstance(child, LogicalAnalogLinear)]
    if not wrappers:
        raise RuntimeError("mapping warm-up requires at least one LogicalAnalogLinear")
    for child in wrappers:
        child.mapping_warmup_multiplier.fill_(float(multiplier))
        child.analog_linear.analog_module.analog_ctx.logical_mapping_multiplier = float(
            multiplier
        )
    return len(wrappers)


# New descriptive alias; keep the original function name for the paused
# prototype's import surface.
wrap_logical_analog_linears = wrap_scaled_analog_linears
