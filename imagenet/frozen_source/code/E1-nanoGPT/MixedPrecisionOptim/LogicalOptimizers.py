"""Alternative logical-space optimizers for open-loop AIMC programming.

Each class preserves :class:`LogicalMpAdamW`'s crucial invariant: an
``AnalogContext`` stores a residual and every optimizer update is expressed in
logical-weight units.  The residual is then transferred as a quantized physical
array command by the inherited open-loop transfer routine.  No optimizer in
this module reads an analog weight to form its update.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor
from torch.autograd import no_grad
from torch.optim import Optimizer

from aihwkit.optim.context import AnalogContext

from MixedPrecisionOptim.LogicalAdamW import LogicalMpAdamW
from MixedPrecisionOptim.Muon import muon_update


class _LogicalOptimizer(LogicalMpAdamW):
    """Shared logical-buffer bookkeeping with a pluggable first-order rule."""

    def _ddp_step_protocol_is_supported(self) -> bool:
        """Accept only subclasses that retain this centralized DDP step.

        The inherited transfer routine updates a physical AIMC tile only on
        rank 0, while ``_ddp_finish_step`` broadcasts the resulting complete
        simulator state to every follower.  A subclass that implements its
        own ``step`` must explicitly integrate those two operations rather
        than being admitted to DDP by inheritance alone.
        """

        # ``Optimizer._patch_step_function`` wraps ``type(self).step`` after
        # construction for profiler hooks.  Compare the unwrapped function so
        # that the hook itself is not mistaken for a subclass implementation.
        step_impl = type(self).step
        while hasattr(step_impl, "__wrapped__"):
            step_impl = step_impl.__wrapped__
        return step_impl is _LogicalOptimizer.step

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore generic optimizer state without AdamW-only assumptions.

        ``LogicalMpAdamW.load_state_dict`` deliberately supplies the common
        logical-residual validation, but PyTorch dispatches ``__setstate__``
        dynamically.  Alternative rules such as SGD do not carry Adam's
        mandatory ``step`` entry, so their deserialization must use the base
        Optimizer implementation.
        """
        Optimizer.__setstate__(self, state)

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        raise NotImplementedError

    @no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None, **_: Any
    ) -> Optional[float]:
        # AnalogContexts are excluded from PyTorch DDP's reducer.  Their
        # backward-time digital gradients must therefore be averaged by the
        # AIMC protocol before any logical residual or physical tile changes.
        self._ddp_require_synchronized_gradients()
        self._validate_analog_groups()
        self.prepare_grad()
        for _, param in self._iter_analog_params():
            factor = float(getattr(param, "logical_native_gradient_scale", 1.0))
            if factor != 1.0 and param.grad is not None:
                param.grad.div_(factor)
        fresh = self._prepare_states_for_torch_step()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self._update_parameter(param, group)
        self._finish_states_after_torch_step(fresh)
        if self.freeze_analog_weight:
            self._ddp_finish_step()
            return loss
        for _, param in self._iter_analog_params():
            if self.state[param]:
                self._transfer_one(param, self.state[param])
        # The physical tile advances only on rank 0 inside _transfer_one().
        # Synchronize its full hidden state and reset the per-step gradient
        # guard before the next gradient-accumulation cycle.
        self._ddp_finish_step()
        return loss

    @staticmethod
    def _decay(param: Tensor, group: dict[str, Any]) -> None:
        decay = float(group.get("weight_decay", 0.0))
        if decay:
            param.mul_(1.0 - float(group["lr"]) * decay)


class LogicalMpLion(_LogicalOptimizer):
    """Lion (evolved sign momentum) in logical-weight units."""

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        momentum = state.setdefault("exp_avg", torch.zeros_like(param))
        beta1, beta2 = group.get("betas", (0.9, 0.99))
        self._decay(param, group)
        update = momentum.lerp(grad, 1.0 - beta1).sign()
        param.add_(update, alpha=-float(group["lr"]))
        momentum.lerp_(grad, 1.0 - beta2)


class LogicalMpRMSprop(_LogicalOptimizer):
    """RMSProp with a logical residual buffer; useful non-momentum control."""

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        square_avg = state.setdefault("square_avg", torch.zeros_like(param))
        alpha = float(group.get("alpha", 0.99))
        eps = float(group.get("eps", 1e-8))
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)
        self._decay(param, group)
        param.addcdiv_(grad, square_avg.sqrt().add_(eps), value=-float(group["lr"]))


class LogicalMpSGD(_LogicalOptimizer):
    """SGD with S-style logical residualized array programming.

    Analog contexts hold the pending logical increment rather than a physical
    weight.  Each call adds the ordinary SGD increment to that residual, and
    the inherited transfer routine emits only complete logical pulse quanta.
    Momentum, dampening, and Nesterov are evaluated digitally in logical-weight
    units before the same open-loop quantized transfer. Coupled SGD weight
    decay uses the logical target shadow maintained by the inherited
    indexed-safe residual mode. The shadow follows commanded optimizer updates
    from the initial logical weight and never reads the analog array.
    """

    def __init__(
        self,
        *args: Any,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        disable_context_buffer: bool = True,
        **kwargs: Any,
    ) -> None:
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if dampening < 0.0:
            raise ValueError(f"Invalid dampening value: {dampening}")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires positive momentum and zero dampening"
            )
        # ResNet's indexed Conv2d path derives a gradient directly from x/d;
        # its open-loop residual is separate optimizer state, not TensorBuffer.
        super().__init__(
            *args, disable_context_buffer=disable_context_buffer, **kwargs
        )
        # LogicalMpAdamW supplies the parameter-group machinery. Add the SGD
        # fields to its defaults and to groups created before analog regrouping.
        self.defaults.update({
            "momentum": float(momentum),
            "dampening": float(dampening),
            "nesterov": bool(nesterov),
        })
        for group in self.param_groups:
            group.setdefault("momentum", float(momentum))
            group.setdefault("dampening", float(dampening))
            group.setdefault("nesterov", bool(nesterov))

    @staticmethod
    def _momentum_direction(
        grad: Tensor, state: dict[str, Any], group: dict[str, Any]
    ) -> Tensor:
        """Return PyTorch-SGD's momentum direction without changing ``grad``."""
        momentum = float(group.get("momentum", 0.0))
        dampening = float(group.get("dampening", 0.0))
        nesterov = bool(group.get("nesterov", False))
        if momentum < 0.0 or dampening < 0.0:
            raise ValueError("momentum and dampening must be non-negative")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires positive momentum and zero dampening"
            )
        if momentum == 0.0:
            return grad
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = grad.detach().clone()
            state["momentum_buffer"] = buffer
        else:
            buffer.mul_(momentum).add_(grad, alpha=1.0 - dampening)
        if nesterov:
            return grad.add(buffer, alpha=momentum)
        return buffer

    @no_grad()
    def step(
        self, closure: Optional[Callable[[], float]] = None, **_: Any
    ) -> Optional[float]:
        """Apply SGD to logical residuals and transfer complete pulse quanta.

        In the indexed-Conv mode the residual is an optimizer-state tensor,
        because AnalogContext itself is deliberately a read-only array view.
        """
        self._validate_analog_groups()
        self.prepare_grad()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            decay = float(group.get("weight_decay", 0.0))
            for param in group["params"]:
                if param.grad is None:
                    continue
                if isinstance(param, AnalogContext):
                    factor = float(
                        getattr(param, "logical_native_gradient_scale", 1.0)
                    )
                    grad = param.grad if factor == 1.0 else param.grad.div(factor)
                    state = self.state[param]
                    if not state:
                        self._initialize_logical_state(param, state)
                    if decay:
                        # Match torch.optim.SGD's coupled L2 penalty. The
                        # shadow is an open-loop logical target, not readback.
                        grad = grad.add(
                            state["logical_weight_shadow"], alpha=decay
                        )
                    direction = self._momentum_direction(grad, state, group)
                    residual = state["logical_residual"]
                    residual.add_(direction, alpha=-lr)
                    if decay:
                        state["logical_weight_shadow"].add_(
                            direction, alpha=-lr
                        )
                    continue

                state = self.state[param]
                if decay:
                    grad = param.grad.add(param, alpha=decay)
                else:
                    grad = param.grad
                direction = self._momentum_direction(grad, state, group)
                param.add_(direction, alpha=-lr)

        if self.freeze_analog_weight:
            return loss
        for _, param in self._iter_analog_params():
            state = self.state[param]
            if state:
                self._transfer_one(param, state)
        return loss

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        direction = self._momentum_direction(
            param.grad, self.state[param], group
        )
        self._decay(param, group)
        param.add_(direction, alpha=-float(group["lr"]))


class LogicalMpAdafactor(_LogicalOptimizer):
    """Factored Adafactor, with unfactored state for vectors and scalars."""

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        beta2 = float(group.get("beta2", 0.999))
        eps = float(group.get("eps", 1e-30))
        if grad.ndim >= 2:
            matrix = grad.reshape(grad.shape[0], -1)
            row = state.setdefault("exp_avg_sq_row", torch.zeros(matrix.shape[0], device=grad.device, dtype=grad.dtype))
            col = state.setdefault("exp_avg_sq_col", torch.zeros(matrix.shape[1], device=grad.device, dtype=grad.dtype))
            row.mul_(beta2).add_(matrix.square().mean(dim=1), alpha=1.0 - beta2)
            col.mul_(beta2).add_(matrix.square().mean(dim=0), alpha=1.0 - beta2)
            denom = (row / row.mean().clamp_min(eps)).sqrt().unsqueeze(1) * col.sqrt().unsqueeze(0)
            update = (matrix / denom.clamp_min(eps)).reshape_as(grad)
        else:
            square_avg = state.setdefault("exp_avg_sq", torch.zeros_like(param))
            square_avg.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            update = grad / square_avg.sqrt().clamp_min(eps)
        rms = update.square().mean().sqrt().clamp_min(1.0)
        self._decay(param, group)
        param.add_(update / rms, alpha=-float(group["lr"]))


class LogicalMpShampoo(_LogicalOptimizer):
    """Two-sided Shampoo for matrices, with a diagonal fallback otherwise."""

    @staticmethod
    def _inverse_quarter_root(matrix: Tensor, epsilon: float) -> Tensor:
        values, vectors = torch.linalg.eigh(matrix)
        roots = values.clamp_min(epsilon).pow(-0.25)
        return (vectors * roots.unsqueeze(0)) @ vectors.transpose(0, 1)

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        beta = float(group.get("shampoo_beta", 0.99))
        eps = float(group.get("eps", 1e-8))
        if grad.ndim >= 2:
            matrix = grad.reshape(grad.shape[0], -1)
            left = state.setdefault("shampoo_left", torch.zeros(matrix.shape[0], matrix.shape[0], device=grad.device, dtype=grad.dtype))
            right = state.setdefault("shampoo_right", torch.zeros(matrix.shape[1], matrix.shape[1], device=grad.device, dtype=grad.dtype))
            left.mul_(beta).addmm_(matrix, matrix.transpose(0, 1), beta=1.0, alpha=1.0 - beta)
            right.mul_(beta).addmm_(matrix.transpose(0, 1), matrix, beta=1.0, alpha=1.0 - beta)
            left_root = self._inverse_quarter_root(left, eps)
            right_root = self._inverse_quarter_root(right, eps)
            update = (left_root @ matrix @ right_root).reshape_as(grad)
        else:
            square_avg = state.setdefault("square_avg", torch.zeros_like(param))
            square_avg.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)
            update = grad / square_avg.sqrt().clamp_min(eps)
        self._decay(param, group)
        param.add_(update, alpha=-float(group["lr"]))


class LogicalMpMuon(_LogicalOptimizer):
    """Muon on matrices and Adam-style auxiliary updates on other tensors."""

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        use_muon = bool(group.get("use_muon", grad.ndim >= 2)) and grad.ndim >= 2
        if use_muon:
            momentum = state.setdefault("momentum_buffer", torch.zeros_like(param))
            update = muon_update(
                grad,
                momentum,
                beta=float(group.get("momentum", 0.95)),
                nesterov=bool(group.get("nesterov", True)),
            )
        else:
            exp_avg = state.setdefault("exp_avg", torch.zeros_like(param))
            exp_avg_sq = state.setdefault("exp_avg_sq", torch.zeros_like(param))
            step = int(state.get("step", 0)) + 1
            state["step"] = step
            beta1, beta2 = group.get("betas", (0.9, 0.95))
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1.0 - beta2)
            update = (exp_avg / (1.0 - beta1 ** step)) / ((exp_avg_sq / (1.0 - beta2 ** step)).sqrt() + float(group.get("eps", 1e-8)))
        self._decay(param, group)
        param.add_(update.reshape_as(param), alpha=-float(group["lr"]))


class LogicalMpMoonlight(_LogicalOptimizer):
    """Moonlight's scalable-Muon update rule (Liu et al., 2025).

    The Moonlight reference implementation applies Nesterov Muon to matrices,
    uses AdamW for all remaining parameters, and scales every matrix update by
    ``0.2 * sqrt(max(A, B))`` to make its RMS comparable across shapes.  The
    decoupled decay is exact for ordinary torch tensors.  An ``AnalogContext``
    is a *pending update residual*, rather than the live array weight, so
    applying decoupled decay to it would be mathematically wrong.  We therefore
    reject non-zero decay on analog contexts instead of silently introducing a
    readback/reprogramming path that violates this project's open-loop model.
    """

    def _ddp_step_protocol_is_supported(self) -> bool:
        """Moonlight inherits _LogicalOptimizer.step's centralized protocol.

        This explicit declaration is deliberately per-concrete-class: PyTorch
        installs a profiler wrapper on ``step`` during optimizer construction,
        making function-identity checks unreliable.  If Moonlight ever grows
        its own ``step``, that implementation must be updated with the DDP
        synchronization protocol before this declaration may remain true.
        """

        return True

    @staticmethod
    def _moonlight_decay(param: Tensor, group: dict[str, Any]) -> None:
        decay = float(group.get("weight_decay", 0.0))
        if isinstance(param, AnalogContext) and decay:
            raise ValueError(
                "LogicalMpMoonlight cannot apply decoupled weight decay to an "
                "AnalogContext residual. Use weight_decay=0 for the open-loop "
                "AIMC path, or run the pure-digital Moonlight control."
            )
        if decay:
            param.mul_(1.0 - float(group["lr"]) * decay)

    def _update_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        grad = param.grad
        state = self.state[param]
        use_muon = bool(group.get("use_muon", grad.ndim >= 2)) and grad.ndim >= 2
        if use_muon:
            momentum = state.setdefault("momentum_buffer", torch.zeros_like(param))
            update = muon_update(
                grad, momentum, beta=float(group.get("momentum", 0.95)),
                nesterov=True, scale_by_aspect_ratio=False,
            )
            rows = int(param.shape[0])
            columns = int(math.prod(param.shape[1:]))
            update_scale = 0.2 * math.sqrt(max(rows, columns))
            self._moonlight_decay(param, group)
            param.add_(
                update.reshape_as(param),
                alpha=-float(group["lr"]) * update_scale,
            )
            return

        exp_avg = state.setdefault("exp_avg", torch.zeros_like(param))
        exp_avg_sq = state.setdefault("exp_avg_sq", torch.zeros_like(param))
        step = int(state.get("step", 0)) + 1
        state["step"] = step
        beta1, beta2 = group.get("betas", (0.9, 0.95))
        exp_avg.lerp_(grad, 1.0 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1.0 - beta2)
        update = (exp_avg / (1.0 - beta1 ** step)) / (
            (exp_avg_sq / (1.0 - beta2 ** step)).sqrt()
            + float(group.get("eps", 1e-8))
        )
        self._moonlight_decay(param, group)
        param.add_(update, alpha=-float(group["lr"]))


__all__ = [
    "LogicalMpSGD", "LogicalMpLion", "LogicalMpRMSprop", "LogicalMpAdafactor",
    "LogicalMpShampoo", "LogicalMpMuon", "LogicalMpMoonlight",
]
