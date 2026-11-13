import warnings
from typing import Any, Callable, Dict, Optional
from weakref import ref

import torch
from torch import cat
from torch.autograd import no_grad
from torch.nn.parallel import DistributedDataParallel

from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs.compounds import UnitCell
from aihwkit.simulator.configs.devices import FloatingPointDevice, PulsedDevice
from aihwkit.simulator.parameters.enums import PulseType


def _conv2d_weight_gradient(
    x_input: torch.Tensor,
    d_input: torch.Tensor,
    *,
    weight_size: tuple[int, int, int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    """Compute an indexed Conv2d weight gradient without Python im2col."""
    # `convolution_backward` only uses this tensor's shape when producing the
    # weight derivative. A real allocation avoids overlapping expanded storage.
    weight = d_input.new_empty(weight_size)
    # Match the full-precision accumulation used by the non-indexed path while
    # leaving the process-wide cuDNN setting unchanged.
    with torch.backends.cudnn.flags(allow_tf32=False):
        return torch.ops.aten.convolution_backward(
            d_input,
            x_input,
            weight,
            None,
            stride,
            padding,
            dilation,
            False,
            [0],
            groups,
            (False, True, False),
        )[1]


class AnalogOptimizerMixin:
    """Mixin for analog optimizers.

    This class contains the methods needed for enabling analog in an existing
    ``Optimizer``. It is designed to be used as a mixin in conjunction with an
    ``AnalogOptimizer`` or torch ``Optimizer``.
    """

    def regroup_param_groups(self, *_: Any) -> None:
        """Reorganize the parameter groups, isolating analog layers.

        Update the `param_groups` of the optimizer, moving the parameters for
        each analog layer to a new single group.
        """
        # Cache param id -> name mapping when a model/module is passed.
        # `regroup_param_groups(model)` is called from training scripts.
        if len(_) > 0 and hasattr(_[0], "named_parameters"):
            try:
                model = _[0]
                self._param_name_by_id = {
                    id(param): name for name, param in model.named_parameters()
                }
                self._register_indexed_conv_modules(model)
            except Exception:
                self._param_name_by_id = {}

        # Create the new param groups.
        analog_param_groups = []
        rm_group_lst = []
        for group in self.param_groups:  # type: ignore[has-type]
            rm_lst = []
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.set_learning_rate(
                        # self.defaults["lr"]  # type: ignore[attr-defined]
                        # TODO:
                        group["lr"]  # type: ignore[attr-defined]
                    )
                    analog_param_groups.append({"params": [param], **{
                        k: v for k, v in group.items() if k != "params"
                    }})
                    rm_lst.append(id(param))

            group["params"] = [p for p in group["params"] if id(p) not in rm_lst]

            if len(group["params"]) == 0:
                rm_group_lst.append(id(group))

        self.param_groups = [
            g for g in self.param_groups if id(g) not in rm_group_lst  # type: ignore[has-type]
        ]

        # Add analog groups.
        for group in analog_param_groups:
            self.add_param_group(group)  # type: ignore[attr-defined]

    def _register_indexed_conv_modules(self, model: Any) -> None:
        """Associate indexed tile contexts with their owning Conv2d modules."""
        for module in model.modules():
            analog_module = getattr(module, "analog_module", None)
            analog_ctx = getattr(analog_module, "analog_ctx", None)
            if not isinstance(analog_ctx, AnalogContext):
                continue
            if not hasattr(module, "fold_indices") or not hasattr(module, "kernel_size"):
                continue
            analog_ctx._mp_conv_module_ref = ref(module)

    @no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[float]:
        """Perform an analog-aware single optimization step.

        If a group containing analog parameters is detected, the optimization
        step calls the related RPU controller. For regular parameter groups,
        the optimization step has the same behaviour as ``torch.optim.SGD``.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            kwargs: additional arguments if any

        Returns:
            The loss, if ``closure`` has been passed as a parameter.
        """
        # pylint: disable=too-many-branches
        # Update non-analog parameters using the given optimizer
        ret = super().step(closure, **kwargs)  # type: ignore[misc]

        # Update analog parameters
        for group in self.param_groups:
            learning_rate = group.get("lr")

            # Use analog_tile object.
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # In this case a separate weight parameter exists: do nothing.
                        continue

                    # Call `update` in the tile.
                    if not analog_ctx.has_gradient():
                        # Forward never used.
                        continue

                    # Update learning rate.
                    if learning_rate == 0.0:
                        analog_ctx.reset()
                        continue

                    if learning_rate is not None:
                        analog_tile.set_learning_rate(learning_rate)

                    runtime = analog_tile.get_runtime()
                    if analog_ctx.use_indexed:
                        for x_input, d_input in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                (
                                    x_input.to(analog_tile.device)
                                    if runtime.offload_input
                                    else x_input
                                ),
                                (
                                    d_input.to(analog_tile.device)
                                    if runtime.offload_gradient
                                    else d_input
                                ),
                            )
                    else:
                        x_input = cat(
                            analog_ctx.analog_input, axis=-1 if analog_tile.in_trans else 0
                        )
                        d_input = cat(
                            analog_ctx.analog_grad_output, axis=-1 if analog_tile.out_trans else 0
                        )
                        analog_tile.update(
                            x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                            d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                        )

                    analog_ctx.reset()

        # Apply post-update step operations (diffuse, decay, etc).
        # (only here because of unknown params order and shared weights)
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.analog_tile.post_update_step()
        return ret

    def set_learning_rate(self, learning_rate: float = 0.1) -> None:
        """Update the learning rate to a new value.

        Update the learning rate of the optimizer, propagating the changes
        to the analog tiles accordingly.

        Args:
            learning_rate: learning rate for the optimizer.
        """
        for param_group in self.param_groups:
            param_group["lr"] = learning_rate
            for param in param_group["params"]:
                if isinstance(param, AnalogContext):
                    # Update learning rate on the tile
                    param.analog_tile.set_learning_rate(learning_rate)


class AnalogMpOptimizerMixin(AnalogOptimizerMixin):
    """Mixed-precision support with a correctness-first training-tile DDP protocol.

    AIHWKit training tiles explicitly do not support ordinary module-level DDP:
    their physical weights and device state are not PyTorch parameters or
    buffers.  This mixin excludes :class:`AnalogContext` objects from PyTorch's
    reducer, reduces their digital gradients explicitly once, performs noisy
    tile writes only on rank 0, and broadcasts checkpointable simulator state
    after every optimizer step.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_analog_weight = False
        self.torch_step = None
        self._ddp_prepared = False
        self._ddp_rank = 0
        self._ddp_world_size = 1
        self._ddp_gradients_synchronized = False
        self._ddp_analog_params = []
        self._ddp_analog_param_names = []
        self.check_update_config()
        
        # initalize the digital buffer
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue
                
                if getattr(self, "disable_context_buffer_default", False):
                    # Indexed Conv2d cannot use AIHWKit's direct digital
                    # accumulation: its backward path raises
                    # ``NotImplementedError``. Logical MP-SGD keeps only the
                    # current micro-batch x/d traces, reconstructs the exact
                    # Conv gradient in ``prepare_grad()``, then writes the
                    # independent open-loop residual. `zero_grad()` clears
                    # these traces before the next micro-batch, preventing
                    # their accumulation across an effective batch.
                    param._buffer_data = False
                    param._data_buffer = None
                    param.digital_grad_accumulation = False
                    continue
                if hasattr(param, "enable_buffer"):
                    param.enable_buffer()
                else:
                    mes = "AnalogContext does not have `enable_buffer` method. Update AIHWkit for better optimization."
                    warnings.warn(mes, UserWarning)
                    
                if hasattr(param, "set_digital_grad_accumulation"):
                    param.set_digital_grad_accumulation()
                else:
                    mes = "AnalogContext does not have `set_digital_grad_accumulation` method. Update AIHWkit for better optimization."
                    warnings.warn(mes, UserWarning)

    def zero_grad(self, *args: Any, **kwargs: Any):
        super().zero_grad(*args, **kwargs)
        self._ddp_gradients_synchronized = False
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    param.reset() # remove the x and d signals

    @staticmethod
    def _ddp_to_cpu(value: Any) -> Any:
        """Clone a nested optimizer/tile payload onto CPU."""

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: AnalogMpOptimizerMixin._ddp_to_cpu(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [AnalogMpOptimizerMixin._ddp_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(AnalogMpOptimizerMixin._ddp_to_cpu(item) for item in value)
        return value

    @staticmethod
    def _ddp_collective_device() -> torch.device:
        if torch.distributed.get_backend() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL DDP requires an available CUDA device")
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _ddp_broadcast_object(self, value: Any) -> Any:
        objects = [value if self._ddp_rank == 0 else None]
        torch.distributed.broadcast_object_list(
            objects, src=0, device=self._ddp_collective_device()
        )
        return objects[0]

    @staticmethod
    def _ddp_capture_tile_state(param: AnalogContext) -> Dict[str, Any]:
        """Capture every AIHWKit component used by checkpoint restore."""

        analog_tile = param.analog_tile
        simulator_tile = getattr(analog_tile, "tile", None)
        required = (
            "get_weights",
            "set_weights",
            "get_hidden_parameters",
            "set_hidden_parameters",
            "get_hidden_parameter_names",
            "get_learning_rate",
            "set_learning_rate",
            "dump_extra",
            "load_extra",
        )
        missing = [
            name for name in required
            if not callable(getattr(simulator_tile, name, None))
        ]
        if missing:
            raise RuntimeError(
                "Cannot guarantee DDP synchronization for AIHWKit tile "
                f"{type(analog_tile).__name__}: missing simulator state APIs "
                + ", ".join(missing)
            )

        extra = simulator_tile.dump_extra()
        if not isinstance(extra, dict):
            raise RuntimeError(
                "Cannot guarantee DDP synchronization: tile.dump_extra() did not "
                f"return a dictionary for {type(analog_tile).__name__}."
            )
        try:
            context_buffer = param.data.detach().cpu().clone()
        except RuntimeError as exc:
            raise RuntimeError(
                "DDP mixed-precision synchronization requires AnalogContext BUFFER "
                "mode so the logical residual can be broadcast."
            ) from exc
        return {
            "weights": simulator_tile.get_weights().detach().cpu().clone(),
            "hidden_parameters": (
                simulator_tile.get_hidden_parameters().detach().cpu().clone()
            ),
            "hidden_parameter_names": list(
                simulator_tile.get_hidden_parameter_names()
            ),
            "extra": AnalogMpOptimizerMixin._ddp_to_cpu(extra),
            "learning_rate": float(simulator_tile.get_learning_rate()),
            "context_buffer": context_buffer,
        }

    @staticmethod
    def _ddp_restore_tile_state(
        param: AnalogContext, state: Dict[str, Any]
    ) -> None:
        analog_tile = param.analog_tile
        simulator_tile = analog_tile.tile
        live_names = list(simulator_tile.get_hidden_parameter_names())
        saved_names = list(state["hidden_parameter_names"])
        if live_names != saved_names:
            raise RuntimeError(
                "Cannot restore rank-0 AIHWKit tile state: hidden parameter names "
                f"differ (live={live_names}, rank0={saved_names})."
            )

        # Match AIHWKit's own __setstate__ order. Hidden parameters include
        # device-to-device variation and other state invisible in the weight
        # tensor; dump_extra() carries counters and compound-device state.
        simulator_tile.set_hidden_parameters(state["hidden_parameters"])
        simulator_tile.set_weights(state["weights"])
        simulator_tile.set_learning_rate(float(state["learning_rate"]))
        simulator_tile.load_extra(state["extra"], True)
        param.data.copy_(
            state["context_buffer"].to(device=param.device, dtype=param.dtype)
        )

    def _ddp_broadcast_optimizer_state(self) -> None:
        state = None
        if self._ddp_rank == 0:
            state = self._ddp_to_cpu(self.state_dict())
        state = self._ddp_broadcast_object(state)
        if self._ddp_rank != 0:
            self.load_state_dict(state)

    def _ddp_broadcast_tile_states(self) -> None:
        for param in self._ddp_analog_params:
            state = None
            if self._ddp_rank == 0:
                state = self._ddp_capture_tile_state(param)
            state = self._ddp_broadcast_object(state)
            if self._ddp_rank != 0:
                self._ddp_restore_tile_state(param, state)

    def _ddp_step_protocol_is_supported(self) -> bool:
        """Return whether this class uses the centralized base-mixin step."""

        return type(self).step is AnalogMpOptimizerMixin.step

    def prepare_for_ddp(self, model: Any) -> None:
        """Broadcast rank-0 state before constructing PyTorch DDP.

        The optimizer must already be regrouped and checkpoint state must
        already be loaded. AnalogContexts are then excluded from the standard
        reducer so their gradients cannot be reduced twice.
        """

        if self._ddp_prepared:
            raise RuntimeError("prepare_for_ddp() may only be called once")
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "prepare_for_ddp() requires an initialized torch.distributed "
                "process group"
            )
        if not self._ddp_step_protocol_is_supported():
            raise RuntimeError(
                f"{type(self).__name__} overrides optimizer.step() without "
                "integrating the centralized AIHWKit DDP tile-update protocol."
            )

        analog_params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if isinstance(param, AnalogContext)
        ]
        if not analog_params:
            raise RuntimeError(
                "AIHWKit DDP preparation found no AnalogContext parameters"
            )
        if len({id(param) for param in analog_params}) != len(analog_params):
            raise RuntimeError(
                "An AnalogContext occurs in more than one optimizer group"
            )

        name_by_id = {id(param): name for name, param in model.named_parameters()}
        if any(id(param) not in name_by_id for param in analog_params):
            raise RuntimeError(
                "Every optimizer AnalogContext must be registered in the model "
                "before DDP"
            )
        for param in analog_params:
            if param.use_torch_update:
                raise RuntimeError(
                    "The centralized training-tile DDP protocol does not support "
                    "AnalogContext.use_torch_update."
                )
            if not getattr(param, "digital_grad_accumulation", False):
                raise RuntimeError(
                    "The centralized training-tile DDP protocol requires "
                    "backward-time digital AnalogContext gradient accumulation. "
                    "Indexed/deferred X/D reconstruction is not supported."
                )

        self._ddp_rank = torch.distributed.get_rank()
        self._ddp_world_size = torch.distributed.get_world_size()
        if self._ddp_world_size < 2:
            raise RuntimeError("AIHWKit DDP protocol requires world_size >= 2")
        self._ddp_analog_params = analog_params
        self._ddp_analog_param_names = [
            name_by_id[id(param)] for param in analog_params
        ]

        self._ddp_broadcast_optimizer_state()
        self._ddp_broadcast_tile_states()

        ignored = set(
            getattr(model, "_ddp_params_and_buffers_to_ignore", set())
        )
        ignored.update(self._ddp_analog_param_names)
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model, ignored
        )
        self._ddp_prepared = True
        self._ddp_gradients_synchronized = False

    def ddp_synchronize_after_load(self) -> None:
        """Broadcast rank-0 optimizer and tile state after an in-place resume."""

        if not self._ddp_prepared:
            raise RuntimeError(
                "ddp_synchronize_after_load() requires prior prepare_for_ddp()"
            )
        self._ddp_broadcast_optimizer_state()
        self._ddp_broadcast_tile_states()
        self._ddp_gradients_synchronized = False

    @no_grad()
    def ddp_synchronize_gradients(self, ddp_model: Any) -> bool:
        """Average AnalogContext gradients exactly once for this step."""

        if not self._ddp_prepared:
            raise RuntimeError(
                "ddp_synchronize_gradients() called before prepare_for_ddp()"
            )
        sync_required = getattr(ddp_model, "require_backward_grad_sync", None)
        if sync_required is None:
            raise TypeError(
                "ddp_synchronize_gradients() expects a "
                "DistributedDataParallel model"
            )
        if not bool(sync_required):
            return False
        if self._ddp_gradients_synchronized:
            raise RuntimeError(
                "AnalogContext gradients were already synchronized for this "
                "optimizer step"
            )

        collective_device = self._ddp_collective_device()
        for param in self._ddp_analog_params:
            present = torch.tensor(
                [1 if param.grad is not None else 0],
                device=collective_device,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(
                present, op=torch.distributed.ReduceOp.SUM
            )
            if int(present.item()) == 0:
                continue
            grad = param.grad
            if grad is None:
                grad = torch.zeros(
                    tuple(param.shape), device=param.device, dtype=param.dtype
                )
            elif tuple(grad.shape) != tuple(param.shape):
                raise RuntimeError(
                    "AnalogContext gradient shape differs from its logical shape: "
                    f"grad={tuple(grad.shape)}, context={tuple(param.shape)}"
                )
            torch.distributed.all_reduce(
                grad, op=torch.distributed.ReduceOp.SUM
            )
            grad.div_(self._ddp_world_size)
            param.grad = grad

        self._ddp_gradients_synchronized = True
        return True

    def _ddp_require_synchronized_gradients(self) -> None:
        if self._ddp_prepared and not self._ddp_gradients_synchronized:
            raise RuntimeError(
                "optimizer.step() would update an AIHWKit training tile before "
                "AnalogContext gradients were synchronized. Call "
                "optimizer.ddp_synchronize_gradients(ddp_model) after the final "
                "accumulation micro-step."
            )

    def _ddp_is_update_rank(self) -> bool:
        return not self._ddp_prepared or self._ddp_rank == 0

    def _ddp_finish_step(self) -> None:
        if not self._ddp_prepared:
            return
        self._ddp_broadcast_tile_states()
        self._ddp_gradients_synchronized = False

    def check_update_config(self):
        for group in self.param_groups:
            # Use analog_tile object.
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile
        
                    rpu_config = analog_tile.rpu_config

                    if isinstance(rpu_config.device, FloatingPointDevice):
                        continue

                    '''
                    Mixed precision update requires the tile are updated by
                    deterministic pulse update scheme, since the pulse number is
                    managed in the Python side.
                    '''
                    config_updated = False
                    
                    # if rpu_config.update.fixed_bl != True:
                    #     warnings.warn(f"rpu_config.update.fixed_bl was not set to True (current: {getattr(rpu_config.update, 'fixed_bl', 'not set')}). Setting to True.", UserWarning)
                    #     rpu_config.update.fixed_bl = True
                    #     config_updated = True

                    if rpu_config.update.pulse_type != PulseType.DETERMINISTIC_IMPLICIT:
                        warnings.warn(f"rpu_config.update.pulse_type was not set to PulseType.DETERMINISTIC_IMPLICIT (current: {getattr(rpu_config.update, 'pulse_type', 'not set')}). Setting to PulseType.DETERMINISTIC_IMPLICIT.", UserWarning)
                        rpu_config.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
                        config_updated = True
                    
                    # if rpu_config.update.update_bl_management != True:
                    #     warnings.warn(f"rpu_config.update.update_bl_management was not set to True (current: {getattr(rpu_config.update, 'update_bl_management', 'not set')}). Setting to True.", UserWarning)
                    #     rpu_config.update.update_bl_management = True
                    #     config_updated = True
                    
                    if rpu_config.update.update_management != True:
                        warnings.warn(f"rpu_config.update.update_management was not set to True (current: {getattr(rpu_config.update, 'update_management', 'not set')}). Setting to True.", UserWarning)
                        rpu_config.update_management = True
                        config_updated = True
                    
                    # Apply the configuration to analog_tile if any updates were made
                    if config_updated:
                        warnings.warn("RPU configuration was updated. Applying changes with analog_tile.to(real_rpu_config).", UserWarning)
                        analog_tile.to(rpu_config)
                        
    def get_processed_x_d_input(self, analog_ctx):
        analog_tile = analog_ctx.analog_tile
        runtime = analog_tile.get_runtime()
        if analog_ctx.use_indexed:
            module_ref = getattr(analog_ctx, "_mp_conv_module_ref", None)
            conv_module = module_ref() if module_ref is not None else None
            if conv_module is None:
                raise RuntimeError(
                    "Indexed AnalogConv requires optimizer.regroup_param_groups(model) "
                    "after model conversion so its convolution metadata can be registered."
                )

            processed_batches = []
            for x_input, d_input in zip(
                analog_ctx.analog_input, analog_ctx.analog_grad_output
            ):
                if runtime.offload_input:
                    x_input = x_input.to(analog_tile.device)
                if runtime.offload_gradient:
                    d_input = d_input.to(analog_tile.device)

                # Mirror AnalogTile.update_indexed(): mapping and input-range
                # processing happen on the original NCHW tensors.
                x_input, d_input = analog_tile.pre_update(x_input, 1, d_input, 1)
                if x_input.dim() != 4 or d_input.dim() != 4:
                    raise NotImplementedError(
                        "Digital indexed MP gradients currently support AnalogConv2d "
                        f"only; got X={tuple(x_input.shape)}, D={tuple(d_input.shape)}."
                    )

                batch_grad = _conv2d_weight_gradient(
                    x_input,
                    d_input,
                    weight_size=(
                        conv_module.out_channels,
                        conv_module.in_channels // conv_module.groups,
                        *conv_module.kernel_size,
                    ),
                    stride=conv_module.stride,
                    padding=conv_module.padding,
                    dilation=conv_module.dilation,
                    groups=conv_module.groups,
                )
                processed_batches.append(batch_grad.flatten(1))
            return processed_batches

        # A shared Conv can be called at several pyramid resolutions in one
        # autograd graph (FPN/RPN).  Its unfolded traces then differ in their
        # sequence dimension and cannot be concatenated directly.  Process
        # each call independently and return additive weight gradients; this
        # is exactly the same sum as concatenation when shapes do agree.
        gradients = []
        for x_input, d_input in zip(
            analog_ctx.analog_input, analog_ctx.analog_grad_output
        ):
            if runtime.offload_input:
                x_input = x_input.to(analog_tile.device)
            if runtime.offload_gradient:
                d_input = d_input.to(analog_tile.device)
            x_input, d_input = analog_tile.pre_update(
                x_input,
                0 if analog_tile.in_trans else x_input.dim() - 1,
                d_input,
                0 if analog_tile.out_trans else d_input.dim() - 1,
            )
            gradients.append(self.compute_gradient(x_input, d_input))
        return gradients

    def _get_RPU_device(self, rpu_config):
        if isinstance(rpu_config.device, PulsedDevice):
            return rpu_config.device
        elif isinstance(rpu_config.device, UnitCell):
            return rpu_config.device.unit_cell_devices[0]
        elif isinstance(rpu_config.device, FloatingPointDevice):
            return rpu_config.device
        else:
            return None  # floating point

    @no_grad()
    def compute_gradient(self, x_input, d_input):
        """Compute gradient via digital MAC operations.

        Args:
            x_input: input activations
            d_input: gradient w.r.t. output activations

        Returns:
            The computed gradient.
        """
        # Indexed Conv2d returns a weight gradient per saved forward batch.
        if isinstance(x_input, list):
            if not x_input:
                raise RuntimeError("Indexed AnalogConv has no saved X/D batches.")
            grad = None
            for batch_grad in x_input:
                grad = batch_grad if grad is None else grad.add_(batch_grad)
            return grad

        # Compute gradient as average of outer products over batch
        # W: (N, M)
        # x_input: (B, M), d_input: (B, N)
        # gradient: (M, N) = sum(d_i ⊗ x_i) for i in batch
        # The avearage over the batchsize is implemented in the training code
        #   here we only sum up the gradient
        # Feature maps from maintained vision models can be non-contiguous
        # (for example after FPN interpolation).  ``reshape`` preserves the
        # mathematical flattening while materialising a contiguous copy only
        # when necessary; ``view`` would incorrectly reject those valid
        # traces.
        grad = torch.matmul(
            d_input.reshape(-1, d_input.size(-1)).transpose(0, 1),
            x_input.reshape(-1, x_input.size(-1))
        )
        
        return grad

    @no_grad()
    def prepare_grad(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if not isinstance(param, AnalogContext):
                    continue

                analog_ctx = param

                if getattr(analog_ctx, "digital_grad_accumulation", False) and param.grad is not None:
                    # LogicalAnalogLinear explicitly writes this gradient in
                    # logical units.  Some converted Conv modules retain x/d
                    # traces instead; those use the fallback below.
                    continue

                # Legacy path: build the gradient from retained x and d traces.
                if not analog_ctx.has_gradient():
                    # A converted composite module can retain a wrapper context
                    # without owning a trainable tile. It has no x/d trace and
                    # must simply be skipped, like an unused torch parameter.
                    continue

                processed = self.get_processed_x_d_input(analog_ctx)
                if isinstance(processed, list):
                    grad = self.compute_gradient(processed, None)
                else:
                    x_input, d_input = processed
                    grad = self.compute_gradient(x_input, d_input)
                
                # TODO: use param as transfer buffer
                
                if param.size() != grad.size():
                    param.data = torch.zeros_like(grad)
                param.grad = grad
    
    @no_grad()
    def step(self, closure: Optional[Callable] = None, **kwargs: Any) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # pylint: disable=too-many-branches
        self._ddp_require_synchronized_gradients()
        self.prepare_grad()

        # ====================================
        # Update non-analog parameters using the given optimizer
        # call torch optimizer and achieve the desired update on AdamW, shored in param
        # ====================================
        assert self.torch_step is not None, "need to define torch_step"
        ret = self.torch_step(self, closure, **kwargs)  # type: ignore[misc]

        if self.freeze_analog_weight:
            self._ddp_finish_step()
            return ret

        # ====================================
        # analog update: transfer
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, AnalogContext):
                    # Handle internal analog update.
                    analog_ctx = param
                    analog_tile = analog_ctx.analog_tile

                    if analog_ctx.use_torch_update:
                        # In this case a separate weight parameter exists: do nothing.
                        continue

                    # A model can contain an analog layer unused by the current
                    # forward pass. It has no new residual to transfer.  In
                    # AIHWKit's digital-gradient-accumulation mode, however,
                    # backward writes the already preconditioned gradient
                    # directly to ``AnalogContext.grad`` and intentionally
                    # leaves no x/d trace.  Treat that direct gradient as a
                    # pending update as well; otherwise native MpAdamW updates
                    # only its digital Adam buffer and never writes the array.
                    has_pending_gradient = analog_ctx.has_gradient() or param.grad is not None
                    if not has_pending_gradient:
                        continue

                    analog_tile.set_learning_rate(1.)

                    rpu_device = self._get_RPU_device(analog_tile.rpu_config)
                    if not isinstance(rpu_device, FloatingPointDevice):
                        # Don't need to scale dw_min by mapping coefficient
                        # The mapping coefficient is already considered in the gradient computation
                        dw_min = rpu_device.dw_min
                        max_bit_length:int = analog_tile.rpu_config.update.desired_bl
                    else:
                        dw_min = max_bit_length = None
                        
                    if isinstance(rpu_device, FloatingPointDevice):
                        desired_update = param
                    else:
                        desired_pulse = param.div(dw_min).trunc()
                        available_pulse = desired_pulse.clamp(
                            min=-max_bit_length, max=max_bit_length
                        )
                        desired_update = available_pulse.mul(dw_min)

                    # The transfer is a dense update of the digital residual,
                    # independent of whether the forward pass used indexed
                    # Conv2d traces.
                    # All ranks update the deterministic digital residual and
                    # optimizer state. Only rank 0 is allowed to advance the
                    # stochastic physical simulator.
                    if self._ddp_is_update_rank():
                        self.update_tile(analog_tile, desired_update)
                    if hasattr(analog_ctx, "writable"):
                        with analog_ctx.writable():
                            param.data.sub_(desired_update)
                    else:
                        param.data.sub_(desired_update)

        # Apply post-update step operations (diffuse, decay, etc).
        # (only here because of unknown params order and shared weights)
        if self._ddp_is_update_rank():
            for group in self.param_groups:
                for param in group["params"]:
                    if isinstance(param, AnalogContext):
                        param.analog_tile.post_update_step()

        self._ddp_finish_step()
        return ret

    @staticmethod
    def update_tile(tile, desired_update):
        '''Define a function here since the subclass may need to update other tiles.

        ``desired_update`` is already in the physical tile coordinate: its
        gradient was produced by ``pre_update``, which applies the native
        mapping compensation once.  ``tile.update`` invokes ``pre_update``
        again, so pass a correspondingly scaled command to ensure that this
        second invocation is algebraically neutral rather than applying an
        erroneous extra inverse mapping factor to every array write.
        '''
        mapping_scales = tile.get_mapping_scales()
        if mapping_scales is not None:
            if mapping_scales.numel() == 1:
                desired_update = desired_update * mapping_scales
            elif mapping_scales.numel() == desired_update.size(0):
                desired_update = desired_update * mapping_scales.view(-1, 1)
            else:
                raise ValueError(
                    "Cannot align native mapping scales with the requested "
                    f"physical update: scales={tuple(mapping_scales.shape)}, "
                    f"update={tuple(desired_update.shape)}."
                )
        if not hasattr(tile, '_eye_matrix'):
            tile._eye_matrix = torch.eye(tile.in_size, device=tile.device)
        tile.update(
            tile._eye_matrix,
            -desired_update.transpose(0, 1)
        )
