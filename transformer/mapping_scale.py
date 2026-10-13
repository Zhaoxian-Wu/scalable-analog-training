"""Mapping-scale views and validation of restored native tile scales."""

from __future__ import annotations

import torch


class MappingScaleMixin:
    """Expose wrapper scale attributes as views of checkpointed tile scales."""

    def _mapping_tiles(self):
        child = getattr(self, "analog_linear", None)
        if child is None:
            child = self.analog_layer
        tile = child.analog_module
        if hasattr(tile, "analog_ctx"):
            return [tile]
        return [module for module in tile.modules() if hasattr(module, "analog_ctx")]

    @property
    def output_scale(self) -> torch.Tensor:
        tile = self._mapping_tiles()[0]
        if getattr(self, "learned_output_scale", False):
            return tile.out_scaling_alpha
        scale = tile.get_mapping_scales()
        multiplier = getattr(self, "mapping_warmup_multiplier", 1.0)
        return scale / multiplier

    @property
    def logical_scale(self) -> torch.Tensor:
        return self._mapping_tiles()[0].get_mapping_scales()

    def _init_mapping_checkpoint(self):
        self.register_load_state_dict_post_hook(self._validate_restored_mapping)

    def _validate_restored_mapping(self, module, incompatible_keys):
        for tile in self._mapping_tiles():
            scales = tile.get_mapping_scales()
            if (
                scales is None
                or scales.numel() not in (1, tile.out_size)
                or not torch.isfinite(scales).all()
                or (scales <= 0).any()
            ):
                raise ValueError("Restored mapping_scales must be finite, positive and match tile rows")
            tile.analog_ctx.mapped_gradient_units = True
            tile.analog_ctx.logical_mapping_multiplier = float(
                getattr(self, "mapping_warmup_multiplier", 1.0)
            )
