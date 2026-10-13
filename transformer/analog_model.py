"""Li-ECRAM model construction for the S-OUT Adam scaling comparison."""
import math

import torch
from torch import nn
from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, PulseType
from aihwkit.simulator.presets.devices import EcRamPresetDevice

from analog_tied_embedding import TiedAnalogEmbedding
from scaled_analog_linear import LogicalAnalogLinear, derive_rpu_construction_seed, fixed_output_scale


def layer_rpu_config(config, name, layer):
    rpu = SingleRPUConfig(device=EcRamPresetDevice(
        construction_seed=derive_rpu_construction_seed(config.seed, name)))
    rpu.mapping.max_input_size = rpu.mapping.max_output_size = 2**30
    rpu.mapping.weight_scaling_omega = 0
    rpu.mapping.learn_out_scaling = False
    rpu.mapping.out_scaling_columnwise = False
    physical_std = config.conductance_bound / config.omega
    for io in (rpu.forward, rpu.backward):
        io.inp_res = 1 / (2**config.dac_bits - 2)
        io.out_res = 1 / (2**config.adc_bits - 2)
        io.noise_management = NoiseManagementType.ABS_MAX
    rpu.forward.bound_management = BoundManagementType.ITERATIVE
    rpu.backward.bound_management = BoundManagementType.NONE
    rpu.forward.out_bound = config.c_out * physical_std * math.sqrt(layer.in_features)
    rpu.backward.out_bound = config.backward_rail * physical_std * (
        1 if name == 'lm_head' else math.sqrt(layer.out_features))
    rpu.update.fixed_bl = True
    rpu.update.update_management = True
    rpu.update.update_bl_management = True
    rpu.update.pulse_type = PulseType.DETERMINISTIC_IMPLICIT
    return rpu


def convert_scaling_model(model, config):
    def per_layer(name, layer, rpu):
        return layer_rpu_config(config, name, layer)

    model = convert_to_analog(model, rpu_config=SingleRPUConfig(device=EcRamPresetDevice()),
                              specific_rpu_config_fun=per_layer, inplace=True)

    def wrap(parent, prefix=''):
        for name, child in list(parent.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(child, AnalogLinear):
                scale = fixed_output_scale(config.shape['n_layer'], child.in_features,
                                           full_name.endswith('c_proj'), config.omega,
                                           config.conductance_bound)
                setattr(parent, name, LogicalAnalogLinear(
                    child, scale, normalize_backward=full_name == 'lm_head'))
            else:
                wrap(child, full_name)

    wrap(model)
    model.transformer.wte = TiedAnalogEmbedding(model.lm_head, config.embedding_read_chunk_size)
    return model
