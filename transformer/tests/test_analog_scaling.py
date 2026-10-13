import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from aihwkit.nn import AnalogLinear
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import FloatingPointRPUConfig, SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.parameters import IOParameters, UpdateParameters
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, PulseType
from MixedPrecisionOptim.AdamW import MpAdamW

from analog_model import layer_rpu_config
from analog_tied_embedding import TiedAnalogEmbedding
from scaled_analog_linear import LogicalAnalogLinear
from scaling_config import ScalingConfig


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device required'))]


def mapped_layer(device, normalize=False, scale=0.2, pulsed=False):
    if pulsed:
        rpu = SingleRPUConfig(device=ConstantStepDevice(
            w_min=-10, w_max=10, w_min_dtod=0, w_max_dtod=0,
            dw_min=0.002, dw_min_dtod=0, dw_min_std=0, up_down=0, up_down_dtod=0),
            forward=IOParameters(is_perfect=True), backward=IOParameters(is_perfect=True),
            update=UpdateParameters(pulse_type=PulseType.DETERMINISTIC_IMPLICIT))
    else:
        rpu = FloatingPointRPUConfig()
    rpu.mapping.learn_out_scaling = False
    base = AnalogLinear(4, 3, bias=False, rpu_config=rpu).to(device)
    base.set_weights(torch.arange(12, device=device).reshape(3, 4).float() * 0.01,
                     apply_weight_scaling=False)
    return LogicalAnalogLinear(base, scale, normalize_backward=normalize)


def test_sout_preserves_native_management_and_per_matrix_rails():
    config = ScalingConfig(mode='analog')
    ordinary = layer_rpu_config(config, 'transformer.h.0.attn.c_attn', nn.Linear(128, 384))
    head = layer_rpu_config(config, 'lm_head', nn.Linear(128, 50_257))
    for rpu in (ordinary, head):
        assert rpu.forward.inp_res == pytest.approx(1 / (2**16 - 2))
        assert rpu.forward.out_res == pytest.approx(1 / (2**9 - 2))
        assert rpu.backward.inp_res == rpu.forward.inp_res
        assert rpu.backward.out_res == rpu.forward.out_res
        assert rpu.forward.noise_management == NoiseManagementType.ABS_MAX
        assert rpu.backward.noise_management == NoiseManagementType.ABS_MAX
        assert rpu.forward.bound_management == BoundManagementType.ITERATIVE
        assert rpu.backward.bound_management == BoundManagementType.NONE
        assert not rpu.mapping.learn_out_scaling
        assert rpu.update.pulse_type == PulseType.DETERMINISTIC_IMPLICIT
    assert ordinary.forward.out_bound == pytest.approx(6 / 3 * 128**0.5)
    assert ordinary.backward.out_bound == pytest.approx(3.5 / 3 * 384**0.5)
    assert head.backward.out_bound == pytest.approx(3.5 / 3)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('normalize', [False, True])
def test_reads_and_logical_gradients_match_digital_oracle(device, normalize):
    layer = mapped_layer(device, normalize)
    context = layer.analog_linear.analog_module.analog_ctx
    context.enable_buffer()
    context.set_digital_grad_accumulation()
    weight = layer.analog_linear.get_weights(apply_weight_scaling=False)[0].to(device) * 0.2
    target = nn.Parameter(weight)
    x = torch.randn(2, 5, 4, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    errors = torch.randn(2, 5, 3, device=device)
    errors[0, 0].zero_()
    actual, reference = layer(x), F.linear(reference_x, target)
    torch.testing.assert_close(actual, reference)
    actual.backward(errors)
    reference.backward(errors)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(context.grad, target.grad)


@pytest.mark.parametrize('device', DEVICES)
def test_tied_embedding_and_head_accumulate_into_one_weight(device):
    head = mapped_layer(device, normalize=True)
    embedding = TiedAnalogEmbedding(head, read_chunk_size=2)
    model = nn.ModuleDict({'head': head, 'embedding': embedding})
    optimizer = MpAdamW(model.parameters(), lr=0.001, weight_decay=0)
    contexts = [param for param in model.parameters() if isinstance(param, AnalogContext)]
    assert len(contexts) == 1
    target = nn.Parameter(head.analog_linear.get_weights(apply_weight_scaling=False)[0].to(device) * 0.2)
    tokens = torch.tensor([[0, 1, 0, 2]], device=device)
    actual = head(embedding(tokens))
    reference = F.linear(F.embedding(tokens, target), target)
    torch.testing.assert_close(actual, reference)
    actual.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(contexts[0].grad, target.grad)
    optimizer.step()


@pytest.mark.parametrize('device', DEVICES)
def test_quantized_adam_write_settles_only_sent_logical_increment(device):
    layer = mapped_layer(device, pulsed=True)
    tile = layer.analog_linear.analog_module
    optimizer = MpAdamW(layer.parameters(), lr=0.003, betas=(0.0, 0.0), eps=0.1, weight_decay=0)
    before = layer.analog_linear.get_weights(apply_weight_scaling=False)[0].to(device)
    layer(torch.randn(3, 4, device=device)).square().mean().backward()
    target = nn.Parameter(before * 0.2)
    reference = torch.optim.AdamW([target], lr=0.003, betas=(0.0, 0.0), eps=0.1, weight_decay=0)
    target.grad = tile.analog_ctx.grad.clone()
    reference.step()
    optimizer.step()
    after = layer.analog_linear.get_weights(apply_weight_scaling=False)[0].to(device)
    residual = optimizer.logical_state(tile.analog_ctx)['residual']
    torch.testing.assert_close(residual + 0.2 * (after - before), target.detach() - 0.2 * before,
                               atol=2e-6, rtol=1e-4)
    assert residual.abs().max() <= 0.2 * 0.002 + 1e-7


@pytest.mark.parametrize('device', DEVICES)
def test_model_and_optimizer_checkpoint_restore_mapping_and_residual(device):
    model = mapped_layer(device)
    optimizer = MpAdamW(model.parameters(), lr=0.001, weight_decay=0)
    model(torch.ones(2, 4, device=device)).square().mean().backward()
    optimizer.step()
    saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    replacement = mapped_layer(device, scale=0.9)
    replacement.load_state_dict(saved_model)
    restored = MpAdamW(replacement.parameters(), lr=0.001, weight_decay=0)
    restored.load_state_dict(saved_optimizer)
    tile = replacement.analog_linear.analog_module
    torch.testing.assert_close(tile.get_mapping_scales(), torch.tensor([0.2], device=device))
    torch.testing.assert_close(replacement(torch.ones(2, 4, device=device)), model(torch.ones(2, 4, device=device)))
    torch.testing.assert_close(restored.logical_state(tile.analog_ctx)['residual'],
                               optimizer.logical_state(model.analog_linear.analog_module.analog_ctx)['residual'])
