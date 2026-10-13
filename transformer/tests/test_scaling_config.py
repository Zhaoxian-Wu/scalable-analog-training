from pathlib import Path

import pytest
import torch

from model import GPT, GPTConfig
from scaling_config import MODEL_SCALES, ScalingConfig, learning_rate_at, load_config, should_evaluate


@pytest.mark.parametrize('scale,updates,last_eval', [
    ('tiny', 278, 273), ('small', 652, 640), ('medium', 1218, 1200),
    ('large', 2072, 2060), ('xlarge', 5028, 5020),
])
def test_budget_and_validation_match_manuscript_table(scale, updates, last_eval):
    config = ScalingConfig(model_scale=scale)
    assert config.tokens_per_update == 491_520
    assert config.planned_updates == updates
    assert config.final_validation_step == last_eval
    with torch.device('meta'):
        shape = config.shape
        model = GPT(GPTConfig(n_layer=shape['n_layer'], n_head=shape['n_head'],
                              n_embd=shape['n_embd'], vocab_size=50_257, bias=False))
    assert model.get_num_params() == shape['parameters']


def test_adam_and_converter_settings_match_selected_appendix():
    root = Path(__file__).resolve().parents[1]
    digital = load_config(root / 'config/digital_adam.py')
    analog = load_config(root / 'config/analog_adam.py')
    assert (digital.mode, analog.mode) == ('digital', 'analog')
    for config in (digital, analog):
        assert (config.beta1, config.beta2, config.weight_decay) == (0.9, 0.95, 0.0)
        assert (config.learning_rate, config.min_lr, config.grad_clip) == (6e-4, 6e-5, 1.0)
    assert (analog.dac_bits, analog.adc_bits, analog.c_out) == (16, 9, 6.0)


def test_short_prefix_preserves_full_learning_rate_schedule():
    full = ScalingConfig(model_scale='xlarge')
    prefix = ScalingConfig(model_scale='xlarge', max_iters=3)
    assert prefix.warmup_iters == full.warmup_iters == 251
    assert learning_rate_at(0, full) == 0
    assert learning_rate_at(full.warmup_iters, full) == pytest.approx(6e-4)
    assert learning_rate_at(full.planned_updates, full) == pytest.approx(6e-5)
    for step in (1, 2, 3):
        assert learning_rate_at(step, prefix) == learning_rate_at(step, full)
    assert should_evaluate(5020, full)
    assert not should_evaluate(5028, full)


def test_execution_topology_cannot_change_global_batch():
    ScalingConfig(batch_size=30).validate(world_size=2)
    with pytest.raises(ValueError, match='divide global_batch_size'):
        ScalingConfig(batch_size=31).validate(world_size=2)
    with pytest.raises(ValueError, match='only digital AdamW'):
        ScalingConfig(mode='unsupported').validate()
