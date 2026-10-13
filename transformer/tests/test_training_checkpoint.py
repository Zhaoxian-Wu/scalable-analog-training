from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch

from scaling_config import ScalingConfig
from train import train


@pytest.mark.parametrize('mode', ['digital', 'analog'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device required'))])
def test_short_synthetic_prefix_and_resume_keep_scaling_settings(tmp_path, mode, device):
    dataset = tmp_path / 'tokens'
    dataset.mkdir()
    generator = np.random.default_rng(1337)
    for split in ('train', 'val'):
        generator.integers(0, 65, size=2048, dtype=np.uint16).tofile(dataset / f'{split}.bin')
    with (dataset / 'meta.pkl').open('wb') as handle:
        pickle.dump({'vocab_size': 65}, handle)
    config = ScalingConfig(mode=mode, vocab_size=65, block_size=16, global_batch_size=2,
                           batch_size=1, device=device, dataset_dir=str(dataset), max_iters=2,
                           eval_iters=1, checkpoint_interval=1, output_dir=str(tmp_path / 'run'))
    train(config)
    first = torch.load(tmp_path / 'run/ckpt.pt', map_location='cpu', weights_only=False)
    assert first['step'] == 2
    assert first['config']['adc_bits'] == 9
    metrics_path = tmp_path / 'run/metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [row['step'] for row in metrics] == [0, 1, 2]
    assert all(np.isfinite(row['val_loss']) for row in metrics)
    train(replace(config, max_iters=3), resume=True)
    resumed = torch.load(tmp_path / 'run/ckpt.pt', map_location='cpu', weights_only=False)
    assert resumed['step'] == 3
    assert resumed['config']['planned_updates'] == first['config']['planned_updates']
    assert resumed['scientific_config'] == first['scientific_config']
    # Restore does not duplicate the validation already recorded at step 2.
    assert len(metrics_path.read_text().splitlines()) == len(metrics)
