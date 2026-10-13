# Transformer Adam scaling training

This folder contains the training workload for `fig:scaling` in the manuscript:
five decoder-only GPT models on OpenWebText, comparing digital AdamW with
S-OUT analog AdamW. The first release follows the current appendix's **16-bit
DAC / 9-bit ADC** setting. Earlier figure assets used 16-bit ADCs, so the current
configuration should not be described as a bit-for-bit replay of those runs.

All Transformer-specific source, configs, Python requirements, data preparation,
and tests live here. Install the shared native dependencies as described in the
repository-root README, then run these basic commands from this folder:

```bash
"$TRAINING_PYTHON" data/openwebtext/prepare.py
"$TRAINING_PYTHON" -u train.py config/digital_adam.py
"$TRAINING_PYTHON" -u train.py config/analog_adam.py
```

Both default to the Tiny model on one CUDA GPU. Config files and
`scaling_config.py` contain the scientific settings. Detailed hardware-specific
launch commands will be supplied in the next version.

| Scale | Blocks | Width | Heads | Budget parameters | Updates | Last scheduled validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny | 2 | 128 | 4 | 6,826,752 | 278 | 273 |
| Small | 4 | 256 | 4 | 16,013,824 | 652 | 640 |
| Medium | 6 | 384 | 6 | 29,920,512 | 1,218 | 1,200 |
| Large | 8 | 512 | 8 | 50,906,112 | 2,072 | 2,060 |
| XLarge | 12 | 768 | 12 | 123,551,232 | 5,028 | 5,020 |

The shared recipe uses a 50,257-token GPT-2 vocabulary, length-1,024 contexts,
480 sequences per update, seed 1337, and a D=20N budget. AdamW uses
`betas=(0.9, 0.95)`, zero weight decay, and gradient clipping at 1.0. Linear warm-up
is followed by cosine decay from `6e-4` to `6e-5`. Each validation averages ten
batches; the last recorded validation can precede the last optimizer update.

Digital training uses BF16 autocasting on CUDA. Analog training uses float32,
Li-ECRAM tiles, native input management, forward iterative bound management,
and matrix-specific rails (`c_out=6`). The vocabulary head shares its array with
token embeddings, while positions, normalization, softmax, attention-score
products, and nonlinear operations remain digital. Arrays have no size tiling
limit in this simulation. Digital logical gradients and quantized writes are
handled by the external `MixedPrecisionOptim.AdamW.MpAdamW` adapter.

Prepare `train.bin` and `val.bin` as `uint16` token streams plus `meta.pkl` with
`vocab_size=50257`. Local `data/openwebtext/` is searched by default; use
`SCALABLE_ANALOG_DATA_ROOT` or an explicit dataset directory for existing data.
W&B is opt-in in the config. Results, resolved configuration, JSONL validation
metrics, and checkpoints are written to `outputs/<scale>-<mode>-adam/` and are
excluded from Git. Checkpoints from this entry point have format
`adam-scaling-v1`; legacy research checkpoints are not automatically migrated.

Run the workload's correctness tests from this folder:

```bash
"$TRAINING_PYTHON" -m pytest -q tests
```

Tests use small synthetic fixtures to check the scaling budgets, native mapped
gradients, tied weights, quantized writes, and save/restore behavior.
