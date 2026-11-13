# ImageNet-1K ResNet training

This directory releases the training source used for the paper's ImageNet ResNet depth comparison. It contains the digital SGD reference and the frozen Native MP SGD and SAMP (selected S-OUT SGD) sources for ResNet-18, -34, and -50. ResNet-101 source is included for provenance, but its analog comparison was not completed. ImageNet images, checkpoints, logs, and W&B credentials are not included.

## Contents and provenance

- `digital_reference/` is the digital baseline source, adapted from the TorchVision classification reference at `pytorch/vision@336d36e8db990a905498c73933e35231876e28bc`; its license is retained in that directory.
- `frozen_source/code/` is a deduplicated pool of the Python files captured with the analog runs. It includes the legacy `MixedPrecisionOptim` modules required by these runs and the TorchVision helper files, with their upstream license.
- `source_manifest.json` records the original run directory, W&B run ID, complete per-recipe source file list, individual file SHA256 checksums, and the source digest stored in the analog checkpoints. The ResNet-18 SAMP final run also records its separate continuation driver.
- `materialize.py` restores a specific historical `source/code/` tree and checks its digest. Its default behavior preserves the original source bytes, including historical machine-specific path defaults.

The source pool is deliberately frozen: several scripts have overlapping names but different model-specific behavior. Use the materializer to get only the files associated with one run. For a new method, copy a materialized tree and edit that copy; edits change the checkpoint source digest.

## Environment and data

Use a CUDA PyTorch environment with matching `torchvision`, NumPy, W&B for logged analog runs, and the project's [AIHWKit beta fork](../README.md#aihwkit-beta-build). The analog source was written against the experiment-time AIHWKit API; a later fork revision may need compatibility work. `MixedPrecisionOptim` is included in each materialized analog source tree, so the current Transformer optimizer checkout is not substituted for its historical version.

Point `--data-path` at the extracted **full** ImageNet-1K directory:

```text
/path/to/imagenet/
  train/<1000 class folders>/*.JPEG
  val/<1000 class folders>/*.JPEG
```

The analog drivers check for 1,281,167 training images and 50,000 validation images. They use the TorchVision reference transforms and a 90-epoch schedule with learning-rate drops every 30 epochs. All shown runs use batch size 64 and seed 1337. Digital runs use AMP; analog runs use FP32.

The frozen analog scripts read a W&B API key from `/home/zw868/.config/wandb/imagenet.key` and log to `analog-transformer/ImageNet`. On another machine, change that path in a **copy** of the materialized source or create the corresponding key file. The SAMP drivers always initialize W&B; Native MP SGD does so with `--wandb`. Use a new `--wandb-id` for each new experiment. Changing a source file changes the recorded digest, so a historical checkpoint cannot be resumed after that edit without adapting its provenance check.

## Restore and run

Run these commands from the repository root. Replace `/path/to/imagenet`, `/path/to/output`, and the example W&B IDs. The materializer refuses a nonempty output directory and verifies the historical source digest before and after copying.

```bash
python imagenet/materialize.py resnet50-samp --verify-only
python imagenet/materialize.py resnet50-samp --output-dir /path/to/output/resnet50-samp
```

The digital baseline uses the same source for all depths. For example, ResNet-50:

```bash
python imagenet/digital_reference/train.py \
  --data-path /path/to/imagenet --model resnet50 --seed 1337 \
  --epochs 90 --batch-size 64 --lr 0.025 --workers 8 --amp \
  --save-every 10 --output-dir /path/to/output/digital-resnet50
```

This is SGD with momentum 0.9 and weight decay 1e-4, the reference defaults. Change `--model` to `resnet18` or `resnet34` for the other reported digital depths.

For Native MP SGD, restore `resnet18-mp-sgd`, `resnet34-mp-sgd`, or `resnet50-mp-sgd`. This example uses ResNet-50:

```bash
python imagenet/materialize.py resnet50-mp-sgd --output-dir /path/to/output/resnet50-mp
PYTHONPATH=/path/to/output/resnet50-mp/source/code${PYTHONPATH:+:$PYTHONPATH} \
python /path/to/output/resnet50-mp/source/code/imagenet_reference/train_full_analog_resnet50.py \
  --mode native_mp --data-path /path/to/imagenet \
  --output-dir /path/to/output/resnet50-mp/run --epochs 90 \
  --batch-size 64 --workers 8 --seed 1337 --lr 0.025 \
  --momentum 0 --weight-decay 0.0001 --dac-bits 16 --adc-bits 16 \
  --write-profile ecram --wandb --wandb-id my-new-resnet50-mp-run
```

For SAMP, restore `resnet34-samp` or `resnet50-samp`, then run its `train_resnet{depth}_sout.py` driver. The selected ResNet-50 recipe is:

```bash
python /path/to/output/resnet50-samp/source/code/imagenet_reference/train_resnet50_sout.py \
  --source /path/to/output/resnet50-samp/source/code \
  --data-path /path/to/imagenet --output-dir /path/to/output/resnet50-samp/run \
  --epochs 90 --batch-size 64 --workers 8 \
  --wandb-id my-new-resnet50-samp-run
```

Its implementation starts both digital and analog parameter groups at learning rate 0.025, remaps the analog weights by a factor of 10/3 after epoch 1, and reduces the analog learning rate to 0.0025. Momentum is 0.9, weight decay is zero, and the device model uses 16-bit DAC/ADC with ECRAM. For ResNet-34, use `resnet34-samp` and `train_resnet34_sout.py` with the same argument pattern.

**ResNet-18 SAMP was a two-stage run.** First, `resnet18-samp`'s `train_full_sout_ls_wd0_nomask_resnet18.py` produced an epoch-1 checkpoint with the earlier learning-rate settings. The selected 90-epoch result then resumed that checkpoint using `entrypoints/resnet18_samp_continuation.py`, which applies the weight remap and reduced analog learning rate after epoch 1. A new run can generate the first-stage checkpoint with `--epochs 1` and then resume it:

```bash
python imagenet/materialize.py resnet18-samp --output-dir /path/to/output/resnet18-samp
PYTHONPATH=/path/to/output/resnet18-samp/source/code${PYTHONPATH:+:$PYTHONPATH} \
python /path/to/output/resnet18-samp/source/code/imagenet_reference/train_full_sout_ls_wd0_nomask_resnet18.py \
  --mode s_out --data-path /path/to/imagenet \
  --output-dir /path/to/output/resnet18-samp/first-stage \
  --epochs 1 --batch-size 64 --workers 8 --seed 1337 --lr 0.025 \
  --momentum 0.9 --weight-decay 0 --dac-bits 16 --adc-bits 16 \
  --write-profile ecram --wandb-id my-new-resnet18-first-stage
python imagenet/entrypoints/resnet18_samp_continuation.py \
  --source /path/to/output/resnet18-samp/source/code \
  --checkpoint /path/to/output/resnet18-samp/first-stage/checkpoint.pth \
  --data-path /path/to/imagenet \
  --output-dir /path/to/output/resnet18-samp/continuation \
  --epochs 90 --batch-size 64 --workers 8 \
  --wandb-id my-new-resnet18-samp-run
```

The epoch-1 checkpoint from the published run is not in this repository. An exact replay of the published continuation requires that checkpoint and the experiment-time dependencies; the continuation driver is not a standalone from-scratch trainer. The source manifest records both W&B run IDs and the continuation driver SHA256.

All analog drivers retain `checkpoint.pth` every epoch, and the digital reference retains `checkpoint.pth` plus optional periodic checkpoints. To resume, use the respective driver's `--resume` option (the ResNet-18 SAMP continuation driver uses `--checkpoint`). Keep the same materialized source and a unique output directory.

## Reported 90-epoch validation accuracy

| Method | ResNet-18 | ResNet-34 | ResNet-50 |
| --- | ---: | ---: | ---: |
| Digital SGD | 69.976% | 73.492% | 75.898% |
| Native MP SGD | 62.766% | 65.674% | 66.580% |
| SAMP | 66.768% | 69.770% | 69.888% |

These are full ImageNet-1K validation top-1 percentages from the completed single-seed runs, not results recomputed by this source release. The analog W&B run IDs and exact source digests are in `source_manifest.json`; the digital run's local provenance is recorded there too. ResNet-101 SAMP was paused after epoch 8 and ResNet-101 Native MP SGD was not started, so neither has a 90-epoch value.
