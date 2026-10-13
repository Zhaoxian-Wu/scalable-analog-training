# Scalable Analog Training

Research code for scalable training with analog in-memory computing (AIMC). Analog crossbar
arrays execute matrix passes; digital optimizers accumulate gradients and
control quantized weight updates. This repository contains architecture-level
AIHWKit simulations, not measurements from a fabricated chip.

The initial workload is the Transformer Adam scaling comparison in
Figure `fig:scaling`, with all task-specific files under [`transformer/`](transformer/README.md).
The code is still being organized. More code will be released in future versions,
with other training workloads added in separate folders.
Optimizer implementations are an external dependency from
[MixedPrecisionOptim](https://github.com/Zhaoxian-Wu/MixedPrecisionOptim);
their sources are not copied into this repository.

## Dependencies and installation

Use Python 3.10+ and a CUDA-compatible PyTorch build matching AIHWKit.
Linux with an NVIDIA GPU is the intended training platform. Install a
CUDA-compatible PyTorch build before compiling AIHWKit in the same environment.
For example, using a dedicated Conda environment and a system CUDA 12.6 Toolkit:

```bash
conda create -n scalable-analog-training -c conda-forge python=3.11 pip openblas
export CONDA_PREFIX="$(conda info --base)/envs/scalable-analog-training"
export TRAINING_PYTHON="$CONDA_PREFIX/bin/python"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
"$TRAINING_PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126
"$TRAINING_PYTHON" -m pip install -r transformer/requirements-dev.txt
"$TRAINING_PYTHON" -m pip install setuptools wheel 'scikit-build>=0.18.1' \
  'pybind11>=3.0.1' 'cmake>=4.2.1' ninja mypy scikit-learn scipy protobuf
```

Set `CONDA_PREFIX` to the actual environment directory if Conda uses a custom
location. Select CUDA/PyTorch versions compatible with your driver, GPU and host
compiler. A CUDA PyTorch wheel supplies runtime libraries; compiling AIHWKit
also requires a CUDA Toolkit with `nvcc`, a C++ compiler, and BLAS headers.

### Compose and compile AIHWKit beta-release

**Use the beta version of [Zhaoxian-Wu/aihwkit](https://github.com/Zhaoxian-Wu/aihwkit).**
Here, `beta-release` means a local branch composed by merging the branches in
AIHWKit's `.claude/beta-branches.yaml`, in order. It is not a published upstream
package. The ordered list used for this release is retained in
[dependencies/aihwkit-beta-branches.yaml](dependencies/aihwkit-beta-branches.yaml).
Do not replace this build with `pip install aihwkit` or an upstream wheel: the
trainer requires the fork's native buffer and digital-gradient APIs.

From this repository's root, clone AIHWKit outside the release directory and
compose the beta manually:

```bash
export TRAINING_REPO="$PWD"
git clone https://github.com/Zhaoxian-Wu/aihwkit.git ../aihwkit-beta
cd ../aihwkit-beta
git remote add upstream https://github.com/IBM/aihwkit.git
git fetch upstream master
git fetch origin

# Merge order from .claude/beta-branches.yaml; commented entries are excluded.
BETA_BRANCHES=(
  feat/incremental-analog-grad
  fix/cuda-large-mbatch-int64
  feat/backward-BM
  feat/dw-min-zero-infinite-granularity
  perf/parallel-device-initialization
)

# Verify every branch before creating the beta. Source branches must be
# available on origin; fetch them from their maintainer if not yet published.
for branch in "${BETA_BRANCHES[@]}"; do
  git rev-parse --verify "origin/$branch" || exit 1
done
git switch -c beta-release upstream/master
for branch in "${BETA_BRANCHES[@]}"; do
  git merge --no-ff -m "beta: merge $branch" "origin/$branch" || {
    git merge --abort
    exit 1
  }
done
git rev-parse HEAD
cd "$TRAINING_REPO"

export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
PYTHON="$TRAINING_PYTHON" bash scripts/build_aihwkit.sh \
  --source-dir ../aihwkit-beta --ref beta-release \
  --blas OpenBLAS --arch '80;90' --jobs 1
```

Run the shell block in Bash. Git needs your normal author identity to create
merge commits. For an existing development checkout, use its local branch names
instead of `origin/<branch>`; create the beta in a fresh clone or worktree so
ongoing work is preserved. Stop on conflicts and resolve the source branches
before retrying. The commands do not delete or reset source branches.

Some listed development branches were local-only when this directory was
prepared. Publish/fetch those refs before following the fresh-clone recipe.
The YAML list alone does not make unpublished source commits available.

Replace `80;90` with your GPU architecture values. For CPU builds use `--cpu`
instead of `--arch`; for MKL use `--blas MKL` after installing MKL headers and
libraries. The script prints the composed commit, builds a wheel against the
installed PyTorch, and installs it into the same Python environment.
See [build details and troubleshooting](docs/aihwkit-build.md).

### Install external MixedPrecisionOptim

Use tag **`test-v3`**, pinned at
`c7c01a856a4c6283e3681a95df611ec35b127005`. Its default `master` exposes an
older API. The tag has no pip packaging metadata, so the helper clones the
package outside this repository and adds its parent directory to the Python
environment with a `.pth` file:

```bash
PYTHON="$TRAINING_PYTHON" bash scripts/setup_mixed_precision.sh
"$TRAINING_PYTHON" scripts/check_environment.py --cuda
```

The default dependency location is `~/.local/share/scalable-analog-training/`
(or under `XDG_DATA_HOME`). Pass a different parent directory to the helper if
needed. Keep the external checkout named `MixedPrecisionOptim`. The environment
checker prints imported paths and the optimizer checkpoint version, checks the
required native gradient APIs, and exercises a CUDA forward/backward/update.

## Transformer Adam scaling

This release includes only the OpenWebText Adam training comparison for
`fig:scaling`: digital AdamW and S-OUT analog AdamW, from Tiny (6.8M) through
XLarge (123.6M). It follows the current appendix's **16-bit DAC / 9-bit ADC**
setting. The shared schedule uses D=20N, 480 length-1,024 sequences per update,
seed 1337, and zero weight decay.

Run the basic commands from the Transformer task folder:

```bash
cd transformer
"$TRAINING_PYTHON" data/openwebtext/prepare.py
"$TRAINING_PYTHON" -u train.py config/digital_adam.py
"$TRAINING_PYTHON" -u train.py config/analog_adam.py
```

Both default to Tiny on a CUDA GPU. See the [Transformer training notes](transformer/README.md)
for the model sizes, scientific settings, dataset format, and correctness tests.
Detailed launch commands will be provided in the next version.

## Layout

```text
transformer/          Adam scaling workload: training, model, data, configs, requirements, tests
scripts/              Shared external dependency setup and environment checks
dependencies/         Ordered AIHWKit beta branch list
docs/                 Shared AIHWKit build instructions
dependencies.json     External refs and pinned optimizer revision
```

Datasets, outputs, checkpoints, inventories, and generated builds are untracked.

## Third-party notices

The Transformer model and OpenWebText preparation code derive from
[Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), under MIT.
The original copyright and license are retained in `transformer/LICENSE` and
the root `LICENSE`. AIMC integrations derive from this project's research code.

[AIHWKit](https://github.com/Zhaoxian-Wu/aihwkit) and
[MixedPrecisionOptim](https://github.com/Zhaoxian-Wu/MixedPrecisionOptim) are
external dependencies with their own licensing; their sources are not vendored.

## Related project material

- [Project homepage and source (`website` branch)](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/website)
- [Presentation source (`slides` branch)](https://github.com/Zhaoxian-Wu/scalable-analog-training/tree/slides)
