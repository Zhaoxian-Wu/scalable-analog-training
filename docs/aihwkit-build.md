# AIHWKit beta source build

The root README supplies manual composition commands for `beta-release`.
The authoritative development list is the AIHWKit checkout's
`.claude/beta-branches.yaml`; this repository retains an ordered snapshot in
`dependencies/aihwkit-beta-branches.yaml`. Merge from `upstream/master` with
one `--no-ff` merge per active branch, in list order. Commented branches are
excluded. Verify all source refs first. Missing branches must be published or
obtained from the maintainer; never substitute another branch automatically.

## Record the composed revision

After composition, from the AIHWKit checkout:

```bash
git rev-parse HEAD
git rev-parse upstream/master
git log --first-parent --format='%H %s' upstream/master..beta-release
```

Record the base, each input branch commit, final beta commit, environment, and
build command with experiment results. Branch tips can change. The input SHAs
observed during packaging are recorded in `dependencies.json`; use those SHAs
instead of moving branch names when reproducing that source composition.
The resulting merge SHA also depends on merge metadata, so API/tests and input
commits matter in addition to the branch name.

## Build prerequisites

Use one environment for PyTorch, compilation, and training. Install the Python
build tools shown in the README, a compatible C++ compiler, and OpenBLAS headers
and libraries. GPU builds need a CUDA Toolkit with `nvcc`; the toolkit must
support both the host compiler and target GPU architecture. A PyTorch CUDA wheel
does not supply a full development toolkit.

The wrapper uses `CMAKE_BUILD_TYPE=Release`, `USE_CUDA`, `RPU_BLAS`, and
`RPU_CUDA_ARCHITECTURES`. It includes the current PyTorch CMake prefix and Python
environment prefix in `CMAKE_PREFIX_PATH`. Set `CC`, `CXX`, `CUDAHOSTCXX`, and
`CUDA_HOME` as appropriate; there are no workstation-specific paths in the script.

From the training repository root, after composing `../aihwkit-beta`:

```bash
PYTHON="$TRAINING_PYTHON" bash scripts/build_aihwkit.sh \
  --source-dir ../aihwkit-beta --ref beta-release \
  --blas OpenBLAS --arch '80;90' --jobs 1
```

The existing checkout must be clean and already at `beta-release` (or the SHA
passed with `--ref`). The script refuses to switch or overwrite an existing
checkout. If no checkout exists, it can clone a published ref; this does not
compose unpublished development branches. Build in a fresh checkout/cache when
changing Python, PyTorch, CUDA, or compiler ABI.

The equivalent wheel build, from the composed AIHWKit directory, is:

```bash
export TORCH_CMAKE_PREFIX="$("$TRAINING_PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DRPU_BLAS=OpenBLAS -DRPU_CUDA_ARCHITECTURES=80;90 -DCMAKE_PREFIX_PATH=$TORCH_CMAKE_PREFIX;$CONDA_PREFIX"
export CMAKE_BUILD_PARALLEL_LEVEL=1
"$TRAINING_PYTHON" -m pip wheel --no-build-isolation --no-deps . --wheel-dir wheelhouse
"$TRAINING_PYTHON" -m pip install --no-deps --force-reinstall wheelhouse/aihwkit-*.whl
```

Use an empty wheelhouse to avoid selecting old wheels. `--no-build-isolation`
compiles against the installed PyTorch; `--no-deps` prevents its replacement.
Install AIHWKit's runtime requirements first, as documented in the README.

## Architectures, CPU, and MKL

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Remove the decimal point to obtain an architecture value (`8.0` becomes `80`).
For several GPU generations, pass a semicolon-separated list. Newer targets
may require a newer toolkit and PyTorch wheel. CPU builds use:

```bash
PYTHON="$TRAINING_PYTHON" bash scripts/build_aihwkit.sh \
  --source-dir ../aihwkit-beta --ref beta-release --cpu --blas OpenBLAS --jobs 1
```

For MKL, install `mkl`, `mkl-include`, and its OpenMP runtime, set
`MKLROOT=$CONDA_PREFIX`, and select `--blas MKL`. Keep `$CONDA_PREFIX/lib` on
`LD_LIBRARY_PATH` to resolve MKL libraries. The local development installation
uses MKL; OpenBLAS is offered as a portable build alternative.

## Verify

After installing external MixedPrecisionOptim, from the training repository:

```bash
"$TRAINING_PYTHON" scripts/check_environment.py --cuda
```

Omit `--cuda` for CPU builds. The checker validates native gradient APIs and
`use_pre_update`, imported paths, and Adam optimizer checkpoint
format. CUDA mode performs a simulated tile forward/backward/update; PyTorch
CUDA availability alone does not establish an AIHWKit CUDA build.

| Failure | Action |
| --- | --- |
| Source branch missing | Obtain the exact branch/commit; stop before composition. |
| Merge conflict | Abort the merge and resolve/rebase the source branches before rebuilding. |
| `nvcc` missing | Install a CUDA Toolkit and add its `bin/` to `PATH`. |
| Unsupported architecture/compiler | Choose a compatible toolkit and host compiler. |
| BLAS missing | Install headers and libraries and set the environment prefix. |
| MKL library import error | Add the environment's `lib/` to `LD_LIBRARY_PATH`. |
| Missing `AnalogContext` APIs | Inspect `aihwkit.__file__` for an older build shadowing beta. |
| Extension fails after upgrading PyTorch | Rebuild AIHWKit against the new environment. |
