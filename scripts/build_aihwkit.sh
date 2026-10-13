#!/usr/bin/env bash
# Build the required AIHWKit fork into a wheel and install it in this Python env.
set -euo pipefail
training_python=${PYTHON:-python3}
aihwkit_ref=beta-release
source_dir=${XDG_DATA_HOME:-$HOME/.local/share}/scalable-analog-training/aihwkit
blas_backend=OpenBLAS
cuda_enabled=ON
cuda_archs=
build_jobs=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) aihwkit_ref=$2; shift 2 ;;
        --source-dir) source_dir=$2; shift 2 ;;
        --blas) blas_backend=$2; shift 2 ;;
        --arch) cuda_archs=$2; shift 2 ;;
        --jobs) build_jobs=$2; shift 2 ;;
        --cpu) cuda_enabled=OFF; shift ;;
        --help) echo 'Usage: build_aihwkit.sh [--ref REF] [--source-dir DIR] [--blas OpenBLAS|MKL] [--arch "80;90"] [--jobs N] [--cpu]'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ "$blas_backend" != OpenBLAS && "$blas_backend" != MKL ]]; then
    echo 'ERROR: --blas must be OpenBLAS or MKL.' >&2; exit 2
fi
if [[ "$cuda_enabled" == ON && -z "$cuda_archs" ]]; then
    echo 'ERROR: pass --arch for your GPU(s), for example --arch "80;90".' >&2; exit 2
fi
aihwkit_repo=https://github.com/Zhaoxian-Wu/aihwkit.git
if [[ ! -e "$source_dir" ]]; then
    mkdir -p -- "$(dirname -- "$source_dir")"
    # Fetch the requested ref explicitly. Missing beta-release must fail; never
    # substitute master or a PyPI wheel for the required native gradient APIs.
    git clone --no-checkout "$aihwkit_repo" "$source_dir"
    git -C "$source_dir" fetch origin "$aihwkit_ref"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
expected_commit=$(git -C "$source_dir" rev-parse "${aihwkit_ref}^{commit}" 2>/dev/null || git -C "$source_dir" rev-parse "origin/${aihwkit_ref}^{commit}")
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
    echo "ERROR: existing checkout $source_dir is not at $aihwkit_ref; refusing to change it." >&2; exit 1
fi
if [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=no)" ]]; then
    echo 'ERROR: AIHWKit has modified tracked files; build a clean checkout for reproducibility.' >&2; exit 1
fi
source_dir=$(cd -- "$source_dir" && pwd)
torch_prefix=$("$training_python" -c 'import torch; print(torch.utils.cmake_prefix_path)')
env_prefix=$("$training_python" -c 'import sys; print(sys.prefix)')
cmake_prefix="$torch_prefix;$env_prefix"
if [[ -n "${CMAKE_PREFIX_PATH:-}" ]]; then
    cmake_prefix="$cmake_prefix;$CMAKE_PREFIX_PATH"
fi
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=$cuda_enabled -DRPU_BLAS=$blas_backend -DCMAKE_PREFIX_PATH=$cmake_prefix"
if [[ "$blas_backend" == MKL ]]; then
    export CMAKE_ARGS="$CMAKE_ARGS -DINTEL_MKL_DIR=${MKLROOT:-$env_prefix}"
fi
if [[ "$cuda_enabled" == ON ]]; then
    command -v nvcc >/dev/null || { echo 'ERROR: CUDA Toolkit nvcc is missing from PATH.' >&2; exit 1; }
    export CMAKE_ARGS="$CMAKE_ARGS -DRPU_CUDA_ARCHITECTURES=$cuda_archs"
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$build_jobs"
wheel_dir=$(mktemp -d)
trap 'rm -rf -- "$wheel_dir"' EXIT
echo "AIHWKit ref=$aihwkit_ref commit=$actual_commit"
echo "CMAKE_ARGS=$CMAKE_ARGS"
"$training_python" -m pip wheel --no-build-isolation --no-deps "$source_dir" --wheel-dir "$wheel_dir"
"$training_python" -m pip install --no-deps --force-reinstall "$wheel_dir"/aihwkit-*.whl
"$training_python" -c 'import aihwkit; from aihwkit.simulator.rpu_base import cuda; print(aihwkit.__file__); print("AIHWKit CUDA compiled:", cuda.is_compiled())'
