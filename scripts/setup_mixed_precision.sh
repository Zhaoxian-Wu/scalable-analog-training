#!/usr/bin/env bash
# Install the external, unpackaged Python module without vendoring its source.
set -euo pipefail
training_python=${PYTHON:-python3}
dependency_prefix=${1:-${XDG_DATA_HOME:-$HOME/.local/share}/scalable-analog-training}
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
dependency_dir="$dependency_prefix/MixedPrecisionOptim"
mapfile -t dependency_spec < <("$training_python" - "$script_root/dependencies.json" <<'PY'
import json
import sys
spec = json.load(open(sys.argv[1]))['MixedPrecisionOptim']
print(spec['repository'])
print(spec['ref'])
print(spec['commit'])
PY
)
mkdir -p -- "$dependency_prefix"
if [[ ! -e "$dependency_dir" ]]; then
    git clone --depth 1 --branch "${dependency_spec[1]}" "${dependency_spec[0]}" "$dependency_dir"
fi
actual_commit=$(git -C "$dependency_dir" rev-parse HEAD)
if [[ -n "$(git -C "$dependency_dir" status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: $dependency_dir contains modified tracked files; use a clean external checkout." >&2
    exit 1
fi
if [[ "$actual_commit" != "${dependency_spec[2]}" ]]; then
    echo "ERROR: $dependency_dir is at $actual_commit; required ${dependency_spec[2]}." >&2
    echo "Use a fresh dependency directory, or explicitly update dependencies.json after validation." >&2
    exit 1
fi
"$training_python" - "$dependency_dir" <<'PY'
import pathlib
import sys
import sysconfig
module = pathlib.Path(sys.argv[1]).resolve()
if module.name != 'MixedPrecisionOptim' or not (module / 'MpOptimizerMixin.py').is_file():
    raise SystemExit('ERROR: expected the unified MixedPrecisionOptim package.')
pth = pathlib.Path(sysconfig.get_path('purelib')) / 'scalable_analog_training_mp.pth'
pth.write_text(str(module.parent) + '\n', encoding='utf-8')
print(f'External optimizer: {module}')
print(f'Python search path registered in {pth}')
PY
