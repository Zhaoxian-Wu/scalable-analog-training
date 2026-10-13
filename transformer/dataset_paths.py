"""Resolve portable nanoGPT datasets without searching workstation directories."""
import os
from pathlib import Path


def resolve_dataset_dir(dataset_name, *, dataset_dir='', dataset_root='', local_data_root=None):
    if dataset_dir:
        candidates = [Path(dataset_dir).expanduser()]
    else:
        roots = [Path(local_data_root) if local_data_root else Path(__file__).resolve().parent / 'data']
        if dataset_root:
            roots.append(Path(dataset_root).expanduser())
        env_root = os.environ.get('SCALABLE_ANALOG_DATA_ROOT')
        if env_root:
            roots.append(Path(env_root).expanduser())
        names = [dataset_name]
        if dataset_name == 'owt_full':
            names.append('openwebtext')
        elif dataset_name == 'openwebtext':
            names.append('owt_full')
        candidates = [root / name for root in roots for name in names]
    required = ('train.bin', 'val.bin', 'meta.pkl')
    for path in candidates:
        if all((path / name).is_file() for name in required):
            return str(path.resolve())
    checked = '\n  - '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'Dataset {dataset_name!r} needs train.bin, val.bin, and meta.pkl. '
        f'Run its prepare.py or set --dataset_dir / SCALABLE_ANALOG_DATA_ROOT. Checked:\n  - {checked}'
    )
