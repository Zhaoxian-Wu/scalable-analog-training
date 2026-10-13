#!/usr/bin/env python3
"""Check the external optimizer and native AIHWKit gradient API before training."""
import argparse
import inspect
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true', help='also exercise a CUDA training tile')
    args = parser.parse_args()
    try:
        import torch
        import aihwkit
        import MixedPrecisionOptim
        from MixedPrecisionOptim.AdamW import MpAdamW
        from MixedPrecisionOptim.MpOptimizerMixin import MP_STATE_VERSION
        from aihwkit.optim.context import AnalogContext
        from aihwkit.simulator.rpu_base import cuda
    except (ImportError, OSError) as exc:
        raise SystemExit(
            f'ERROR: {exc}\nInstall the required AIHWKit beta-release and external '
            'MixedPrecisionOptim test-v3; see README.md.'
        ) from exc
    required = ('enable_buffer', 'set_digital_grad_accumulation', 'accumulate_digital_gradient')
    missing = [name for name in required if not hasattr(AnalogContext, name)]
    if missing:
        raise SystemExit(f'ERROR: wrong AIHWKit build; missing AnalogContext APIs: {missing}')
    if 'use_pre_update' not in inspect.signature(AnalogContext.set_digital_grad_accumulation).parameters:
        raise SystemExit('ERROR: AIHWKit lacks the required use_pre_update gradient coordinate API.')
    print(f'PyTorch: {torch.__version__}')
    print(f'AIHWKit: {aihwkit.__file__}')
    print(f'AIHWKit CUDA compiled: {cuda.is_compiled()}')
    module_dir = Path(MixedPrecisionOptim.__file__).resolve().parent
    print(f'MixedPrecisionOptim: {module_dir}')
    print(f'MP checkpoint format: {MP_STATE_VERSION}')
    if (module_dir / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(module_dir), 'rev-parse', 'HEAD'], text=True).strip()
        print(f'MixedPrecisionOptim commit: {revision}')
    if args.cuda:
        if not torch.cuda.is_available() or not cuda.is_compiled():
            raise SystemExit('ERROR: CUDA-enabled PyTorch, AIHWKit, and an accessible NVIDIA GPU are required.')
        from aihwkit.nn import AnalogLinear
        from aihwkit.simulator.configs import FloatingPointRPUConfig
        layer = AnalogLinear(4, 3, rpu_config=FloatingPointRPUConfig()).cuda()
        optimizer = MpAdamW(layer.parameters(), lr=0.001, weight_decay=0)
        loss = layer(torch.ones(2, 4, device='cuda')).square().mean()
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise SystemExit('ERROR: nonfinite CUDA tile loss.')
        print('CUDA forward/backward/update: PASS')
    print('Environment API check: PASS')


if __name__ == '__main__':
    main()
