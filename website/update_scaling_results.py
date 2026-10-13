"""Export the source data of manuscript Figure 'Scaling' for the homepage.

Uses the archived figure JSON and constants from its plotting script without
running experiments or regenerating manuscript assets. Standalone checkouts
use the bundled JSON; pass --research-root to refresh from a research workspace.
"""
import argparse
import ast
import json
import math
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent
SOURCE = Path('code/E2-scaling-law/assets/scaling_local_ddp2_results.json')
SCRIPT = Path('code/E2-scaling-law/plot_local_ddp2_scaling.py')


def export(root):
    raw = json.loads((root / SOURCE).read_text())
    wanted = {'SCALES', 'TOKENS_PER_STEP', 'VOCAB_SIZE', 'CONTEXT_LENGTH', 'PFDAY'}
    constants = {}
    for node in ast.parse((root / SCRIPT).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in wanted:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    assert constants.keys() == wanted, 'Figure constants missing'
    profiles = ['Digital AdamW', 'S-OUT analog (16-bit DAC/ADC)']
    data = {'source': str(SOURCE), 'script': str(SCRIPT),
            'tokensPerStep': constants['TOKENS_PER_STEP'],
            'fits': {}, 'models': []}
    for name, cfg in constants['SCALES'].items():
        ops = (6 * cfg['non_embedding'] + 6 * constants['VOCAB_SIZE'] * cfg['width']
               + 12 * cfg['layers'] * cfg['width'] * constants['CONTEXT_LENGTH'])
        model = {'id': name, 'label': cfg['label'], 'params': cfg['params'],
                 'computePerToken': ops / constants['PFDAY'], 'runs': {}}
        for profile, key in zip(profiles, ['digital', 'analog']):
            run = raw[profile][name]
            assert run['final_step'] == cfg['final_step'] == run['steps'][-1]
            assert run['final_loss'] == run['losses'][-1]
            assert run['final_tokens'] == run['final_step'] * constants['TOKENS_PER_STEP']
            assert len(run['steps']) == len(run['losses'])
            assert all(a < b for a, b in zip(run['steps'], run['steps'][1:]))
            model['runs'][key] = {k: run[k] for k in ['steps', 'losses', 'final_loss', 'final_step', 'final_tokens']}
        data['models'].append(model)
    for profile, key in zip(profiles, ['digital', 'analog']):
        data['fits'][key] = raw['preliminary_power_law_fits'][profile]
        # Verify that the saved fits still correspond to these five endpoints.
        for coordinate in ['params', 'tokens', 'compute']:
            x = [math.log(m['params'] if coordinate == 'params' else
                          m['runs'][key]['final_tokens'] * (m['computePerToken'] if coordinate == 'compute' else 1))
                 for m in data['models']]
            y = [math.log(m['runs'][key]['final_loss']) for m in data['models']]
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
            fit = data['fits'][key][coordinate]
            assert math.isclose(-slope, fit['exponent'], abs_tol=1e-10)
            assert math.isclose(math.exp(ym - slope * xm), fit['coefficient'], rel_tol=1e-10)
    destination = WEBSITE / 'assets/scaling-results-data.json'
    destination.write_text(json.dumps(data, indent=2) + '\n')
    print(f'Exported 10 runs, all recorded evaluations, and verified fits: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path)
    args = parser.parse_args()
    candidates = [args.research_root.resolve()] if args.research_root else WEBSITE.parents
    root = next((p for p in candidates if (p / SOURCE).is_file()), None)
    if root is None:
        parser.error('Source figure JSON not found; pass --research-root /path/to/analog-transformer')
    export(root)
