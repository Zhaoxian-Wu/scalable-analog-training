"""Export the frozen numerical inputs of the manuscript's four gradient panels."""
import argparse
import hashlib
import json
import math
from pathlib import Path

WEBSITE = Path(__file__).resolve().parent
SOURCE = Path('outputs/2026-09-24-precision-cancellation-compact/cache/'
              'plot_data_four_panels_all_seed_ranges_1-2-3-4.json')
FIELDS = {
    'panel_a': ('target', 'precision', 'validation_loss'),
    'panel_b': ('kind', 'mean_kappa'),
    'panel_c': ('format', 'mode', 'log10_kappa_left', 'log10_kappa_right', 'relative_l2_error'),
    'panel_d': ('label', 'micro_batch_size', 'method', 'relative_error'),
}
METRICS = dict(zip(FIELDS, ('validation_loss', 'mean_kappa', 'relative_l2_error', 'relative_error')))


def export(root):
    source = root / SOURCE
    raw = json.loads(source.read_text())
    assert raw['schema_version'] == 1
    assert raw['all_panel_seed_ranges']['seeds'] == [1, 2, 3, 4]
    assert raw['panel_b']['value_key'] == 'mean_kappa'
    assert raw['panel_d']['config']['batch_size'] == 128
    data = {'source': str(SOURCE), 'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'seeds': [1, 2, 3, 4], 'rangeNote': raw['cross_seed']['summary']}
    for panel, fields in FIELDS.items():
        rows = [r for r in raw[panel]['rows'] if panel != 'panel_c' or r['mode'] == 'acc']
        for row in rows:
            values = row['seed_values']
            assert len(values) == 4 and all(math.isfinite(v) for v in values)
            assert math.isclose(row[METRICS[panel]], sum(values) / 4, rel_tol=1e-12)
            assert row['seed_min'] == min(values) and row['seed_max'] == max(values)
        data[panel] = [{key: row[key] for key in (*fields, 'seed_min', 'seed_max', 'seed_values')}
                       for row in rows]
    assert [len(data[p]) for p in FIELDS] == [6, 6, 21, 6]
    destination = WEBSITE / 'assets/gradient-results-data.json'
    destination.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(f'Exported 39 measurements with verified four-seed means and ranges: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path)
    args = parser.parse_args()
    candidates = [args.research_root.resolve()] if args.research_root else WEBSITE.parents
    root = next((p for p in candidates if (p / SOURCE).is_file()), None)
    if root is None:
        parser.error('Frozen figure data not found; pass --research-root /path/to/analog-transformer')
    export(root)
