"""Synthetic arithmetic and completeness gate; no GPU or timing."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('m4_analysis', Path(__file__).parents[1]/'studies/m4_nms_dispatch/analyze.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_paired_ratios_and_incomplete_gate(tmp_path):
    manifest = dict(config=dict(experiment='M4',smoke=True,n_images=2,repeats=3), images={'a.jpg':'a','b.jpg':'b'})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    rows = []
    for image in manifest['images']:
        for arm in ('clean','attacked'):
            for stage in ('nms_tv','nms_pure','mask_False','mask_True','total_tv_False',
                          'total_tv_True','forward','aa_total_tv_A','aa_total_tv_B'):
                for rep in range(3):
                    ns = (2 if stage=='nms_pure' else 1)*1000000
                    rows.append(dict(kind='timing', image=image, arm=arm, stage=stage, rep=rep, ns=ns))
    rows.append(dict(kind='done'))
    raw = tmp_path/'raw.jsonl'
    raw.write_text('\n'.join(map(json.dumps, rows)))
    result = mod.analyze(tmp_path)
    assert result['quality_pass']
    assert result['primary_pure_over_tv']['geometric_ratio'] == 2.0
    assert result['mde']['clean']['mde_frac'] == 0.0
    raw.write_text('\n'.join(map(json.dumps, rows[1:])))
    assert not mod.analyze(tmp_path)['complete']
