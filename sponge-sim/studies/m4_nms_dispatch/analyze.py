"""Read M4 raw data; write derived summary separately. No inference/timing."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def analyze(run):
    rows = [json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    manifest = json.loads((run/'manifest.json').read_text())
    groups = defaultdict(list)
    for r in rows:
        if r['kind'] == 'timing':
            groups[(r['image'], r['arm'], r['stage'])].append(r['ns']/1e6)
    meds = {k: float(np.median(v)) for k, v in groups.items()}
    images = sorted({k[0] for k in meds})
    def paired(stage_a, stage_b, arm_a='attacked', arm_b='attacked', ids=None):
        ids = images if ids is None else ids
        if not all((i,arm_a,stage_a) in meds and (i,arm_b,stage_b) in meds for i in ids):
            return None
        logs = np.array([np.log(meds[i, arm_a, stage_a]/meds[i, arm_b, stage_b]) for i in ids])
        if not len(logs):
            return None
        rng = np.random.default_rng(20261106)
        means = logs[rng.integers(0, len(logs), (20000, len(logs)))].mean(1)
        return dict(n_images=len(logs), geometric_ratio=float(np.exp(logs.mean())),
                    ci95=np.exp(np.quantile(means, [.025, .975])).tolist())
    flagged = [r for r in rows if r['kind'] == 'reference' and r['flagged']]
    flagged_blocks = {r['block'] for r in flagged}
    valid_ids = [name for ix, name in enumerate(manifest['images']) if ix not in flagged_blocks and name in images]
    expected_stages = {'nms_tv','mask_False','mask_True','total_tv_False',
                       'total_tv_True','forward','aa_total_tv_A','aa_total_tv_B'}
    if manifest['config']['experiment']=='M4': expected_stages.add('nms_pure')
    complete = (rows[-1]['kind'] == 'done' and len(images) == manifest['config']['n_images']
                and all(len(groups[i,arm,s]) == manifest['config']['repeats']
                        for i in manifest['images'] for arm in ('clean','attacked') for s in expected_stages))
    out = dict(smoke=manifest['config']['smoke'], complete=complete,
               raw_sha256=hashlib.sha256((run/'raw.jsonl').read_bytes()).hexdigest(),
               quality_pass=complete and len(flagged) <= 2, flagged_references=flagged,
               primary_pure_over_tv=paired('nms_pure', 'nms_tv'),
               sensitivity_pure_over_tv=paired('nms_pure', 'nms_tv', ids=valid_ids),
               counts=[r for r in rows if r['kind'] == 'counts'], mde={}, stages={}, contrasts={})
    for arm in ('clean', 'attacked'):
        aa = np.array([np.log(meds[i, arm, 'aa_total_tv_A']/meds[i, arm, 'aa_total_tv_B']) for i in images])
        n = len(aa)
        mde = (stats.t.ppf(.975,n-1)+stats.t.ppf(.8,n-1))*aa.std(ddof=1)/np.sqrt(n) if n>1 else None
        out['mde'][arm] = dict(n=n, log_mde=None if mde is None else float(mde),
                               mde_frac=None if mde is None else float(np.expm1(mde)))
        for stage in sorted({k[2] for k in meds}):
            vs = [meds[i, arm, stage] for i in images]
            raw = [v for (i,a,s), values in groups.items() if a==arm and s==stage for v in values]
            out['stages'][f'{arm}|{stage}'] = dict(image_medians_ms=vs,
                raw_p50_p90_max_ms=[float(np.median(raw)),float(np.quantile(raw,.9)),float(max(raw))])
    for stage in ('forward','nms_tv','nms_pure','mask_False','mask_True','total_tv_False','total_tv_True'):
        if not all((i,'attacked',stage) in meds and (i,'clean',stage) in meds for i in images): continue
        out['contrasts'][stage] = paired(stage, stage, arm_a='attacked', arm_b='clean')
        out['contrasts'][stage]['paired_delta_ms'] = [meds[i,'attacked',stage]-meds[i,'clean',stage] for i in images]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    a = ap.parse_args()
    result = analyze(a.run)
    with (a.run/'summary.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:result[k] for k in ('smoke','complete','quality_pass','primary_pure_over_tv','mde')},indent=2))
