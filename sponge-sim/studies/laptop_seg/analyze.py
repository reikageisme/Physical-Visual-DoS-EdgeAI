"""Offline smoke assessment only; exclusive-create output, preserve all raw."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


def analyze(run):
    rows = [json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    manifest = json.loads((run/'manifest.json').read_text())
    cfg = manifest['config']
    groups = defaultdict(list)
    repetitions = defaultdict(list)
    for r in rows:
        if r['kind'] == 'timing':
            groups[r['image'],r['stage'],r['label']].append(r['ns'])
            repetitions[r['image'],r['stage'],r['label']].append(r['rep'])
    images = sorted({k[0] for k in groups})
    complete = bool(rows) and rows[-1]['kind']=='done' and len(images)==cfg['n_images'] and all(
        sorted(repetitions[i,s,l])==list(range(cfg['repeats']))
        for i in images for s in ('forward','total') for l in ('A','B'))
    references = [r for r in rows if r['kind']=='reference']
    flags = sum(r['flagged'] for r in references)
    pin = [r['verify']['passed'] for r in rows if r['kind'] in ('pin_start','pin_end')]
    smoke = cfg.get('smoke', True)
    out = dict(smoke=smoke, complete=complete, flagged_references=flags, n_references=len(references),
               references=references, pin_checks=pin,
               feasibility_pass=complete and len(references)==cfg['n_images']+2
                   and flags<=cfg.get('max_reference_flags',0) and pin==[True,True],
               inference_allowed=False, raw_sha256=hashlib.sha256((run/'raw.jsonl').read_bytes()).hexdigest(), stages={})
    for stage in ('forward','total'):
        if not images or not all(groups[i,stage,l] for i in images for l in ('A','B')):
            continue
        a = np.array([np.median(groups[i,stage,'A']) for i in images])
        b = np.array([np.median(groups[i,stage,'B']) for i in images])
        d = np.log(a/b)
        mde = ((stats.t.ppf(.975,len(d)-1)+stats.t.ppf(.8,len(d)-1))*d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 else None
        out['stages'][stage] = dict(image_ids=images, median_A_ms=(a/1e6).tolist(), median_B_ms=(b/1e6).tolist(),
                                    aa_geometric_ratio=float(np.exp(d.mean())),
                                    aa_mde_fraction=None if mde is None else float(np.expm1(mde)))
    out['quality_pass'] = None if smoke else (out['feasibility_pass'] and all(
        out['stages'].get(s,{}).get('aa_mde_fraction') is not None
        and out['stages'][s]['aa_mde_fraction']<=cfg['max_mde_fraction'] for s in ('forward','total')))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run', type=Path)
    args = ap.parse_args()
    result = analyze(args.run)
    filename = 'smoke_summary.json' if result['smoke'] else 'controls_summary.json'
    with (args.run/filename).open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
