"""Exploratory audit AFTER observing a shifted registered median interval.

Keep registered outputs unchanged. Reproduce every registered median interval
and compare image-only bootstrap conditional on the five observed starts.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze import load, TASKS, VARIANTS


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder', type=Path); ap.add_argument('--cohort', type=Path, required=True)
    a = ap.parse_args(); cohort = json.loads(a.cohort.read_text())
    endpoints = [(0,4),(1,4),(2,0),(3,0),(2,4),(3,4)]
    result = dict(status='exploratory, after primary outcomes were inspected', repetitions=20000, seed=20261106,
        motivation='Registered26-many median287.8 lies below its hierarchical percentile CI288.4–292.0 at n1500.',
        interpretation='Image-only intervals condition on the five observed starts; they do not replace registered two-level uncertainty.', cohorts={})
    for n in [1000,1500]:
        clean, attacked, _ = load(a.folder, cohort, [1000] if n==1000 else [1000,1500])
        summary = json.loads((a.folder/f'summary{n}.json').read_text())
        values = np.stack([attacked[:,:,t,v] for t,v in endpoints], axis=2)
        means = values.mean(1); rng = np.random.default_rng(20261106)
        hierarchical = []; image_only = []
        for _ in range(200):
            image = rng.integers(n, size=(100,n,1)); start = rng.integers(5, size=(100,n,5))
            hierarchical.append(np.median(values[image,start].mean(2), axis=1))
            image_only.append(np.median(means[image[:,:,0]], axis=1))
        h = np.concatenate(hierarchical); fixed = np.concatenate(image_only); rows = {}
        for e,(t,v) in enumerate(endpoints):
            primary = summary['tasks'][TASKS[t]][VARIANTS[v]]
            ci = np.quantile(h[:,e],[.025,.975]).tolist()
            assert ci == primary['attacked_median_image_mean_ci95'], 'Registered bootstrap not reproduced'
            point = primary['attacked_median_image_mean']
            rows[TASKS[t]+':'+VARIANTS[v]] = dict(point=point, registered_hierarchical_ci95=ci,
                hierarchical_bootstrap_median=float(np.median(h[:,e])), hierarchical_bootstrap_mean=float(h[:,e].mean()),
                hierarchical_median_shift=float(np.median(h[:,e])-point),
                registered_interval_contains_point=bool(ci[0]<=point<=ci[1]),
                conditional_image_only_ci95=np.quantile(fixed[:,e],[.025,.975]).tolist())
        result['cohorts'][str(n)] = rows
    with (a.folder/'bootstrap_sensitivity.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
