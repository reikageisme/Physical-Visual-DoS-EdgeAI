"""Nested-cohort analysis; resample images then starts within sampled images.

Per-start Wilson n=images. ANY-start reach has five times the query budget.
No latency inference and no independent-replication interpretation of prefixes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = [0, 12345, 23456, 34567, 45678]
TASKS = ['y26_many', 'y26_native', 'v8', 'v11']
VARIANTS = ['baseline', 'maxdet50', 'prenms300', 'prenms1000', 'native26']


def wilson(values):
    values = np.asarray(values, dtype=bool)
    hits = int(values.sum()); n = values.size; p = hits / n
    z = 1.959963984540054; d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return dict(hits=hits, n_images=n, fraction=p, wilson_ci95=[max(0., center-half), min(1., center+half)])


def descriptive(clean, attacked):
    n, starts = attacked.shape
    out = dict(n_images=n, n_starts=starts, n_cases=n * starts,
               clean_median_image_mean=float(np.median(clean.mean(1))),
               attacked_median_image_mean=float(np.median(attacked.mean(1))),
               mean_paired_gain=float((attacked-clean).mean()),
               median_image_mean_gain=float(np.median((attacked-clean).mean(1))),
               mean_start_reach50=float((attacked >= 50).mean()),
               exploratory_clean_mean_start_reach50=float((clean >= 50).mean()),
               exploratory_new_crossing50_fraction=float(((attacked >= 50) & (clean < 50)).mean()),
               per_start={}, any_start={}, mean_start_reach={})
    for j, seed in enumerate(SEEDS):
        a = attacked[:, j]; c = clean[:, j]
        out['per_start'][str(seed)] = dict(clean_median=float(np.median(c)), attacked_median=float(np.median(a)),
            median_gain=float(np.median(a-c)), mean_gain=float((a-c).mean()),
            attacked_quantiles=np.quantile(a, [0, .05, .25, .5, .75, .95, 1]).tolist(),
            exploratory_clean_reach50=wilson(c >= 50),
            exploratory_new_crossing50=wilson((a >= 50) & (c < 50)),
            reach={str(k): wilson(a >= k) for k in [50, 100, 300]})
    for k in [50, 100, 300]:
        out['any_start'][str(k)] = wilson((attacked >= k).any(1))
        out['mean_start_reach'][str(k)] = float((attacked >= k).mean())
    out['exploratory_any_start_new_crossing50'] = wilson(((attacked >= 50) & (clean < 50)).any(1))
    return out


def hierarchical(clean, attacked, repetitions=20000):
    """Arrays [image,start,endpoint]; same draws preserve endpoint pairing."""
    n, starts, endpoints = attacked.shape
    rng = np.random.default_rng(20261106)
    values = {k: [] for k in ['attacked_median_image_mean', 'mean_paired_gain', 'mean_start_reach50']}
    for offset in range(0, repetitions, 100):
        b = min(100, repetitions-offset)
        image = rng.integers(n, size=(b, n, 1))
        start = rng.integers(starts, size=(b, n, starts))
        a = attacked[image, start]; c = clean[image, start]
        values['attacked_median_image_mean'].append(np.median(a.mean(2), axis=1))
        values['mean_paired_gain'].append((a-c).mean((1, 2)))
        values['mean_start_reach50'].append((a >= 50).mean((1, 2)))
    return [{k + '_ci95': np.quantile(np.concatenate(v)[:, e], [.025, .975]).tolist()
             for k, v in values.items()} for e in range(endpoints)]


def load(folder, cohort, stages):
    n = 1000 if stages == [1000] else 1500
    names = [r['file_name'] for r in cohort['images'][:n]]
    lookup = {name: j for j, name in enumerate(names)}
    clean = np.full((n, 5, 4, 5), -1, dtype=np.int16)
    attacked = clean.copy(); seen = set(); raw_hashes = {}
    for stage in stages:
        path = folder / f'stage{stage}_records.jsonl'
        raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = json.loads((folder / f'stage{stage}_receipt.json').read_text())
        assert raw_hashes[path.name] == receipt['raw_sha256']
        for line in path.read_text().splitlines():
            r = json.loads(line); t = TASKS.index(r['task']); s = SEEDS.index(r['seed'])
            for b, name in enumerate(r['images']):
                i = lookup[name]; key = (i, s, t); assert key not in seen; seen.add(key)
                for arm, dest in [('clean', clean), ('attacked', attacked)]:
                    for v in range(4):
                        assert r[f'{arm}_variants'][v]['variant'] == VARIANTS[v]
                        dest[i, s, t, v] = r[f'{arm}_variants'][v]['k'][b]
                    dest[i, s, t, 4] = r[f'{arm}_native_k'][b]
    assert len(seen) == n * 20 and (clean >= 0).all() and (attacked >= 0).all()
    assert (clean == clean[:, :1]).all(), 'clean counts vary across starts'
    assert (clean[:, :, :, 4] == clean[:, :, :1, 4]).all(), 'native clean counts vary across tasks'
    return clean, attacked, raw_hashes


def analyze(folder, cohort_path, stages, repetitions=20000):
    cohort = json.loads(cohort_path.read_text())
    clean, attacked, hashes = load(folder, cohort, stages)
    n = len(clean)
    out = dict(n_images=n, n_cases=n*20, seeds=SEEDS, complete=True, raw_sha256=hashes,
               cohort_sha256=hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
               bootstrap=dict(repetitions=repetitions, seed=20261106,
                              method='resample images then five starts within each sampled image; paired endpoint draws'),
               nested_not_independent=True, tasks={})
    primary = [(0, 4), (1, 4), (2, 0), (3, 0), (2, 4), (3, 4)]
    cis = hierarchical(np.stack([clean[:, :, t, v] for t, v in primary], axis=2),
                       np.stack([attacked[:, :, t, v] for t, v in primary], axis=2), repetitions)
    for t, task in enumerate(TASKS):
        out['tasks'][task] = {v: descriptive(clean[:, :, t, j], attacked[:, :, t, j]) for j, v in enumerate(VARIANTS)}
    for (t, v), ci in zip(primary, cis):
        out['tasks'][TASKS[t]][VARIANTS[v]].update(ci)
    # Exploratory confidence-rank diagnostics, linked to the same saved tensors.
    name_index = {r['file_name']: i for i, r in enumerate(cohort['images'][:n])}
    confidence = {arm: np.full((n, 5, 4, 4), np.nan) for arm in ['clean', 'attacked']}
    for stage in stages:
        for line in (folder/f'stage{stage}_records.jsonl').read_text().splitlines():
            r = json.loads(line); t = TASKS.index(r['task']); s = SEEDS.index(r['seed'])
            for b, name in enumerate(r['images']):
                for arm in confidence:
                    confidence[arm][name_index[name], s, t] = r[f'{arm}_native_confidence_ranks_1_50_100_300'][b]
    for t, task in enumerate(TASKS):
        out['tasks'][task]['exploratory_native_confidence'] = {
            arm+'_median_image_mean_ranks_1_50_100_300': np.median(values[:, :, t].mean(1), axis=0).tolist()
            for arm, values in confidence.items()}
    assert all(np.isfinite(v).all() for v in confidence.values())
    # Paired descriptive objective difference, with identical image/start keys.
    difference = attacked[:, :, 0, 4] - attacked[:, :, 1, 4]
    out['paired_target26_many_minus_native'] = dict(mean_native_k_difference=float(difference.mean()),
        median_image_mean_native_k_difference=float(np.median(difference.mean(1))),
        image_start_wins=int((difference > 0).sum()), ties=int((difference == 0).sum()), losses=int((difference < 0).sum()),
        interpretation='paired objective treatment, not architecture causality')
    path = folder / f'summary{n}.json'
    with path.open('x') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(dict(summary=str(path), images=n, cases=n*20, primary={TASKS[t]+':'+VARIANTS[v]:
        out['tasks'][TASKS[t]][VARIANTS[v]]['mean_start_reach50'] for t,v in primary})), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder', type=Path); ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--stages', type=int, nargs='+', choices=[1000, 1500], required=True)
    a = ap.parse_args(); assert a.stages in [[1000], [1000, 1500]]
    analyze(a.folder, a.cohort, a.stages)
