"""R1 — CI hai tầng (ảnh × seed) cho SG-GA − đối chứng, OLD128 và X1-500. POST-HOC, chỉ đọc.
Kế hoạch: prereg/ANALYSIS_R1_twolevel.md. Fail-closed nếu trung bình không tái tạo số theo seed đã xác minh."""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import cohorts, manifest, stats  # noqa: E402

SEED = 20261021
SEEDS = list(range(10))
REFS = ['ga_center', 'ga_random_location', 'random_search']
CASES = ['clean', 'sg_ga'] + REFS
# Số theo seed đã xác minh (CLAUDE.md; REVIEW/03 §6) — dùng để fail-closed.
VERIFIED = {('OLD128', 'ga_center'): -0.005132356025296, ('X1-500', 'ga_center'): -0.029335098118660065}
CONFIG = dict(analysis='R1', label='POST-HOC, chỉ đọc', tau=.01, cap=None, q=2500, seeds=SEEDS, refs=REFS,
              bootstrap_seed=SEED, n_boot=stats.N_BOOT, method='pigeonhole (rows=images, cols=seeds), Owen 2007')


def load_counts(cohort):
    counts, ids = {}, None
    for case in CASES:
        q = 0 if case == 'clean' else 2500
        per_seed = []
        for s in SEEDS:
            rec = cohorts.records(cohorts.read_case(cohort, case, s, q))
            per_seed.append({i: r['active_candidates'] for i, r in rec.items()})
            del rec
            gc.collect()
        common = set.intersection(*(set(p) for p in per_seed))
        ids = common if ids is None else ids & common
        counts[case] = per_seed
    ids = sorted(ids)
    return {c: np.array([[counts[c][s][i] for s in SEEDS] for i in ids], float) for c in CASES}, ids


def ci_rows(m, rng):
    ni = m.shape[0]
    return np.quantile([m[rng.integers(0, ni, ni)].mean() for _ in range(stats.N_BOOT)], [.025, .975])


def ci_cols(m, rng):
    ns = m.shape[1]
    return np.quantile([m[:, rng.integers(0, ns, ns)].mean() for _ in range(stats.N_BOOT)], [.025, .975])


def main():
    run_dir = SIM / 'results' / 'r1' / time.strftime('%Y%m%dT%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest.write(run_dir / 'manifest.json', manifest.build(CONFIG))
    out = dict(analysis='R1', label=CONFIG['label'], cohorts={})
    for cohort in ['OLD128', 'X1-500']:
        N, ids = load_counts(cohort)
        clean = np.maximum(N['clean'], 1)
        r = {c: N[c] / clean for c in CASES if c != 'clean'}
        res = dict(n_images=len(ids), n_seeds=len(SEEDS), comparisons={})
        for ref in REFS:
            D = r['sg_ga'] - r[ref]
            per_seed = D.mean(axis=0)
            mean = float(D.mean())
            if (cohort, ref) in VERIFIED:
                assert abs(mean - VERIFIED[(cohort, ref)]) < 1e-12, (cohort, ref, mean)
            seed_ci = ci_cols(D, np.random.default_rng(SEED))
            img_ci = ci_rows(D, np.random.default_rng(SEED))
            two_ci = stats.two_level_bootstrap(D, SEED)
            per_image = D.mean(axis=1)
            res['comparisons'][f'sg_ga_minus_{ref}'] = dict(
                mean=mean, per_seed_means=per_seed.tolist(),
                ci95_seed_only=[float(x) for x in seed_ci], ci95_image_only=[float(x) for x in img_ci],
                ci95_two_level=[float(x) for x in two_ci],
                width_ratio_two_level_over_seed_only=float((two_ci[1] - two_ci[0]) / (seed_ci[1] - seed_ci[0])),
                two_level_contains_zero=bool(two_ci[0] <= 0 <= two_ci[1]),
                frac_images_mean_diff_negative=float((per_image < 0).mean()),
                frac_images_mean_diff_zero=float((per_image == 0).mean()),
                verified_against=VERIFIED.get((cohort, ref)))
        out['cohorts'][cohort] = res
        del N, r
        gc.collect()
    (run_dir / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
