"""R3 — proxy N vs postprocess vs total trên biến thể tham chiếu (THĂM DÒ, chỉ đọc). Kế hoạch: prereg/ANALYSIS_R3_reference_proxy.md."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import cohorts, manifest, stats  # noqa: E402

SEED = 20261021
REF = cohorts.COHORTS['OLD128'] / 'reference'
METHODS = ['overload', 'phantom_sponges']
CONFIG = dict(analysis='R3', label='thăm dò, lịch sử, không ghim core; YOLOv5s 640 UAP toàn ảnh', methods=METHODS,
              bootstrap_seed=SEED, n_boot=stats.N_BOOT, source=str(REF))


def load(method):
    per = {}
    for s in range(10):
        d = json.loads((REF / f'{method}_seed{s}.json').read_text(encoding='utf-8'))
        for r in d['records']:
            N = {x['active_candidates'] for x in r['samples']}
            assert len(N) == 1
            per[(r['condition'], r['image_id'], s)] = dict(
                N=N.pop(), pp=float(np.median([x['postprocess_ms'] for x in r['samples']])),
                tot=float(np.median([x['total_ms'] for x in r['samples']])),
                fwd=float(np.median([x['forward_ms'] for x in r['samples']])))
    ids = sorted({k[1] for k in per})
    M = lambda c, f: np.array([[per[(c, i, s)][f] for s in range(10)] for i in ids], float)  # noqa: E731
    return {c: {f: M(c, f) for f in ('N', 'pp', 'tot', 'fwd')} for c in ('clean', 'patched')}, ids


def summ(m):
    lo, hi = stats.two_level_bootstrap(m, SEED)
    return dict(mean=float(m.mean()), ci95_two_level=[float(lo), float(hi)])


def main():
    run_dir = SIM / 'results' / 'r3' / time.strftime('%Y%m%dT%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest.write(run_dir / 'manifest.json', manifest.build(CONFIG))
    out = dict(analysis='R3', label=CONFIG['label'], methods={})
    for m in METHODS:
        d, ids = load(m)
        c, p = d['clean'], d['patched']
        out['methods'][m] = dict(
            n_images=len(ids), n_seeds=10,
            median_N=dict(clean=float(np.median(c['N'])), patched=float(np.median(p['N']))),
            median_pp_ms=dict(clean=float(np.median(c['pp'])), patched=float(np.median(p['pp']))),
            median_total_ms=dict(clean=float(np.median(c['tot'])), patched=float(np.median(p['tot']))),
            pp_share_of_total_median=dict(clean=float(np.median(c['pp'] / c['tot'])), patched=float(np.median(p['pp'] / p['tot']))),
            ratio_N=summ(p['N'] / np.maximum(c['N'], 1)), ratio_pp=summ(p['pp'] / c['pp']),
            ratio_total=summ(p['tot'] / c['tot']), ratio_forward=summ(p['fwd'] / c['fwd']),
            delta_pp_ms=summ(p['pp'] - c['pp']), delta_total_ms=summ(p['tot'] - c['tot']),
            clean_total_cv_between_seeds_median=float(np.median(c['tot'].std(axis=1, ddof=1) / c['tot'].mean(axis=1))),
            rho_N_pp_pooled=stats.spearman(np.r_[c['N'].ravel(), p['N'].ravel()], np.r_[c['pp'].ravel(), p['pp'].ravel()]),
            rho_N_total_pooled=stats.spearman(np.r_[c['N'].ravel(), p['N'].ravel()], np.r_[c['tot'].ravel(), p['tot'].ravel()]))
    (run_dir / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
