"""R2 — lát cắt bản đồ điều kiện từ OLD128 backend_qos (THĂM DÒ, chỉ đọc). Kế hoạch: prereg/ANALYSIS_R2_condition_slices.md."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import cohorts, manifest, stats  # noqa: E402

SEED = 20261021
N_BOOT_RHO = 2000
BQ = cohorts.COHORTS['OLD128'] / 'backend_qos'
CONFIG = dict(analysis='R2', label='thăm dò, dữ liệu lịch sử OLD128, không ghim core', bootstrap_seed=SEED,
              n_boot=stats.N_BOOT, n_boot_rho=N_BOOT_RHO, source=str(BQ))


def rho_ci(x, y, clusters):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return stats.cluster_bootstrap(np.zeros(len(x)), clusters, SEED, n_boot=N_BOOT_RHO,
                                   fn=lambda idx: stats.spearman(x[idx], y[idx])).tolist()


def matrix(d, ids, seeds):
    return np.array([[d[(i, s)] for s in seeds] for i in ids], float)


def backend_part():
    recs = json.loads((BQ / 'backend.json').read_text(encoding='utf-8'))['records']
    cells = defaultdict(dict)  # (backend, tau, cap) -> (cond, img, seed) -> (N, t)
    for r in recs:
        N = {s['nms_input_candidates'] for s in r['samples']}
        assert len(N) == 1
        t = float(np.median([s['nms_ms'] for s in r['samples']]))
        cells[(r['backend'], r['threshold'], r['pre_nms_cap'])][(r['condition'], r['image_id'], r['seed'])] = (N.pop(), t)
    out = {}
    for key, d in sorted(cells.items(), key=lambda kv: str(kv[0])):
        ids = sorted({k[1] for k in d})
        seeds = sorted({k[2] for k in d})
        keys = list(d)
        N = np.array([d[k][0] for k in keys], float)
        t = np.array([d[k][1] for k in keys], float)
        cl = np.array([k[1] for k in keys])
        Nc = {(i, s): d[('clean', i, s)][0] for i in ids for s in seeds}
        Ns = {(i, s): d[('sg_ga', i, s)][0] for i in ids for s in seeds}
        tc = {(i, s): d[('clean', i, s)][1] for i in ids for s in seeds}
        ts = {(i, s): d[('sg_ga', i, s)][1] for i in ids for s in seeds}
        dN = matrix(Ns, ids, seeds) - matrix(Nc, ids, seeds)
        dt = matrix(ts, ids, seeds) - matrix(tc, ids, seeds)
        ratio = matrix(ts, ids, seeds) / matrix(tc, ids, seeds)
        img = np.repeat(np.array(ids), len(seeds))
        lo, hi = stats.two_level_bootstrap(ratio, SEED)
        out['|'.join(map(str, key))] = dict(
            backend=key[0], tau=key[1], cap=key[2], n_images=len(ids), n_seeds=len(seeds),
            N_range=[int(N.min()), int(N.max())], median_nms_ms=float(np.median(t)),
            rho_N_t=stats.spearman(N, t), rho_N_t_ci95_cluster_image=rho_ci(N, t, cl),
            within_rho_dN_dt=stats.spearman(dN.ravel(), dt.ravel()),
            within_rho_ci95_cluster_image=rho_ci(dN.ravel(), dt.ravel(), img),
            mean_dN=float(dN.mean()), frac_dN_positive=float((dN > 0).mean()),
            nms_ratio_sg_over_clean=dict(mean=float(ratio.mean()), ci95_two_level=[float(lo), float(hi)]))
    for tau, cap in [(.01, None), (.01, 100), (.25, None)]:
        a, b = out.get(f'custom|{tau}|{cap}'), out.get(f'torchvision|{tau}|{cap}')
        if a and b:
            out[f'custom_over_torchvision|{tau}|{cap}'] = dict(median_ratio=a['median_nms_ms'] / b['median_nms_ms'])
    return out


def timing_part():
    res = {}
    for t in (1, 2, 4):
        rows = []
        for s in range(10):
            f = BQ / f'timing_seed{s}_threads{t}.json'
            for r in json.loads(f.read_text(encoding='utf-8'))['records']:
                rows.append((s, r['condition'], r['image_id'], r['active_candidates'], r['nms_ms'], r['filter_ms'],
                             r['total_prepared_tensor_ms'], r['forward_ms']))
        a = np.array([x[3:] for x in rows], float)
        cond = np.array([x[1] for x in rows])
        img = np.array([x[2] for x in rows])
        N, nms, filt, total, fwd = a.T
        share = (nms + filt) / total
        cs = np.isin(cond, ['clean', 'sg_ga'])
        med = defaultdict(list)
        for (s, c, i, *_), tot in zip(rows, total):
            med[(c, i, s)].append(tot)
        ids, seeds = sorted(set(img)), list(range(10))
        rc = matrix({(i, s): np.median(med[('clean', i, s)]) for i in ids for s in seeds}, ids, seeds)
        rs = matrix({(i, s): np.median(med[('sg_ga', i, s)]) for i in ids for s in seeds}, ids, seeds)
        lo, hi = stats.two_level_bootstrap(rs / rc, SEED)
        res[f'threads{t}'] = dict(
            n_records=len(rows), postprocess_share_of_total=dict(median=float(np.median(share)), p95=float(np.quantile(share, .95)),
                                                                 max=float(share.max())),
            rho_N_nms=stats.spearman(N[cs], nms[cs]), rho_N_nms_ci95_cluster_image=rho_ci(N[cs], nms[cs], img[cs]),
            rho_N_total=stats.spearman(N[cs], total[cs]), rho_N_total_ci95_cluster_image=rho_ci(N[cs], total[cs], img[cs]),
            total_ratio_sg_over_clean=dict(mean=float((rs / rc).mean()), ci95_two_level=[float(lo), float(hi)]),
            late_over_300ms=int((total > 300).sum()))
    return res


def main():
    run_dir = SIM / 'results' / 'r2' / time.strftime('%Y%m%dT%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest.write(run_dir / 'manifest.json', manifest.build(CONFIG))
    out = dict(analysis='R2', label=CONFIG['label'], backend_cells=backend_part(), timing=timing_part(),
               condition_map_coverage=dict(
                   present=['backend {custom (agnostic), torchvision nms (agnostic)}', 'tau {0.01, 0.25}', 'cap {None, 100}',
                            'intra-op threads {1,2,4} (KHÔNG ghim core)', 'period {100, 300} ms (stream_qos)'],
                   missing=['class-aware backend (batched_nms, Ultralytics native)', 'ghim core thật (E5 mới có)',
                            'N vượt ~270 (cần tiêm tổng hợp: E1)', 'hình học hộp có kiểm soát (E1)',
                            'chu kỳ quanh knee 120-250 ms', 'thiết bị ARM/edge']))
    (run_dir / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
