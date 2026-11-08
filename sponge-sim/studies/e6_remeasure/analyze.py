"""Phân tích E6 theo PREREG_E6.md.   python analyze.py RUN_DIR AA_SUMMARY E1_SUMMARY [PROFILE]

Đơn vị suy luận: cặp (ảnh, seed patch); CI = bootstrap hai tầng (rút lại ảnh và seed), 20.000 lần (Spearman: 2.000), seed 20261021.
Mỗi ô (ảnh, nhánh, seed, ngưỡng, backend) được tóm bằng trung vị các lần lặp. Nhánh clean không có seed: dùng chung cho mọi cột seed.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import stats  # noqa: E402

SEED = 20261021
N_BOOT_RHO = 2000
BACKENDS = ['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native']
GEOMS = ['disjoint', 'dense_cluster', 'mixed']


def two_level_rho(x, y, seed=SEED, n_boot=N_BOOT_RHO):
    rng = np.random.default_rng(seed)
    ni, ns = x.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        r, c = rng.integers(0, ni, ni), rng.integers(0, ns, ns)
        out[b] = stats.spearman(x[np.ix_(r, c)].ravel(), y[np.ix_(r, c)].ravel())
    return np.nanquantile(out, [.025, .975]).tolist()


def curve(curves, profile, anchors, backend, geom, nc=80, cap=None):
    pts = {}
    for k, v in curves.items():
        p, A, be, g, c, cp, n = k.split('|')
        if p == profile and int(A) == anchors and be == backend and g == geom and int(c) == nc and cp == str(cap):
            pts[int(n)] = v['median_ms']
    return np.array(sorted(pts)), np.array([pts[n] for n in sorted(pts)])


def predict(ns_grid, ms_grid, n):
    """Nội suy log-log theo N>0; dưới điểm N>0 nhỏ nhất: nội suy tuyến tính với ô N=0 (nếu có)."""
    pos = ns_grid > 0
    if n <= 0:
        return float(ms_grid[0]) if (ns_grid == 0).any() else float('nan')
    lx, ly = np.log(ns_grid[pos]), np.log(ms_grid[pos])
    if n < ns_grid[pos].min():
        t0 = ms_grid[ns_grid == 0][0] if (ns_grid == 0).any() else ms_grid[pos][0]
        return float(t0 + (ms_grid[pos][0] - t0) * n / ns_grid[pos].min())
    return float(np.exp(np.interp(np.log(n), lx, ly)))


def main():
    run_dir, aa_path, e1_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    profile = sys.argv[4] if len(sys.argv) > 4 else 'P1_cpu2'
    aa = json.loads(aa_path.read_text(encoding='utf-8'))['profiles'][profile]['stages']
    curves = json.loads(e1_path.read_text(encoding='utf-8'))['curves']
    micro, pipe, nmap, refs = defaultdict(list), defaultdict(list), {}, []
    pin = None
    for line in next(run_dir.glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        k = r['kind']
        if k == 'pin':
            pin = r['verify']
        elif k == 'reference':
            refs.append(r)
        elif k == 'micro':
            micro[(r['image_id'], r['arm'], r['seed'], r['tau'], r['backend'])].append(r['nms_ns'] / 1e6)
            nmap[(r['image_id'], r['arm'], r['seed'], r['tau'])] = r['nms_input']
        elif k == 'pipeline':
            pipe[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['total_ns'] / 1e6)
    med = {k: float(np.median(v)) for k, v in micro.items()}
    pmed = {k: float(np.median(v)) for k, v in pipe.items()}
    a1 = len(sys.argv) > 5 and sys.argv[5] == '--a1'  # PREREG_E6_addendum_clean640: C2/C4 so với clean640
    arms = sorted({k[1] for k in med if k[1] not in ('clean', 'clean640')})
    out = dict(experiment='E6', run_dir=str(run_dir), profile=profile, bootstrap_seed=SEED, n_boot=stats.N_BOOT,
               pin=dict(effective_cores=pin['effective_cores'], passed=pin['passed']),
               drift=dict(points=len(refs), flagged=sum(r['flagged'] for r in refs), max_abs=max(abs(r['rel_change']) for r in refs)),
               mde_aa_pct={s: aa[s]['mde_ratio_pct'] for s in ('forward', 'total', 'nms_torch_pure', 'nms_tv_nms')}, arms={})
    for arm in arms:
        ref = 'clean640' if (a1 and arm in ('C2', 'C4')) else 'clean'
        ids = sorted({k[0] for k in med if k[1] == arm})
        seeds = sorted({k[2] for k in med if k[1] == arm})
        a = {}
        for tau in (0.01, 0.25):
            nc = np.array([[nmap[(i, ref, None, tau)] for _ in seeds] for i in ids], float)
            npat = np.array([[nmap[(i, arm, s, tau)] for s in seeds] for i in ids], float)
            ratio = npat / np.maximum(nc, 1)
            a[f'N_tau{tau}'] = dict(clean_median=float(np.median(nc[:, 0])), patched_median=float(np.median(npat)),
                                    ratio_mean=float(ratio.mean()), ratio_ci95=stats.two_level_bootstrap(ratio, SEED).tolist())
            for be in BACKENDS:
                tc = np.array([[med[(i, ref, None, tau, be)] for _ in seeds] for i in ids])
                tp = np.array([[med[(i, arm, s, tau, be)] for s in seeds] for i in ids])
                rt = tp / tc
                dN, dt = npat - nc, tp - tc
                a[f'nms_tau{tau}_{be}'] = dict(
                    ratio_mean=float(rt.mean()), ratio_ci95=stats.two_level_bootstrap(rt, SEED).tolist(),
                    rho_dN_dt=stats.spearman(dN.ravel(), dt.ravel()), rho_ci95=two_level_rho(dN, dt),
                    rho_between_clean=stats.spearman(nc[:, 0], tc[:, 0]))
        for be in ('torch_pure', 'tv_nms'):
            tc = np.array([[pmed[(i, ref, None, be)] for _ in seeds] for i in ids])
            tp = np.array([[pmed[(i, arm, s, be)] for s in seeds] for i in ids])
            rt = tp / tc
            lo, hi = stats.two_level_bootstrap(rt, SEED)
            nms_c = np.array([[med[(i, ref, None, 0.01, be)] for _ in seeds] for i in ids])
            nms_p = np.array([[med[(i, arm, s, 0.01, be)] for s in seeds] for i in ids])
            share = (nms_p - nms_c) / tc
            a[f'total_{be}'] = dict(ratio_mean=float(rt.mean()), ratio_ci95=[float(lo), float(hi)],
                                    ci_within_aa_mde=bool(lo >= 1 - out['mde_aa_pct']['total'] / 100 and hi <= 1 + out['mde_aa_pct']['total'] / 100),
                                    nms_share_clean_median=float(np.median(nms_c / tc)),
                                    nms_delta_share_mean=float(share.mean()), nms_delta_share_ci95=stats.two_level_bootstrap(share, SEED).tolist())
        out['arms'][arm] = a

    # N có dự đoán được độ trễ NMS không? Dự đoán chỉ-từ-N bằng đường cong E1 (tiêm tổng hợp) so với đo thật
    pred = {}
    for be in BACKENDS:
        errs, inside, total_n = [], 0, 0
        per_arm = defaultdict(list)
        for (i, arm, s, tau, b), t in med.items():
            if b != be or tau != 0.01:
                continue
            canvas_anchor = 8400 if arm in ('C2', 'C4', 'clean640') else 2100
            n = nmap[(i, arm, s, tau)]
            if n <= 0:
                continue
            ps = {}
            for g in GEOMS:
                ng, mg = curve(curves, profile, canvas_anchor, be, g)
                if len(ng) == 0:
                    ps = None
                    break
                ps[g] = predict(ng, mg, n)
            if ps is None:
                continue
            lo_e, hi_e = min(ps.values()), max(ps.values())
            e = abs(np.log(t / ps['mixed']))
            errs.append(e); per_arm[arm].append(e)
            inside += int(lo_e <= t <= hi_e); total_n += 1
        if errs:
            pred[be] = dict(n_points=total_n, median_abs_log_error_vs_mixed=float(np.median(errs)),
                            p90_abs_log_error=float(np.quantile(errs, .9)), fraction_inside_envelope=inside / total_n,
                            per_arm_median_abs_log_error={k: float(np.median(v)) for k, v in per_arm.items()})
    out['n_only_prediction_from_e1'] = pred
    (run_dir / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('pin', 'drift', 'mde_aa_pct')}, ensure_ascii=False))
    print({be: (round(v['median_abs_log_error_vs_mixed'], 3), round(v['fraction_inside_envelope'], 3)) for be, v in pred.items()})


if __name__ == '__main__':
    main()
