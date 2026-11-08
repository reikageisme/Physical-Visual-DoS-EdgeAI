"""Phân tích E2b theo PREREG_E2b.md.  Dùng: e2b_analyze.py RUN_DIR AA_SUMMARY_JSON

Đơn vị: (ảnh, seed patch). Bootstrap hai tầng (ảnh × seed), 20.000 lần, seed 20261021 (Spearman: 2.000 lần, chi phí).
Mỗi ô (ảnh, điều kiện, seed, backend, cap) được tóm bằng trung vị các lần lặp.
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


def two_level_rho(dN, dt, seed=SEED, n_boot=N_BOOT_RHO):
    """dN, dt: [ảnh, seed]. Rút lại ảnh và seed, Spearman trên các cặp đã chọn."""
    rng = np.random.default_rng(seed)
    ni, ns = dN.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        r, c = rng.integers(0, ni, ni), rng.integers(0, ns, ns)
        out[b] = stats.spearman(dN[np.ix_(r, c)].ravel(), dt[np.ix_(r, c)].ravel())
    return np.nanquantile(out, [.025, .975]).tolist()


def group_kfold_mae(rows, k=5, seed=SEED):
    """rows: (image_id, backend, N, t_ms). So mô hình log t ~ log(N+1) chung (bỏ qua backend) với riêng từng backend.
    Gập theo ảnh (không rò ảnh giữa train/test). Trả MAE log (xấp xỉ sai số tương đối)."""
    ids = np.array(sorted({r[0] for r in rows}))
    np.random.default_rng(seed).shuffle(ids)
    fold = {i: f for f, chunk in enumerate(np.array_split(ids, k)) for i in chunk}
    err = defaultdict(list)
    for f in range(k):
        train = [r for r in rows if fold[r[0]] != f]
        test = [r for r in rows if fold[r[0]] == f]
        def fit(rs):
            x = np.log1p([r[2] for r in rs]); y = np.log([r[3] for r in rs])
            return np.polyfit(x, y, 1)
        pooled = fit(train)
        per = {b: fit([r for r in train if r[1] == b]) for b in BACKENDS}
        for r in test:
            x, y = np.log1p(r[2]), np.log(r[3])
            err['N_only'].append(abs(np.polyval(pooled, x) - y))
            err['N_plus_backend'].append(abs(np.polyval(per[r[1]], x) - y))
    return {m: dict(mae_log=float(np.mean(e)), median_abs_log=float(np.median(e))) for m, e in err.items()}


def main():
    run_dir, aa_path = Path(sys.argv[1]), Path(sys.argv[2])
    aa = json.loads(aa_path.read_text(encoding='utf-8'))['profiles']['P1_cpu2']['stages']
    micro, pipe = defaultdict(list), defaultdict(list)
    pin = None
    for line in next(run_dir.glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['kind'] == 'pin':
            pin = r['verify']
        elif r['kind'] == 'micro':
            micro[(r['image_id'], r['cond'], r['seed'], r['backend'], r['cap'])].append((r['nms_ns'] / 1e6, r['n_tau']))
        elif r['kind'] == 'pipeline':
            pipe[(r['image_id'], r['cond'], r['seed'], r['backend'])].append(r['total_ns'] / 1e6)
    med = {k: float(np.median([v[0] for v in vs])) for k, vs in micro.items()}
    N = {(k[0], k[1], k[2]): vs[0][1] for k, vs in micro.items()}
    pmed = {k: float(np.median(v)) for k, v in pipe.items()}
    ids = sorted({k[0] for k in N})
    seeds = sorted({k[2] for k in N if k[1] != 'clean'})
    out = dict(experiment='E2b', run_dir=str(run_dir), aa_source=str(aa_path), pin=dict(
        effective_cores=pin['effective_cores'], passed=pin['passed']), n_images=len(ids), n_seeds=len(seeds),
        bootstrap_seed=SEED, n_boot=stats.N_BOOT, n_boot_rho=N_BOOT_RHO,
        mde_aa_pct={s: aa[s]['mde_ratio_pct'] for s in ('nms_torch_pure', 'nms_tv_nms', 'total')})
    dN = np.array([[N[(i, 'sg_ga', s)] - N[(i, 'clean', None)] for s in seeds] for i in ids], float)
    out['delta_N'] = dict(mean=float(dN.mean()), ci95_two_level=stats.two_level_bootstrap(dN, SEED).tolist(),
                          frac_positive=float((dN > 0).mean()))
    res = {}
    for b in BACKENDS:
        for cap in (None, 100):
            t_c = np.array([[med[(i, 'clean', None, b, cap)] for _ in seeds] for i in ids])
            t_p = np.array([[med[(i, 'sg_ga', s, b, cap)] for s in seeds] for i in ids])
            dt = t_p - t_c
            ratio = t_p / t_c
            nc = np.array([N[(i, 'clean', None)] for i in ids], float)
            res[f'{b}|cap{cap}'] = dict(
                within_rho_dN_dt=stats.spearman(dN.ravel(), dt.ravel()), within_rho_ci95=two_level_rho(dN, dt),
                between_rho_clean=stats.spearman(nc, t_c[:, 0]),
                nms_ratio_mean=float(ratio.mean()), nms_ratio_ci95=stats.two_level_bootstrap(ratio, SEED).tolist(),
                median_clean_ms=float(np.median(t_c[:, 0])))
    out['nms'] = res
    # Headroom H2: phần thay đổi độ trễ NMS / toàn trình sạch; và tỷ số toàn trình patch/sạch
    out['pipeline'] = {}
    for b in ('torch_pure', 'tv_nms'):
        tot_c = np.array([[pmed[(i, 'clean', None, b)] for _ in seeds] for i in ids])
        tot_p = np.array([[pmed[(i, 'sg_ga', s, b)] for s in seeds] for i in ids])
        nms_c = np.array([[med[(i, 'clean', None, b, None)] for _ in seeds] for i in ids])
        nms_p = np.array([[med[(i, 'sg_ga', s, b, None)] for s in seeds] for i in ids])
        share = (nms_p - nms_c) / tot_c
        r = tot_p / tot_c
        out['pipeline'][b] = dict(
            total_ratio_mean=float(r.mean()), total_ratio_ci95=stats.two_level_bootstrap(r, SEED).tolist(),
            nms_share_clean_median=float(np.median(nms_c / tot_c)),
            attack_nms_delta_share_mean=float(share.mean()),
            attack_nms_delta_share_ci95=stats.two_level_bootstrap(share, SEED).tolist(),
            mde_total_pct=out['mde_aa_pct']['total'])
    rows = [(i, b, N[(i, c, s)], med[(i, c, s, b, None)]) for (i, c, s, b, cap) in med if cap is None]
    out['proxy_prediction_groupkfold'] = group_kfold_mae(rows)
    (run_dir / 'summary.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('pin', 'delta_N', 'pipeline', 'proxy_prediction_groupkfold')}, ensure_ascii=False)[:1500])


if __name__ == '__main__':
    main()
