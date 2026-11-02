"""E2a — proxy N_tau trên dữ liệu lịch sử X1-500 (thăm dò, chỉ đọc). Xem prereg/PREREG_E2a.md."""
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
N_BOOT_SPEARMAN = 2000
ATTACKS = ['sg_ga', 'ga_center', 'ga_random_location', 'random_search']
SEEDS = range(10)
CONFIG = dict(experiment='E2a', cohort='X1-500', tau=.01, cap=None, attacks=ATTACKS, q=2500, seeds=list(SEEDS),
              bootstrap_seed=SEED, n_boot=stats.N_BOOT, n_boot_spearman=N_BOOT_SPEARMAN,
              label='thăm dò, dữ liệu lịch sử: không ghim core, backend custom, 1 mẫu NMS/record')


def load():
    """arr[case] = dict(N, nms, fwd, post) mỗi cái shape [ảnh, seed]; ids chung."""
    out, ids = {}, None
    for case in ['clean'] + ATTACKS:
        q = 0 if case == 'clean' else 2500
        per_seed = [cohorts.records(cohorts.read_case('X1-500', case, s, q)) for s in SEEDS]
        if ids is None:
            ids = sorted(per_seed[0])
        assert all(sorted(r) == ids for r in per_seed), case
        out[case] = {k: np.array([[per_seed[s][i][f] for s in SEEDS] for i in ids], float)
                     for k, f in [('N', 'active_candidates'), ('nms', 'nms_ms'), ('fwd', 'forward_ms'),
                                  ('post', 'postprocess_ms')]}
    return out, ids


def cluster_spearman_ci(x, y, clusters):
    return stats.cluster_bootstrap(np.zeros(len(x)), clusters, SEED, n_boot=N_BOOT_SPEARMAN,
                                   fn=lambda idx: stats.spearman(x[idx], y[idx])).tolist()


def main():
    run_dir = SIM / 'results' / 'e2a' / time.strftime('%Y%m%dT%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest.write(run_dir / 'manifest.json', manifest.build(CONFIG))
    d, ids = load()
    n_img = len(ids)
    img_idx = np.repeat(np.arange(n_img), len(SEEDS))  # cụm theo ảnh khi ravel [ảnh, seed]
    res = dict(experiment='E2a', label=CONFIG['label'], n_images=n_img, n_seeds=len(SEEDS))

    # 1. Nhiễu đo lịch sử trên clean (N giống hệt giữa seed)
    c = d['clean']
    assert (c['N'] == c['N'][:, :1]).all()
    pos = c['nms'] > 0
    cv = c['nms'].std(axis=1, ddof=1) / c['nms'].mean(axis=1)
    dev = np.abs(np.log(c['nms'] / np.median(c['nms'], axis=1, keepdims=True)))
    res['historical_noise_clean'] = dict(
        n_identical_N=True, all_positive=bool(pos.all()),
        median_within_image_cv=float(np.median(cv)), p90_within_image_cv=float(np.quantile(cv, .9)),
        median_abs_log_dev_from_image_median=float(np.median(dev)))

    # 2. Gộp
    N_all = np.concatenate([d[k]['N'].ravel() for k in d])
    t_all = np.concatenate([d[k]['nms'].ravel() for k in d])
    cl_all = np.concatenate([img_idx for _ in d])
    res['pooled_spearman'] = dict(rho=stats.spearman(N_all, t_all), ci95_cluster_image=cluster_spearman_ci(N_all, t_all, cl_all),
                                  n_records=len(N_all))

    # 3. Giữa ảnh (clean)
    Nc, tc = c['N'][:, 0], np.median(c['nms'], axis=1)
    res['between_image_clean'] = dict(rho=stats.spearman(Nc, tc), ci95=cluster_spearman_ci(Nc, tc, np.arange(n_img)))

    # 4. Trong ảnh: Δ so với clean cùng seed
    res['within_image'] = {}
    for k in ATTACKS:
        dN = (d[k]['N'] - c['N']).ravel()
        dt = (d[k]['nms'] - c['nms']).ravel()
        res['within_image'][k] = dict(
            rho_dN_dt=stats.spearman(dN, dt), ci95_cluster_image=cluster_spearman_ci(dN, dt, img_idx),
            mean_dN=float(dN.mean()), frac_dN_positive=float((dN > 0).mean()), frac_dN_zero=float((dN == 0).mean()))
    b = res['between_image_clean']['rho']
    res['simpson_check'] = {k: dict(between_sign=int(np.sign(b)), within_sign=int(np.sign(v['rho_dN_dt'])),
                                    sign_reversal=bool(np.sign(b) != np.sign(v['rho_dN_dt'])))
                            for k, v in res['within_image'].items()}

    # 5. Headroom
    res['headroom'] = {}
    for k in ['clean'] + ATTACKS:
        h = d[k]['nms'] / (d[k]['fwd'] + d[k]['post'])
        res['headroom'][k] = dict(median=float(np.median(h)), p95=float(np.quantile(h, .95)))
    res['attack_induced_share'] = {}
    for k in ATTACKS:
        g = (d[k]['nms'] - c['nms']) / (c['fwd'] + c['post'])          # [ảnh, seed]
        lo, hi = stats.two_level_bootstrap(g, SEED)
        per_seed = g.mean(axis=0)
        res['attack_induced_share'][k] = dict(mean=float(g.mean()), ci95_two_level=[float(lo), float(hi)],
                                              per_seed_means=per_seed.tolist())
    # 6. So với nhiễu: |Δt| do tấn công vs độ lệch clean-clean giữa hai seed liên tiếp
    cc = np.abs(c['nms'][:, 1:] - c['nms'][:, :-1]).ravel()
    res['noise_vs_effect'] = dict(
        clean_clean_abs_diff_ms=dict(median=float(np.median(cc)), p90=float(np.quantile(cc, .9))),
        attack_abs_diff_ms={k: dict(median=float(np.median(np.abs(d[k]['nms'] - c['nms']))),
                                    p90=float(np.quantile(np.abs(d[k]['nms'] - c['nms']), .9))) for k in ATTACKS})
    (run_dir / 'summary.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
