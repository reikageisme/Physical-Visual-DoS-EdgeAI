"""Phân tích E11 theo prereg/PREREG_E11.md.   python analyze.py RUN_DIR [--exclude-flagged]

Đơn vị suy luận: cặp (ảnh, seed patch); CI = bootstrap hai tầng ảnh × seed, 20.000 lần, seed 20261021.
Mỗi ô (ảnh, nhánh, seed, ngưỡng, backend) tóm bằng trung vị các lần lặp. Mẫu số sạch = trung bình của hai nhánh sạch
cùng canvas (clean_a, clean_b hoặc clean640_a, clean640_b) trên cùng ảnh, trong cùng phiên.
Sàn nhiễu trong phiên: MDE của log(clean_b/clean_a) theo ảnh (stats.mde_paired, 80% công suất, alpha 0,05).
--exclude-flagged: bỏ các khối 10 ảnh có điểm tham chiếu đầu khối bị cờ drift (> 3%); chỉ là phân tích độ nhạy.
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
BACKENDS = ['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native']
PIPE = ['torch_pure', 'tv_nms']
CLEAN = {320: ('clean_a', 'clean_b'), 640: ('clean640_a', 'clean640_b')}
CANVAS = dict(C1=320, C3=320, B1=320, B2=320, B3=320, B4=320, C2=640, C4=640)


def load(run_dir):
    micro, pipe, nmap, refs, pin = defaultdict(list), defaultdict(list), {}, [], None
    for line in next(Path(run_dir).glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['kind'] == 'pin':
            pin = r['verify']
        elif r['kind'] == 'reference':
            refs.append(r)
        elif r['kind'] == 'micro':
            micro[(r['image_id'], r['arm'], r['seed'], r['tau'], r['backend'])].append(r['nms_ns'] / 1e6)
            nmap[(r['image_id'], r['arm'], r['seed'], r['tau'])] = r['nms_input']
        elif r['kind'] == 'pipeline':
            pipe[(r['image_id'], r['arm'], r['seed'], r['backend'], r['block'])].append(r['total_ns'] / 1e6)
    return micro, pipe, nmap, refs, pin


def main():
    run_dir = Path(sys.argv[1])
    excl = '--exclude-flagged' in sys.argv
    micro, pipe, nmap, refs, pin = load(run_dir)
    flagged_blocks = {r['block'] for r in refs if r['flagged'] and r['block'] != 'end'}
    block_of = {k[0]: k[4] for k in pipe}
    keep = lambda i: not (excl and block_of[i] in flagged_blocks)  # noqa: E731
    med = {k: float(np.median(v)) for k, v in micro.items() if keep(k[0])}
    pmed = {k[:4]: float(np.median(v)) for k, v in pipe.items() if keep(k[0])}
    out = dict(experiment='E11', run_dir=str(run_dir), exclude_flagged_blocks=excl, bootstrap_seed=SEED, n_boot=stats.N_BOOT,
               pin=dict(effective_cores=pin['effective_cores'], passed=pin['passed']),
               drift=dict(points=len(refs), flagged=sum(r['flagged'] for r in refs), flagged_blocks=sorted(flagged_blocks),
                          max_abs=max(abs(r['rel_change']) for r in refs)),
               in_session={}, arms={})
    for canvas, (ca, cb) in CLEAN.items():
        ids = sorted({k[0] for k in pmed if k[1] == ca})
        d = {}
        for be in PIPE:
            lr = np.array([np.log(pmed[(i, cb, None, be)] / pmed[(i, ca, None, be)]) for i in ids])
            mde = 100 * stats.mde_paired(float(np.std(lr, ddof=1)), len(lr))
            t_clean = float(np.median([np.mean([pmed[(i, ca, None, be)], pmed[(i, cb, None, be)]]) for i in ids]))
            d[be] = dict(n_images=len(ids), aa_log_ratio_mean=float(lr.mean()), mde_total_pct=mde,
                         t_total_clean_median_ms=t_clean, threshold_delta_nms_ms=mde / 100 * t_clean)
        out['in_session'][str(canvas)] = d
    for arm, canvas in CANVAS.items():
        ca, cb = CLEAN[canvas]
        ids = sorted({k[0] for k in pmed if k[1] == arm})
        seeds = sorted({k[2] for k in pmed if k[1] == arm})
        if not ids:
            continue
        a = dict(canvas=canvas, n_images=len(ids), seeds=seeds)
        for tau in (0.01, 0.25):
            nc = np.array([[nmap[(i, ca, None, tau)] for _ in seeds] for i in ids], float)
            npat = np.array([[nmap[(i, arm, s, tau)] for s in seeds] for i in ids], float)
            ratio = npat / np.maximum(nc, 1)
            a[f'N_tau{tau}'] = dict(clean_median=float(np.median(nc[:, 0])), patched_median=float(np.median(npat)),
                                    ratio_mean=float(ratio.mean()), ratio_ci95=stats.two_level_bootstrap(ratio, SEED).tolist())
            for be in BACKENDS:
                tc = np.array([[np.mean([med[(i, ca, None, tau, be)], med[(i, cb, None, tau, be)]]) for _ in seeds] for i in ids])
                tp = np.array([[med[(i, arm, s, tau, be)] for s in seeds] for i in ids])
                rt = tp / tc
                a[f'nms_tau{tau}_{be}'] = dict(ratio_mean=float(rt.mean()), ratio_ci95=stats.two_level_bootstrap(rt, SEED).tolist(),
                                               delta_ms_mean=float((tp - tc).mean()))
        for be in PIPE:
            tc = np.array([[np.mean([pmed[(i, ca, None, be)], pmed[(i, cb, None, be)]]) for _ in seeds] for i in ids])
            tp = np.array([[pmed[(i, arm, s, be)] for s in seeds] for i in ids])
            rt = tp / tc
            lo, hi = (float(x) for x in stats.two_level_bootstrap(rt, SEED))
            nc_ = np.array([[np.mean([med[(i, ca, None, 0.01, be)], med[(i, cb, None, 0.01, be)]]) for _ in seeds] for i in ids])
            np_ = np.array([[med[(i, arm, s, 0.01, be)] for s in seeds] for i in ids])
            pred = 1 + (np_ - nc_) / tc
            mde = out['in_session'][str(canvas)][be]['mde_total_pct']
            a[f'total_{be}'] = dict(ratio_mean=float(rt.mean()), ratio_ci95=[lo, hi], exceeds_in_session_mde=bool(lo > 1 + mde / 100),
                                    delta_nms_ms_mean=float((np_ - nc_).mean()), delta_nms_ms_ci95=stats.two_level_bootstrap(np_ - nc_, SEED).tolist(),
                                    amdahl_pred_mean=float(pred.mean()), amdahl_pred_inside_ci=bool(lo <= pred.mean() <= hi),
                                    nms_share_clean_median=float(np.median(nc_ / tc)))
        out['arms'][arm] = a
    name = 'summary_excl_flagged.json' if excl else 'summary.json'
    (run_dir / name).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('pin', 'drift', 'in_session')}, ensure_ascii=False))
    for arm, a in out['arms'].items():
        t = a['total_tv_nms']
        print(arm, 'N', round(a['N_tau0.01']['ratio_mean'], 3), 'tot_tv %.4f [%.4f, %.4f]' % (t['ratio_mean'], *t['ratio_ci95']),
              'pred %.4f' % t['amdahl_pred_mean'], 'exceeds', t['exceeds_in_session_mde'])


if __name__ == '__main__':
    main()
