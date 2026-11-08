"""Phân tích E10 theo PREREG_E10.md.   python analyze.py RUN_DIR E1_SUMMARY E1_RAW OUT_JSON

Đơn vị: (ảnh, phương pháp, seed); mỗi ô tóm bằng trung vị các lần lặp; mẫu số sạch = trung vị của clean_a và clean_b cùng ảnh.
CI: bootstrap hai tầng (ảnh, seed) cho ma trận [ảnh, seed], 20.000 lần, seed 20261021.
Mô hình Amdahl (đặc tả cố định trong PREREG_E10, không tham số khớp):
  R_pred = 1 + [f(N_patched) - f(N_clean)] / T_clean
  f = nội suy log-log đường cong NMS của E1 (nc 80, không cap, hình học mixed; lưới A = 10000 cho N >= 100, lưới A = 2100 cho N < 100),
  N = số ứng viên vào NMS ở τ 0,25; T_clean = toàn trình ảnh sạch cùng ảnh, cùng backend, trong chính phiên này.
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import stats  # noqa: E402

SEED = 20261021
spec = importlib.util.spec_from_file_location('e9a', SIM / 'studies' / 'e9_pipeline' / 'analyze.py')
e9a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e9a)


def curve(curves, be, geom):
    pts = {}
    for k, v in curves.items():
        p, A, b, g, nc, cap, n = k.split('|')
        if b == be and g == geom and nc == '80' and cap == 'None' and int(n) > 0:
            if (A == '10000' and int(n) >= 100) or (A == '2100' and int(n) < 100):
                pts[int(n)] = v['median_ms']
    ns = np.array(sorted(pts))
    return ns, np.array([pts[n] for n in ns])


def f_interp(ns, ms, n):
    n = min(max(n, ns[0]), ns[-1])
    return float(np.exp(np.interp(np.log(n), np.log(ns), np.log(ms))))


def main():
    run, e1s, e1raw, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    curves = json.loads(e1s.read_text(encoding='utf-8'))['curves']
    micro, pipe, nmap, refs, pin, pin_end = defaultdict(list), defaultdict(list), {}, [], None, None
    for line in open(next(run.glob('raw_*.jsonl')), encoding='utf-8'):
        r = json.loads(line)
        if r['kind'] == 'micro':
            micro[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['nms_ns'] / 1e6)
            nmap[(r['image_id'], r['arm'], r['seed'])] = (r['n_by_tau'], r['nms_input'], r['kept'])
        elif r['kind'] == 'pipeline':
            pipe[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['total_ns'] / 1e6)
        elif r['kind'] == 'reference':
            refs.append(r)
        elif r['kind'] == 'pin':
            pin = r['verify']
        elif r['kind'] == 'pin_end':
            pin_end = r['verify']
    mm = {k: float(np.median(v)) for k, v in micro.items()}
    pm = {k: float(np.median(v)) for k, v in pipe.items()}
    ids = sorted({k[0] for k in pm})
    seeds = sorted({k[2] for k in pm if k[2] is not None})
    methods = sorted({k[1] for k in pm if k[1] not in ('clean_a', 'clean_b')})
    pbe = sorted({k[3] for k in pm})
    mbe = sorted({k[3] for k in mm})
    out = dict(experiment='E10', bootstrap_seed=SEED, n_boot=stats.N_BOOT, n_images=len(ids), seeds=seeds,
               pin=dict(effective_cores=pin['effective_cores'], passed=pin['passed']),
               pin_end=dict(effective_cores=pin_end['effective_cores'], passed=pin_end['passed']) if pin_end else None,
               drift=dict(points=len(refs), flagged=sum(bool(r['flagged']) for r in refs), max_abs=max(abs(r['rel_change']) for r in refs)),
               in_session_mde_pct={}, n_ratio={}, n_median={}, total_ratio={}, amdahl={}, nms_predictor={})
    # sàn nhiễu trong phiên (clean_b so với clean_a)
    for be in pbe:
        d = [np.log(pm[(i, 'clean_b', None, be)] / pm[(i, 'clean_a', None, be)]) for i in ids]
        out['in_session_mde_pct'][be] = 100 * stats.mde_paired(float(np.std(d, ddof=1)), len(d))
    clean_t = {(i, be): float(np.median([pm[(i, 'clean_a', None, be)], pm[(i, 'clean_b', None, be)]])) for i in ids for be in pbe}
    clean_n = {i: nmap[(i, 'clean_a', None)] for i in ids}
    for m in methods:
        for tau in ('0.01', '0.25'):
            M = np.array([[nmap[(i, m, s)][0][tau] / max(clean_n[i][0][tau], 1) for s in seeds] for i in ids], float)
            out['n_ratio'][f'{m}|tau{tau}'] = dict(mean=float(M.mean()), ci95=stats.two_level_bootstrap(M, SEED).tolist())
            out['n_median'][f'{m}|tau{tau}'] = dict(clean=float(np.median([clean_n[i][0][tau] for i in ids])),
                                                    patched=float(np.median([nmap[(i, m, s)][0][tau] for i in ids for s in seeds])))
        for be in pbe:
            R = np.array([[pm[(i, m, s, be)] / clean_t[(i, be)] for s in seeds] for i in ids])
            lo, hi = stats.two_level_bootstrap(R, SEED)
            out['total_ratio'][f'{m}|{be}'] = dict(mean=float(R.mean()), ci95=[float(lo), float(hi)],
                                                   exceeds_in_session_mde=bool(lo > 1 + out['in_session_mde_pct'][be] / 100))
            ns_c, ms_c = curve(curves, be, 'mixed')
            E = np.zeros_like(R)
            Rp = np.zeros_like(R)
            for a, i in enumerate(ids):
                for b, s in enumerate(seeds):
                    rp = 1 + (f_interp(ns_c, ms_c, nmap[(i, m, s)][1]) - f_interp(ns_c, ms_c, clean_n[i][1])) / clean_t[(i, be)]
                    Rp[a, b] = rp
                    E[a, b] = abs(np.log(R[a, b] / rp))
            lo_e, hi_e = stats.two_level_bootstrap(E, SEED, stat=np.nanmedian)
            out['amdahl'][f'{m}|{be}'] = dict(r_pred_mean=float(Rp.mean()), r_meas_mean=float(R.mean()),
                                              median_abs_log_err=float(np.median(E)), ci95=[float(lo_e), float(hi_e)],
                                              passes_prereg_rule=bool(hi_e < 0.05))
    # bộ dự đoán thời gian NMS trên tập ứng viên thật (micro, τ 0,25)
    train = e9a.train_from_e1(e1raw)
    test = defaultdict(list)
    for (i, arm, s, be), t in mm.items():
        n_in, kept = nmap[(i, arm, s)][1], nmap[(i, arm, s)][2]
        if n_in > 0 and arm != 'clean_b':
            test[be].append(dict(image=i, lnN=np.log(n_in), lnK=np.log(max(kept, 1)), lnT=np.log(t)))
    out['nms_predictor'] = e9a.evaluate(train, test)
    for be in mbe:
        ns_c, ms_c = curve(curves, be, 'mixed')
        errs = [abs(np.log(mm[(i, arm, s, be)] / f_interp(ns_c, ms_c, nmap[(i, arm, s)][1])))
                for (i, arm, s, b) in mm if b == be and arm not in ('clean_b',) and nmap[(i, arm, s)][1] >= 100]
        out['nms_predictor'].setdefault(be, {})['interp_mixed_median_abs_log_err_N_ge_100'] = float(np.median(errs)) if errs else None
    out_path.write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ('pin', 'drift', 'in_session_mde_pct')}, default=str))


if __name__ == '__main__':
    main()
