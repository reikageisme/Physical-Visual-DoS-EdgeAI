"""Phân tích E9 theo PREREG_E9.md.

Phần A:  python analyze.py --part a --run-dir RUN --aa320 AA320_SUMMARY --aa640 AA640_SUMMARY
Phần B:  python analyze.py --part b --e1 E1_RAW_JSONL --test320 E6_RUN_DIR --test640 E6A1_RUN_DIR --out OUT.json
Tự kiểm mã Phần B trên dữ liệu tổng hợp (KHÔNG đọc dữ liệu kiểm tra thật):  python analyze.py --selftest-b

Phần A: đơn vị suy luận là ảnh; mỗi ô (độ phân giải, ảnh, N, hình học, cap, backend) tóm bằng trung vị các lần lặp; CI bootstrap theo ảnh
(20.000 lần, seed 20261021) trên log tỷ số. N* = N nhỏ nhất có cận dưới CI > 1 + MDE_total của độ phân giải đó.
Phần B: M_N (ln t ~ ln N) và M_NK (thêm ln max(K,1)) khớp riêng từng backend trên E1 micro (nc 80, cap None, N > 0); đánh giá trên tập
ứng viên thật; Δ = trung vị |ln(đo/dự đoán)| của M_N trừ của M_NK, CI bootstrap hai tầng (ảnh, rồi đơn vị trong ảnh).
"""
from __future__ import annotations

import argparse
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


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0., c - h)), float(min(1., c + h))]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


# ------------------------------------------------------------------ Phần A
def part_a(run_dir, aa320, aa640):
    mde = {320: json.loads(Path(aa320).read_text(encoding='utf-8'))['profiles']['P1_cpu2']['stages']['total']['mde_ratio_pct'],
           640: json.loads(Path(aa640).read_text(encoding='utf-8'))['profiles']['P1_cpu2']['stages']['total']['mde_ratio_pct']}
    cell = defaultdict(lambda: defaultdict(list))
    refs, pin, pin_end = [], None, None
    for r in read_jsonl(next(Path(run_dir).glob('raw_*.jsonl'))):
        if r['kind'] == 'cell':
            k = (r['canvas'], r['image_id'], r['n'], r['geometry'], r['cap'], r['backend'])
            for f in ('total_ns', 'nms_ns', 'tau_ns'):
                cell[k][f].append(r[f])
            cell[k]['miss'].append(r['miss'])
            cell[k]['kept'] = r['kept']
        elif r['kind'] == 'reference':
            refs.append(r)
        elif r['kind'] == 'pin':
            pin = r['verify']
        elif r['kind'] == 'pin_end':
            pin_end = r['verify']
    med = {k: {f: float(np.median(v[f])) / 1e6 for f in ('total_ns', 'nms_ns', 'tau_ns')} for k, v in cell.items()}
    out = dict(experiment='E9', part='A', run_dir=str(run_dir), bootstrap_seed=SEED, n_boot=stats.N_BOOT, mde_total_pct=mde,
               pin=dict(effective_cores=pin['effective_cores'], passed=pin['passed']) if pin else None,
               pin_end=dict(effective_cores=pin_end['effective_cores'], passed=pin_end['passed']) if pin_end else None,
               drift=dict(points=len(refs), flagged=sum(bool(r['flagged']) for r in refs),
                          flagged_blocks=[r['block'] for r in refs if r['flagged']],
                          max_abs=max(abs(r['rel_change']) for r in refs) if refs else None),
               ratios={}, n_star={}, cap_effect={}, nms_share={}, miss={})
    canvases = sorted({k[0] for k in med})
    for cv in canvases:
        ids = sorted({k[1] for k in med if k[0] == cv})
        ns = sorted({k[2] for k in med if k[0] == cv and k[2] > 0})
        for be in BACKENDS:
            for geom in ('disjoint', 'dense_cluster', 'mixed'):
                for cap in (None, 300):
                    series, nstar = {}, None
                    for n in ns:
                        if cap is not None and n <= cap:
                            continue
                        logs = [np.log(med[(cv, i, n, geom, cap, be)]['total_ns'] / med[(cv, i, 0, 'disjoint', None, be)]['total_ns'])
                                for i in ids if (cv, i, n, geom, cap, be) in med and (cv, i, 0, 'disjoint', None, be) in med]
                        if not logs:
                            continue
                        lo, hi = stats.bootstrap_ci(logs, SEED)
                        ratio, ci = float(np.exp(np.mean(logs))), [float(np.exp(lo)), float(np.exp(hi))]
                        detected = ci[0] > 1 + mde[cv] / 100
                        series[str(n)] = dict(ratio=ratio, ci95=ci, n_images=len(logs), detected_vs_mde=detected)
                        if detected and nstar is None:
                            nstar = n
                    key = f'{cv}|{be}|{geom}|cap{cap}'
                    if series:
                        out['ratios'][key] = series
                        out['n_star'][key] = nstar if nstar is not None else f'>{max(int(x) for x in series)}'
                # cap 300 / cap None tại N lớn nhất
                nmax = ns[-1] if ns else None
                if nmax and nmax > 300:
                    logs = [np.log(med[(cv, i, nmax, geom, 300, be)]['total_ns'] / med[(cv, i, nmax, geom, None, be)]['total_ns'])
                            for i in ids if (cv, i, nmax, geom, 300, be) in med and (cv, i, nmax, geom, None, be) in med]
                    if logs:
                        lo, hi = stats.bootstrap_ci(logs, SEED)
                        out['cap_effect'][f'{cv}|{be}|{geom}|N{nmax}'] = dict(ratio=float(np.exp(np.mean(logs))), ci95=[float(np.exp(lo)), float(np.exp(hi))])
            # tỷ phần NMS và tỷ lệ vượt mốc, theo N (disjoint, cap None)
            for n in [0] + ns:
                keys = [k for k in med if k[0] == cv and k[2] == n and k[3] == 'disjoint' and k[4] is None and k[5] == be]
                if keys:
                    out['nms_share'][f'{cv}|{be}|N{n}'] = float(np.median([med[k]['nms_ns'] / med[k]['total_ns'] for k in keys]))
                    miss = [m for k in keys for m in cell[k]['miss']]
                    out['miss'][f'{cv}|{be}|N{n}'] = dict(k=int(sum(miss)), n=len(miss), wilson95=wilson(int(sum(miss)), len(miss)))
    return out


# ------------------------------------------------------------------ Phần B
def fit(lnN, lnK, lnT, use_k):
    X = np.column_stack([np.ones_like(lnN), lnN] + ([lnK] if use_k else []))
    coef, *_ = np.linalg.lstsq(X, lnT, rcond=None)
    return coef


def predict(coef, lnN, lnK):
    X = np.column_stack([np.ones_like(lnN), lnN] + ([lnK] if len(coef) == 3 else []))
    return X @ coef


def train_from_e1(e1_raw):
    cellv = defaultdict(list)
    meta = {}
    for r in read_jsonl(e1_raw):
        if r['kind'] != 'micro' or r['nc'] != 80 or r['cap'] is not None or r['n'] <= 0:
            continue
        k = (r['backend'], r['anchors'], r['n'], r['geometry'])
        cellv[k].append(r['nms_ns'] / 1e6)
        meta[k] = (r['nms_input'], r['kept'])
    train = defaultdict(list)
    for k, v in cellv.items():
        n_in, kept = meta[k]
        train[k[0]].append((np.log(n_in), np.log(max(kept, 1)), np.log(np.median(v)), k[3]))
    return train


def load_test(run_dir, tau=0.01):
    v = defaultdict(list)
    meta = {}
    for r in read_jsonl(next(Path(run_dir).glob('raw_*.jsonl'))):
        if r['kind'] != 'micro' or r['tau'] != tau or r['nms_input'] <= 0:
            continue
        k = (r['backend'], r['image_id'], r['arm'], r['seed'])
        v[k].append(r['nms_ns'] / 1e6)
        meta[k] = (r['nms_input'], r['kept'])
    out = defaultdict(list)
    for k, t in v.items():
        n_in, kept = meta[k]
        out[k[0]].append(dict(image=k[1], lnN=np.log(n_in), lnK=np.log(max(kept, 1)), lnT=np.log(np.median(t))))
    return out


def delta_ci(units, n_boot=stats.N_BOOT, seed=SEED):
    """units: list(image, errN, errNK). Bootstrap hai tầng: rút lại ảnh, rồi rút lại đơn vị trong từng ảnh đã rút."""
    by = defaultdict(list)
    for img, a, b in units:
        by[img].append((a, b))
    imgs = list(by)
    arrs = [np.asarray(by[i]) for i in imgs]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for bi in range(n_boot):
        pick = rng.integers(0, len(imgs), len(imgs))
        parts = [arrs[p][rng.integers(0, len(arrs[p]), len(arrs[p]))] for p in pick]
        s = np.concatenate(parts)
        out[bi] = np.median(s[:, 0]) - np.median(s[:, 1])
    return np.quantile(out, [.025, .975]).tolist()


def evaluate(train, test, n_boot=stats.N_BOOT):
    res = {}
    for be in BACKENDS:
        if be not in train or be not in test:
            continue
        tr = np.asarray([x[:3] for x in train[be]], float)
        cN, cNK = fit(tr[:, 0], tr[:, 1], tr[:, 2], False), fit(tr[:, 0], tr[:, 1], tr[:, 2], True)
        lnN = np.asarray([u['lnN'] for u in test[be]])
        lnK = np.asarray([u['lnK'] for u in test[be]])
        lnT = np.asarray([u['lnT'] for u in test[be]])
        eN, eNK = np.abs(lnT - predict(cN, lnN, lnK)), np.abs(lnT - predict(cNK, lnN, lnK))
        ss = lambda e: float(1 - np.sum(e ** 2) / np.sum((lnT - lnT.mean()) ** 2))
        units = [(u['image'], a, b) for u, a, b in zip(test[be], eN, eNK)]
        res[be] = dict(n_units=len(units), n_images=len({u[0] for u in units}),
                       coef_M_N=cN.tolist(), coef_M_NK=cNK.tolist(),
                       median_abs_log_err_M_N=float(np.median(eN)), median_abs_log_err_M_NK=float(np.median(eNK)),
                       p90_M_N=float(np.quantile(eN, .9)), p90_M_NK=float(np.quantile(eNK, .9)),
                       r2_log_M_N=ss(eN), r2_log_M_NK=ss(eNK),
                       delta=float(np.median(eN) - np.median(eNK)), delta_ci95=delta_ci(units, n_boot))
        res[be]['improvement_detected'] = res[be]['delta_ci95'][0] > 0
    return res


def cross_geometry(train):
    """Mô tả: khớp trên disjoint + dense_cluster của E1, đánh giá trên mixed."""
    res = {}
    for be, rows in train.items():
        tr = np.asarray([x[:3] for x in rows if x[3] != 'mixed'], float)
        te = np.asarray([x[:3] for x in rows if x[3] == 'mixed'], float)
        if len(tr) < 4 or len(te) < 2:
            continue
        cN, cNK = fit(tr[:, 0], tr[:, 1], tr[:, 2], False), fit(tr[:, 0], tr[:, 1], tr[:, 2], True)
        res[be] = dict(median_abs_log_err_M_N=float(np.median(np.abs(te[:, 2] - predict(cN, te[:, 0], te[:, 1])))),
                       median_abs_log_err_M_NK=float(np.median(np.abs(te[:, 2] - predict(cNK, te[:, 0], te[:, 1])))))
    return res


def selftest_b():
    """Dữ liệu tổng hợp: t = N^0.3 * K^0.9 (hình học quyết định). M_NK phải tốt hơn M_N rõ rệt."""
    rng = np.random.default_rng(0)
    train = defaultdict(list)
    for be in BACKENDS:
        for n in (100, 300, 1000, 3000):
            for frac, geom in ((1.0, 'disjoint'), (0.02, 'dense_cluster'), (0.5, 'mixed')):
                k = max(1, int(n * frac))
                train[be].append((np.log(n), np.log(k), 0.3 * np.log(n) + 0.9 * np.log(k) + rng.normal(0, .02), geom))
    test = defaultdict(list)
    for be in BACKENDS:
        for img in range(30):
            for _ in range(5):
                n = rng.integers(30, 200)
                k = max(1, int(n * rng.uniform(.05, .9)))
                test[be].append(dict(image=img, lnN=np.log(n), lnK=np.log(k), lnT=0.3 * np.log(n) + 0.9 * np.log(k) + rng.normal(0, .05)))
    res = evaluate(train, test, n_boot=500)
    ok = all(r['improvement_detected'] and r['delta'] > 0.2 for r in res.values())
    print(json.dumps({be: (round(r['delta'], 3), [round(x, 3) for x in r['delta_ci95']]) for be, r in res.items()}))
    print('SELFTEST_B', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['a', 'b'])
    ap.add_argument('--run-dir')
    ap.add_argument('--aa320')
    ap.add_argument('--aa640')
    ap.add_argument('--e1')
    ap.add_argument('--test320')
    ap.add_argument('--test640')
    ap.add_argument('--out')
    ap.add_argument('--selftest-b', action='store_true')
    a = ap.parse_args()
    if a.selftest_b:
        raise SystemExit(0 if selftest_b() else 1)
    if a.part == 'a':
        out = part_a(Path(a.run_dir), a.aa320, a.aa640)
        path = Path(a.out) if a.out else Path(a.run_dir) / 'summary_part_a.json'
    else:
        train = train_from_e1(Path(a.e1))
        out = dict(experiment='E9', part='B', bootstrap_seed=SEED, n_boot=stats.N_BOOT, train=str(a.e1),
                   models={'M_N': 'ln t = a + b ln N', 'M_NK': 'ln t = a + b ln N + c ln max(K,1)'},
                   cross_geometry_e1=cross_geometry(train))
        if a.test320:
            out['test1_e6_320_posthoc'] = evaluate(train, load_test(Path(a.test320)))
        if a.test640:
            out['test2_e6a1_640_confirmatory'] = evaluate(train, load_test(Path(a.test640)))
        path = Path(a.out)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
