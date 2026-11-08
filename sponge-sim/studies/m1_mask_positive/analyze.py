"""Phân tích M1 theo PREREG_M1. Đối chứng dương đầy đủ (bản cho bài).

  python analyze.py --run-dir RUN
  python analyze.py --selftest

Đơn vị suy luận = ẢNH. Mỗi ô (model, ảnh, K, retina) tóm bằng trung vị các lần lặp. CI bootstrap theo ảnh (20.000 lần).
- M1-P1 K*: tỷ số toàn trình total@K/total@baseline; K* = K nhỏ nhất có cận dưới KTC > 1 + MDE_total.
- M1-P2 ngân sách: R_pred = 1 + (mask@K − mask@baseline)/total@baseline so R_meas = total@K/total@baseline; đạt nếu |ln(Rpred/Rmeas)| ≤ MDE.
- M1-P3 liều–đáp ứng: đường tỷ số theo K. Phụ: phân rã tỷ phần forward/nms/mask của total.
MDE_total: A/A trong phiên từ K=0 (tách đôi, log tỷ số) -> mde_paired.
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

SEED = 20261104
STAGES = ('total_ns', 'forward_ns', 'nms_ns', 'mask_ns')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def _median_cells(rows):
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['model'], r['image_id'], r['k'], r['retina'])
        for f in STAGES:
            acc[key][f].append(r[f])
    return {k: {f: float(np.median(v[f])) / 1e6 for f in STAGES} for k, v in acc.items()}


def _mde_total(rows, seed=SEED):
    by = defaultdict(list)
    for r in rows:
        if r['k'] == 0:
            by[(r['model'], r['image_id'], r['retina'])].append(r['total_ns'])
    rng = np.random.default_rng(seed)
    per = defaultdict(list)
    for (model, _img, retina), vals in by.items():
        if len(vals) < 2:
            continue
        v = np.asarray(vals, float)
        idx = rng.permutation(len(v))
        per[(model, retina)].append(np.log(np.median(v[idx[:len(v) // 2]]) / np.median(v[idx[len(v) // 2:]])))
    out = {}
    for key, logs in per.items():
        logs = np.asarray(logs)
        if len(logs) < 2:
            out[f'{key[0]}|retina{key[1]}'] = None
            continue
        mde = stats.mde_paired(float(logs.std(ddof=1)), len(logs))
        out[f'{key[0]}|retina{key[1]}'] = dict(n_images=len(logs), mde_frac=float(mde), mde_pct=float(mde * 100))
    return out


def analyze(run_dir, exclude_flagged=False):
    run_dir = Path(run_dir)
    all_rows = list(read_jsonl(next(run_dir.glob('raw_*.jsonl'))))
    rows = [r for r in all_rows if r.get('kind') == 'cell']
    meta = {r['kind']: r for r in all_rows if r.get('kind') in ('pin', 'pin_end')}
    refs = [r for r in all_rows if r.get('kind') == 'reference']
    flagged_blocks = {r['block'] for r in refs if r.get('flagged')}
    if exclude_flagged:   # độ nhạy: bỏ cell thuộc block có tham chiếu bị cờ drift
        rows = [r for r in rows if r['block'] not in flagged_blocks]
    med = _median_cells(rows)
    mde = _mde_total(rows)
    models = sorted({k[0] for k in med})
    images = sorted({k[1] for k in med})
    ks = sorted({k[2] for k in med})
    baseline_k = 10 if 10 in ks else min(x for x in ks if x > 0)
    out = dict(experiment='M1', run_dir=str(run_dir), device=(rows[0].get('device', 'cpu') if rows else 'cpu'),
               bootstrap_seed=SEED, n_boot=stats.N_BOOT, baseline_k=baseline_k, mde_total=mde,
               pin={k: dict(effective_cores=meta[k]['verify']['effective_cores'], passed=meta[k]['verify']['passed'])
                    for k in ('pin', 'pin_end') if k in meta},
               exclude_flagged=exclude_flagged, excluded_blocks=sorted(flagged_blocks, key=str) if exclude_flagged else [],
               drift=dict(points=len(refs), flagged=sum(bool(r.get('flagged')) for r in refs),
                          max_abs=max((abs(r['rel_change']) for r in refs if 'rel_change' in r), default=None)),
               total_ratio={}, k_star={}, budget={}, stage_share={}, stage_ms={})
    for model in models:
        for retina in (False, True):
            mde_frac = (mde.get(f'{model}|retina{retina}') or {}).get('mde_frac') if isinstance(mde.get(f'{model}|retina{retina}'), dict) else None
            kstar = None
            for k in [x for x in ks if x > baseline_k]:
                logs = [np.log(med[(model, i, k, retina)]['total_ns'] / med[(model, i, baseline_k, retina)]['total_ns'])
                        for i in images if (model, i, k, retina) in med and (model, i, baseline_k, retina) in med]
                if not logs:
                    continue
                lo, hi = stats.bootstrap_ci(logs, SEED)
                ratio, ci = float(np.exp(np.mean(logs))), [float(np.exp(lo)), float(np.exp(hi))]
                detected = mde_frac is not None and ci[0] > 1 + mde_frac
                tag = f'{model}|retina{retina}|K{k}'
                out['total_ratio'][tag] = dict(ratio=ratio, ci95=ci, n_images=len(logs), detected_vs_mde=detected)
                if detected and kstar is None:
                    kstar = k
                # M1-P2 ngân sách
                base = [(med[(model, i, baseline_k, retina)]['mask_ns'], med[(model, i, baseline_k, retina)]['total_ns'])
                        for i in images if (model, i, k, retina) in med and (model, i, baseline_k, retina) in med]
                dmask = float(np.median([med[(model, i, k, retina)]['mask_ns'] - b[0] for i, b in zip(
                    [i for i in images if (model, i, k, retina) in med and (model, i, baseline_k, retina) in med], base)]))
                tbase = float(np.median([b[1] for b in base]))
                r_pred = 1 + dmask / tbase
                log_err = abs(np.log(r_pred / ratio))
                out['budget'][tag] = dict(r_pred=r_pred, r_meas=ratio, log_err=float(log_err),
                                          pass_vs_mde=bool(mde_frac is not None and log_err <= mde_frac))
            out['k_star'][f'{model}|retina{retina}'] = kstar if kstar is not None else f'>{max(ks)}'
            for k in ks:
                cells = [med[(model, i, k, retina)] for i in images if (model, i, k, retina) in med]
                if not cells:
                    continue
                tag = f'{model}|retina{retina}|K{k}'
                tot = np.median([c['total_ns'] for c in cells])
                out['stage_share'][tag] = {s: float(np.median([c[s] for c in cells]) / tot) for s in ('forward_ns', 'nms_ns', 'mask_ns')}
                out['stage_ms'][tag] = {s: float(np.median([c[s] for c in cells])) for s in STAGES}
    out['verdict'] = _overall(out['k_star'])
    return out


def _overall(k_star):
    reach = {k: v for k, v in k_star.items() if isinstance(v, int)}
    if reach:
        return f'K* trong tầm max_det cho: {reach} (bề mặt tấn công đạt được; chờ M3 chứng minh tấn công đẩy K tới đó)'
    return 'Không có K* ≤ max_det ở ô nào trên cấu hình này (kết quả âm kèm MDE)'


def selftest():
    """total = forward + nms + mask; mask = c·K (retina lớn). Kiểm K* tìm được và ngân sách khớp (log_err≈0)."""
    rng = np.random.default_rng(0)
    rows = []
    for img in range(20):
        fwd, nm = 180e6 * (1 + rng.normal(0, .023)), 6e6 * (1 + rng.normal(0, .02))
        for retina in (False, True):
            per_k = (0.6e6 if retina else 0.02e6)
            for k in (0, 10, 50, 100, 300):
                for copy in range(3 if k == 0 else 1):
                    for _ in range(4):
                        mask = per_k * k * (1 + rng.normal(0, .005))
                        rows.append(dict(kind='cell', model='synth-seg.pt', image_id=img, k=k, retina=retina, copy=copy,
                                         total_ns=fwd + nm + mask, forward_ns=fwd, nms_ns=nm, mask_ns=mask))
    med = _median_cells(rows)
    mde = _mde_total(rows)
    ks, budget = {}, {}
    for retina in (False, True):
        mf = mde[f'synth-seg.pt|retina{retina}']['mde_frac']
        kstar, bok = None, True
        for k in (50, 100, 300):
            logs = [np.log(med[('synth-seg.pt', i, k, retina)]['total_ns'] / med[('synth-seg.pt', i, 10, retina)]['total_ns']) for i in range(20)]
            lo, _ = stats.bootstrap_ci(logs, SEED)
            ratio = float(np.exp(np.mean(logs)))
            if float(np.exp(lo)) > 1 + mf and kstar is None:
                kstar = k
            dmask = float(np.median([med[('synth-seg.pt', i, k, retina)]['mask_ns'] - med[('synth-seg.pt', i, 10, retina)]['mask_ns'] for i in range(20)]))
            tbase = float(np.median([med[('synth-seg.pt', i, 10, retina)]['total_ns'] for i in range(20)]))
            if abs(np.log((1 + dmask / tbase) / ratio)) > 0.02:   # ngân sách phải khớp sát vì total = forward+nms+mask
                bok = False
        ks[retina], budget[retina] = kstar, bok
    # claim khoa học robust: retina=True phát hiện sớm (K* nhỏ) và sớm hơn/bằng retina=False; ngân sách khớp cả hai
    ok = (budget[True] and budget[False] and ks[True] is not None and ks[True] <= 100
          and (ks[False] is None or ks[True] <= ks[False]))
    print('kstar', ks, 'budget', budget)
    print('SELFTEST_M1', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--out')
    ap.add_argument('--exclude-flagged', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    out = analyze(a.run_dir, exclude_flagged=a.exclude_flagged)
    path = Path(a.out) if a.out else Path(a.run_dir) / ('summary_excl.json' if a.exclude_flagged else 'summary.json')
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print(path)
    print('VERDICT:', out['verdict'])


if __name__ == '__main__':
    main()
