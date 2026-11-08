"""Phân tích M0 theo PREREG_M0_mask_microbench.md. Cổng go/no-go.

  python analyze.py --run-dir RUN
  python analyze.py --selftest        # kiểm logic quyết định trên dữ liệu tổng hợp (không cần run thật)

Đơn vị suy luận = ẢNH. Mỗi ô (model, ảnh, K, retina) tóm bằng trung vị các lần lặp. CI bootstrap theo ảnh (20.000 lần).
- frac_mask@K = mask / total (tỷ phần công đoạn dựng mask trong total = forward + mask).
- MDE_total: A/A trong phiên từ các lần đo lặp của K=0 (tách đôi ngẫu nhiên, log tỷ số total) -> mde_paired.
- rel_gain(retina) = trung vị theo ảnh của (mask@300 − mask@baseline_k) / total@baseline_k:
  mức tăng toàn trình (tỷ phần) nếu tấn công đẩy K từ nền lên 300. Đây là đại lượng quyết định.
Quy tắc (định trước, retina=True, cấu hình bất lợi nhất cho H1): GO nếu rel_gain ≥ max(2·MDE, 0.05);
NO-GO nếu < đó; BIÊN nếu ≥ MDE nhưng < 0.05 hoặc chỉ đạt khi retina=True.
Lưu ý: total bỏ decode/NMS (~2-4%) -> mẫu số hơi nhỏ, rel_gain là cận trên -> NO-GO là kết luận bảo thủ.
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
PRACTICAL = 0.05  # ngưỡng ý nghĩa thực tiễn đặt trước của nhóm


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def _median_cells(rows):
    """(model, image, k, retina) -> median ns mỗi trường. K=0 gộp mọi copy."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r['model'], r['image_id'], r['k'], r['retina'])
        for f in ('total_ns', 'mask_ns', 'forward_ns'):
            acc[key][f].append(r[f])
    return {k: {f: float(np.median(v[f])) / 1e6 for f in ('total_ns', 'mask_ns', 'forward_ns')} for k, v in acc.items()}


def _mde_total(rows, seed=SEED):
    """A/A: với mỗi (model, image, retina), tách các phép đo total của K=0 làm đôi, log tỷ số trung vị.
    sd theo ảnh -> mde_paired. Trả % cho từng (model, retina)."""
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
        a, b = v[idx[:len(v) // 2]], v[idx[len(v) // 2:]]
        per[(model, retina)].append(np.log(np.median(a) / np.median(b)))
    out = {}
    for key, logs in per.items():
        logs = np.asarray(logs)
        if len(logs) < 2:
            out[f'{key[0]}|retina{key[1]}'] = None
            continue
        mde = stats.mde_paired(float(logs.std(ddof=1)), len(logs))
        out[f'{key[0]}|retina{key[1]}'] = dict(n_images=len(logs), mde_frac=float(mde), mde_pct=float(mde * 100))
    return out


def analyze(run_dir):
    run_dir = Path(run_dir)
    rows = [r for r in read_jsonl(next(run_dir.glob('raw_*.jsonl'))) if r.get('kind') == 'cell']
    meta = {r['kind']: r for r in read_jsonl(next(run_dir.glob('raw_*.jsonl'))) if r.get('kind') in ('pin', 'pin_end')}
    refs = [r for r in read_jsonl(next(run_dir.glob('raw_*.jsonl'))) if r.get('kind') == 'reference']
    med = _median_cells(rows)
    mde = _mde_total(rows)
    models = sorted({k[0] for k in med})
    images = sorted({k[1] for k in med})
    ks = sorted({k[2] for k in med})
    baseline_k = 10 if 10 in ks else min(x for x in ks if x > 0)
    out = dict(experiment='M0', run_dir=str(run_dir), bootstrap_seed=SEED, n_boot=stats.N_BOOT,
               baseline_k=baseline_k, practical_threshold=PRACTICAL, mde_total=mde,
               pin={k: dict(effective_cores=meta[k]['verify']['effective_cores'], passed=meta[k]['verify']['passed'])
                    for k in ('pin', 'pin_end') if k in meta},
               drift=dict(points=len(refs), flagged=sum(bool(r.get('flagged')) for r in refs),
                          max_abs=max((abs(r['rel_change']) for r in refs if 'rel_change' in r), default=None)),
               frac_mask={}, mask_ms={}, forward_ms={}, total_ms={}, rel_gain={}, go_no_go={})
    for model in models:
        for retina in (False, True):
            # curve mô tả
            for k in ks:
                cells = [med[(model, i, k, retina)] for i in images if (model, i, k, retina) in med]
                if not cells:
                    continue
                tag = f'{model}|retina{retina}|K{k}'
                if k > 0:
                    fr = [c['mask_ns'] / c['total_ns'] for c in cells]
                    lo, hi = stats.bootstrap_ci(fr, SEED)
                    out['frac_mask'][tag] = dict(median=float(np.median(fr)), ci95=[float(lo), float(hi)], n_images=len(fr))
                out['mask_ms'][tag] = float(np.median([c['mask_ns'] for c in cells]))
                out['forward_ms'][tag] = float(np.median([c['forward_ns'] for c in cells]))
                out['total_ms'][tag] = float(np.median([c['total_ns'] for c in cells]))
            # đại lượng quyết định: (mask@300 − mask@baseline)/total@baseline, theo ảnh
            gains = [(med[(model, i, 300, retina)]['mask_ns'] - med[(model, i, baseline_k, retina)]['mask_ns'])
                     / med[(model, i, baseline_k, retina)]['total_ns']
                     for i in images if (model, i, 300, retina) in med and (model, i, baseline_k, retina) in med]
            if not gains:
                continue
            lo, hi = stats.bootstrap_ci(gains, SEED)
            key = f'{model}|retina{retina}'
            out['rel_gain'][key] = dict(median=float(np.median(gains)), ci95=[float(lo), float(hi)], n_images=len(gains))
            m = mde.get(key)
            mde_frac = m['mde_frac'] if isinstance(m, dict) else None
            out['go_no_go'][key] = _verdict(float(np.median(gains)), mde_frac)
    out['verdict_overall'] = _overall(out['go_no_go'])
    return out


def _verdict(rel_gain, mde_frac):
    if mde_frac is None:
        return dict(rel_gain=rel_gain, verdict='KHÔNG ĐỦ A/A (thiếu MDE)')
    thr = max(2 * mde_frac, PRACTICAL)
    if rel_gain >= thr:
        v = 'GO'
    elif rel_gain > mde_frac:
        v = 'BIÊN'
    else:
        v = 'NO-GO'
    return dict(rel_gain=rel_gain, mde_frac=mde_frac, threshold=thr, verdict=v)


def _overall(go_no_go):
    retina_true = [v for k, v in go_no_go.items() if k.endswith('retinaTrue')]
    if any(v.get('verdict') == 'GO' for v in retina_true):
        return 'GO (ít nhất một model: dựng mask đủ lớn ở K=300 với retina_masks)'
    if any(v.get('verdict') == 'BIÊN' for v in retina_true):
        return 'BIÊN (có tín hiệu nhưng dưới ngưỡng thực tiễn; đo thêm model seg thứ hai)'
    return 'NO-GO cho YOLO-seg CPU (chuyển trục GPU hoặc hướng dự phòng SAM); giữ kết quả âm kèm MDE'


def selftest():
    """Dữ liệu tổng hợp: forward cố định 200 ms; mask = c·K, c rất nhỏ khi retina=False, lớn khi retina=True.
    Chọn số để retina=True GO (mask ~0,84 total) và retina=False NO-GO (mask dưới sàn nhiễu); kiểm logic quyết định."""
    rng = np.random.default_rng(0)
    rows = []
    for img in range(12):
        fwd = 200e6 * (1 + rng.normal(0, .003))
        for retina in (False, True):
            per_k = (0.6e6 if retina else 0.0)  # ns mỗi đối tượng (retina=False: mask không đáng kể -> NO-GO)
            for k in (0, 10, 50, 100, 300):
                for copy in range(3 if k == 0 else 1):
                    for _ in range(4):
                        mask = per_k * k * (1 + rng.normal(0, .01))
                        total = fwd + mask
                        rows.append(dict(kind='cell', model='synth-seg.pt', image_id=img, k=k, retina=retina,
                                         copy=copy, total_ns=total, mask_ns=mask, forward_ns=fwd))
    med = _median_cells(rows)
    mde = _mde_total(rows)
    gains = {}
    for retina in (False, True):
        g = [(med[('synth-seg.pt', i, 300, retina)]['mask_ns'] - med[('synth-seg.pt', i, 10, retina)]['mask_ns'])
             / med[('synth-seg.pt', i, 10, retina)]['total_ns'] for i in range(12)]
        gains[retina] = _verdict(float(np.median(g)), mde[f'synth-seg.pt|retina{retina}']['mde_frac'])
    # retina=True: (0.6e6*(300-10))/(200e6+0.6e6*10) ≈ 0.845 -> GO; retina=False: mask ~0 -> NO-GO
    bands = (_verdict(0.80, 0.005)['verdict'] == 'GO' and _verdict(0.03, 0.005)['verdict'] == 'BIÊN'
             and _verdict(0.001, 0.005)['verdict'] == 'NO-GO')
    ok = gains[True]['verdict'] == 'GO' and gains[False]['verdict'] == 'NO-GO' and bands
    print(json.dumps({str(r): (round(v['rel_gain'], 4), v['verdict']) for r, v in gains.items()}, ensure_ascii=False))
    print('SELFTEST_M0', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    out = analyze(a.run_dir)
    path = Path(a.out) if a.out else Path(a.run_dir) / 'summary.json'
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print(path)
    print('VERDICT:', out['verdict_overall'])


if __name__ == '__main__':
    main()
