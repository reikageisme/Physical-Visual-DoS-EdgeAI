"""Phân tích M3 theo PREREG_M3. K là số ĐẾM THẬT sau NMS.

  python analyze.py --run-dir RUN [--k-star 50]
  python analyze.py --selftest

- M3-P1: K_best so K_clean theo ảnh; tỷ lệ ảnh đạt K_best ≥ K* (M1). Thành công nếu ≥ nửa số ảnh.
- M3-P2: quỹ đạo K theo vòng (mô tả). Đơn vị suy luận = ẢNH; CI bootstrap theo ảnh.
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

SEED = 20261106


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def analyze(run_dir, k_star=50):
    rows = [r for r in read_jsonl(next(Path(run_dir).glob('attacks.jsonl'))) if r.get('kind') == 'attack']
    by = defaultdict(list)
    for r in rows:
        by[r['model']].append(r)
    out = dict(experiment='M3', run_dir=str(run_dir), bootstrap_seed=SEED, n_boot=stats.N_BOOT, k_star_ref=k_star, models={})
    for model, rs in by.items():
        kc = np.array([r['k_clean'] for r in rs], float)
        kb = np.array([r['k_best'] for r in rs], float)
        lo, hi = stats.bootstrap_ci(kb, SEED, stat=np.median)
        reach = {str(t): int((kb >= t).sum()) for t in (k_star, 100, 300)}
        out['models'][model] = dict(
            n_images=len(rs),
            k_clean_median=float(np.median(kc)), k_best_median=float(np.median(kb)), k_best_median_ci95=[float(lo), float(hi)],
            k_best_max=float(kb.max()), gain_median=float(np.median(kb - kc)),
            n_reach=reach, frac_reach_kstar=float((kb >= k_star).mean()),
            success_vs_prereg=bool((kb >= k_star).mean() >= 0.5),
        )
    out['verdict'] = _verdict(out['models'])
    return out


def _verdict(models):
    ok = [m for m, v in models.items() if v['success_vs_prereg']]
    if ok:
        return f'ĐẠT: tấn công đẩy K ≥ K* ở ≥ nửa số ảnh cho {ok} (đưa nhiễu vào M1/M2 đo toàn trình)'
    return 'CHƯA ĐẠT: PGD+proxy chưa đẩy đủ K thật; thử proxy phân tán IoU hoặc kết luận K khó đẩy (kết quả âm, ghi DEVIATIONS)'


def selftest():
    """Dữ liệu tổng hợp: nửa ảnh K_best vượt K*=50 → ĐẠT; đổi để nửa dưới → CHƯA ĐẠT."""
    import tempfile
    import os
    def run(kbests):
        d = Path(tempfile.mkdtemp())
        with open(d / 'attacks.jsonl', 'w', encoding='utf-8') as f:
            for i, kb in enumerate(kbests):
                f.write(json.dumps(dict(kind='attack', model='m.pt', image_id=i, k_clean=5, k_best=kb)) + '\n')
        o = analyze(d, k_star=50)
        for p in d.glob('*'):
            os.remove(p)
        os.rmdir(d)
        return o['models']['m.pt']['success_vs_prereg'], o['models']['m.pt']['frac_reach_kstar']
    a, fa = run([80, 90, 120, 200, 300, 70, 10, 20, 5, 8])   # 6/10 ≥ 50 → ĐẠT
    b, fb = run([10, 20, 30, 40, 45, 60, 8, 12, 5, 9])        # 1/10 ≥ 50 → CHƯA
    ok = a is True and b is False and abs(fa - 0.6) < 1e-9 and abs(fb - 0.1) < 1e-9
    print('selftest frac:', fa, fb, 'success:', a, b)
    print('SELFTEST_M3', 'PASS' if ok else 'FAIL')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir')
    ap.add_argument('--out')
    ap.add_argument('--k-star', type=int, default=50)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    out = analyze(a.run_dir, k_star=a.k_star)
    path = Path(a.out) if a.out else Path(a.run_dir) / 'summary.json'
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    print(path)
    print('VERDICT:', out['verdict'])


if __name__ == '__main__':
    main()
