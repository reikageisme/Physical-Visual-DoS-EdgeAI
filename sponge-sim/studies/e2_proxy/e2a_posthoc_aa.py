"""E2a POST-HOC (nghĩ ra SAU khi thấy summary E2a; xem DEVIATIONS.md). Thăm dò, chỉ đọc.

Đối chứng A/A cho 'attack_induced_share': cùng công thức g nhưng thay case tấn công bằng clean seed khác:
g_AA[i, s] = (nms_clean[i, s+1 mod 10] - nms_clean[i, s]) / (fwd_clean[i, s] + post_clean[i, s]).
Cùng bootstrap hai tầng. Thêm thống kê bền (trung vị) cho cả tấn công lẫn A/A.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from e2a_historical import ATTACKS, SEED, load  # noqa: E402
from sponge_sim import stats  # noqa: E402


def main():
    run_dir = Path(sys.argv[1])
    d, _ = load()
    c = d['clean']
    denom = c['fwd'] + c['post']
    out = dict(label='POST-HOC thăm dò, dữ liệu lịch sử', run_dir=str(run_dir))
    g_aa = (np.roll(c['nms'], -1, axis=1) - c['nms']) / denom
    rows = {'AA_clean_shift1': g_aa}
    for k in ATTACKS:
        rows[k] = (d[k]['nms'] - c['nms']) / denom
    for k, g in rows.items():
        lo, hi = stats.two_level_bootstrap(g, SEED)
        mlo, mhi = stats.two_level_bootstrap(g, SEED, stat=np.nanmedian)
        out[k] = dict(mean=float(g.mean()), ci95_mean=[float(lo), float(hi)],
                      median=float(np.median(g)), ci95_median=[float(mlo), float(mhi)])
    (run_dir / 'posthoc_aa.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    for k in rows:
        v = out[k]
        print(k, round(v['mean'], 4), [round(x, 4) for x in v['ci95_mean']], round(v['median'], 5),
              [round(x, 5) for x in v['ci95_median']])


if __name__ == '__main__':
    main()
