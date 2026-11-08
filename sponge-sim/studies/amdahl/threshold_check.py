"""Mục "ý dương" theo góp ý GVHD (03/11/2026). POST HOC, chỉ dùng dữ liệu đã đo.

Công thức: R ≈ 1 + Δt_NMS / t_total  =>  vượt MDE khi Δt_NMS > MDE × t_total.
1) Ngưỡng Δt_NMS (ms) ở 320 px và 640 px cho toàn trình đo trên tensor đã chuẩn bị (E6: 320 px; E6-A1 lần 2: 640 px), từ t_total của ảnh sạch.
2) Baseline gradient (B4): Δt_NMS đo (micro, τ 0,01) so với ngưỡng; tỷ số toàn trình dự đoán 1 + Δt_NMS/t_total so với đo.
3) Hiệu chỉnh: phép đo toàn trình của E6 tính bước chèn patch cho nhánh có patch (không tính cho ảnh sạch);
   R_corr = t_patched / (t_clean + t_insert), t_insert đo riêng (results/diag_insert).
    python threshold_check.py OUT_JSON
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import stats  # noqa: E402

SEED = 20261021


def load(raw, tau=0.01):
    micro, pipe = defaultdict(list), defaultdict(list)
    for line in open(raw, encoding='utf-8'):
        r = json.loads(line)
        if r['kind'] == 'micro' and r['tau'] == tau:
            micro[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['nms_ns'] / 1e6)
        elif r['kind'] == 'pipeline':
            pipe[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['total_ns'] / 1e6)
    return ({k: float(np.median(v)) for k, v in micro.items()}, {k: float(np.median(v)) for k, v in pipe.items()})


def arm_stats(m, p, arm, clean, t_ins, be):
    ids = sorted({k[0] for k in p if k[1] == arm})
    seeds = sorted({k[2] for k in p if k[1] == arm})
    dN = np.array([[m[(i, arm, s, be)] - m[(i, clean, None, be)] for s in seeds] for i in ids])
    tc = np.array([[p[(i, clean, None, be)] for _ in seeds] for i in ids])
    tp = np.array([[p[(i, arm, s, be)] for s in seeds] for i in ids])
    R, Rc, Rpred = tp / tc, tp / (tc + t_ins), 1 + dN / tc
    ci = lambda M: [float(x) for x in stats.two_level_bootstrap(M, SEED)]
    return dict(n_images=len(ids), delta_nms_ms_mean=float(dN.mean()), delta_nms_ms_ci95=ci(dN), t_clean_ms_median=float(np.median(tc[:, 0])),
                r_meas=float(R.mean()), r_meas_ci95=ci(R), r_corrected=float(Rc.mean()), r_corrected_ci95=ci(Rc),
                r_pred=float(Rpred.mean()), r_pred_ci95=ci(Rpred))


def main():
    res = SIM / 'results'
    ins = json.loads((res / 'diag_insert/insert_cost_20261103.json').read_text(encoding='utf-8'))['cases']
    mde320 = json.loads((res / 'e5/AA_P1cpu2_20261101/summary.json').read_text(encoding='utf-8'))['profiles']['P1_cpu2']['stages']['total']['mde_ratio_pct']
    mde640 = json.loads((res / 'e5/AA640_P1cpu2_20261102b/summary.json').read_text(encoding='utf-8'))['profiles']['P1_cpu2']['stages']['total']['mde_ratio_pct']
    out = dict(analysis='threshold and B4 check (post hoc)', mde_total_pct={'320': mde320, '640': mde640}, insert_ms={k: v['median_ms'] for k, v in ins.items()},
               threshold_ms={}, arms={})
    m320, p320 = load(next((res / 'e6/E6_pooled_baselines_100img').glob('raw_*.jsonl')))
    m320c, p320c = load(next((res / 'e6/E6_P1cpu2_20261102').glob('raw_*.jsonl')))
    m640, p640 = load(next((res / 'e6/E6A1_P1cpu2_20261102b').glob('raw_*.jsonl')))
    for be in ('tv_nms', 'torch_pure'):
        t320 = float(np.median([v for k, v in p320c.items() if k[1] == 'clean' and k[3] == be]))
        t640 = float(np.median([v for k, v in p640.items() if k[1] == 'clean640' and k[3] == be]))
        out['threshold_ms'][be] = {'320': dict(t_total_clean_ms=t320, threshold_ms=mde320 / 100 * t320),
                                   '640': dict(t_total_clean_ms=t640, threshold_ms=mde640 / 100 * t640)}
        for arm in ('B1', 'B2', 'B3', 'B4'):
            out['arms'][f'320|{arm}|{be}'] = arm_stats(m320, p320, arm, 'clean', ins['320px_patch64']['median_ms'], be)
        for arm in ('C1', 'C3'):
            out['arms'][f'320|{arm}|{be}'] = arm_stats(m320c, p320c, arm, 'clean', ins['320px_patch64']['median_ms'], be)
        out['arms'][f'640|C2|{be}'] = arm_stats(m640, p640, 'C2', 'clean640', ins['640px_patch128']['median_ms'], be)
        out['arms'][f'640|C4|{be}'] = arm_stats(m640, p640, 'C4', 'clean640', ins['640px_patch64']['median_ms'], be)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out['threshold_ms']))
    for k, v in out['arms'].items():
        print(k, 'dNMS %.4f ms' % v['delta_nms_ms_mean'], 'meas %.4f' % v['r_meas'], 'corr %.4f %s' % (v['r_corrected'], [round(x, 4) for x in v['r_corrected_ci95']]), 'pred %.5f' % v['r_pred'])


if __name__ == '__main__':
    main()
