"""Mô hình kiểu Amdahl cho tỷ số toàn trình và ngưỡng N*, đối chiếu với phép tiêm toàn pipeline (E9). PHÂN TÍCH POST HOC
(E9 đã được xem trước khi viết mô hình); phép kiểm tiến cứu của cùng mô hình nằm trong PREREG_E10.

    python analyze.py E1_SUMMARY E9_RUN_DIR OUT_JSON

Mô hình (không có tham số khớp):
    R_pred(N) = 1 + [t_NMS(N) - t_NMS(0)] / t_total(0)
  t_NMS(N): trung vị NMS của phép tiêm tổng hợp E1 (cùng tập ứng viên, cùng lưới anchor, cùng backend/hình học/cap; nc = 80);
  t_total(0): trung vị toàn trình của ảnh sạch (N = 0) đo trong chính phiên cần dự đoán, theo ảnh rồi trung bình nhân.
  N*_pred = N nhỏ nhất trong lưới có R_pred > 1 + MDE.
Kèm: sàn nhiễu trong phiên của E9 (từ 3 ô N = 0 lặp lại mỗi ảnh) và N* khi bỏ các khối có điểm tham chiếu bị cờ.
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
BACKENDS = ['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native']
GEOMS = ['disjoint', 'dense_cluster', 'mixed']
ANCHORS = {320: 2100, 640: 8400}


def e1_nms(curves, anchors, be, geom, cap, n):
    if n == 0:
        geom, cap = 'disjoint', None
    k = f'P1_cpu2|{anchors}|{be}|{geom}|80|{cap}|{n}'
    return curves[k]['median_ms'] if k in curves else None


def load_e9(run_dir):
    cell, block_of, refs = defaultdict(list), {}, []
    for line in open(next(Path(run_dir).glob('raw_*.jsonl')), encoding='utf-8'):
        r = json.loads(line)
        if r['kind'] == 'cell':
            cell[(r['canvas'], r['image_id'], r['n'], r['geometry'], r['cap'], r['backend'], r.get('copy', 0))].append(r['total_ns'] / 1e6)
            block_of[(r['canvas'], r['image_id'])] = r['block']
        elif r['kind'] == 'reference':
            refs.append(r)
    med = {k: float(np.median(v)) for k, v in cell.items()}
    flagged = {r['block'] for r in refs if r['flagged'] and r['block'] != 'end'}
    return med, block_of, flagged


def ratios(med, cv, be, geom, cap, n, images):
    out = []
    for i in images:
        base = [med[(cv, i, 0, 'disjoint', None, be, c)] for c in range(3) if (cv, i, 0, 'disjoint', None, be, c) in med]
        k = (cv, i, n, geom, cap, be, 0)
        if base and k in med:
            out.append(np.log(med[k] / np.median(base)))
    return out


def main():
    curves = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))['curves']
    med, block_of, flagged = load_e9(sys.argv[2])
    aa = {320: 0.5523331275826271, 640: 1.8822492535464308}  # MDE đã đăng ký cho E9 (PREREG_E9)
    out = dict(analysis='Amdahl model vs E9 (post hoc)', mde_registered_pct=aa, in_session_mde_pct={}, cells={},
               n_star={}, n_star_unflagged_blocks={}, flagged_blocks=sorted(flagged, key=str))
    for cv in (320, 640):
        images = sorted({k[1] for k in med if k[0] == cv})
        keep = [i for i in images if block_of[(cv, i)] not in flagged]
        ns = sorted({k[2] for k in med if k[0] == cv and k[2] > 0})
        for be in BACKENDS:
            # sàn nhiễu trong phiên: bản sao 1 so với bản sao 0 của ô N = 0
            d = [np.log(med[(cv, i, 0, 'disjoint', None, be, 1)] / med[(cv, i, 0, 'disjoint', None, be, 0)]) for i in images
                 if (cv, i, 0, 'disjoint', None, be, 1) in med and (cv, i, 0, 'disjoint', None, be, 0) in med]
            mde_in = 100 * stats.mde_paired(float(np.std(d, ddof=1)), len(d)) if len(d) > 2 else None
            out['in_session_mde_pct'][f'{cv}|{be}'] = mde_in
            t0 = float(np.exp(np.mean([np.log(np.median([med[(cv, i, 0, 'disjoint', None, be, c)] for c in range(3)
                                                          if (cv, i, 0, 'disjoint', None, be, c) in med])) for i in images])))
            for geom in GEOMS:
                for cap in (None, 300):
                    key = f'{cv}|{be}|{geom}|cap{cap}'
                    rows, ns_pred, ns_meas, ns_unfl = [], None, None, None
                    for n in ns:
                        if cap is not None and n <= cap:
                            continue
                        a, b0 = e1_nms(curves, ANCHORS[cv], be, geom, cap, n), e1_nms(curves, ANCHORS[cv], be, geom, cap, 0)
                        if a is None or b0 is None:
                            continue
                        r_pred = 1 + (a - b0) / t0
                        logs = ratios(med, cv, be, geom, cap, n, images)
                        lo, hi = stats.bootstrap_ci(logs, SEED)
                        r_meas = float(np.exp(np.mean(logs)))
                        logs_u = ratios(med, cv, be, geom, cap, n, keep)
                        lo_u = stats.bootstrap_ci(logs_u, SEED)[0] if len(logs_u) > 2 else None
                        rows.append(dict(n=n, r_pred=r_pred, r_meas=r_meas, ci95=[float(np.exp(lo)), float(np.exp(hi))],
                                         abs_log_err=abs(np.log(r_meas / r_pred))))
                        thr = 1 + aa[cv] / 100
                        if ns_pred is None and r_pred > thr:
                            ns_pred = n
                        if ns_meas is None and np.exp(lo) > thr:
                            ns_meas = n
                        if ns_unfl is None and lo_u is not None and np.exp(lo_u) > thr:
                            ns_unfl = n
                    if rows:
                        out['cells'][key] = dict(t_total_clean_ms=t0, rows=rows)
                        out['n_star'][key] = dict(predicted=ns_pred or f'>{rows[-1]["n"]}', measured=ns_meas or f'>{rows[-1]["n"]}')
                        out['n_star_unflagged_blocks'][key] = dict(n_images=len(keep), measured=ns_unfl or f'>{rows[-1]["n"]}')
    errs = [r['abs_log_err'] for c in out['cells'].values() for r in c['rows']]
    agree = [v['predicted'] == v['measured'] for v in out['n_star'].values()]
    out['summary'] = dict(n_cells=len(errs), median_abs_log_err=float(np.median(errs)), p90_abs_log_err=float(np.quantile(errs, .9)),
                          n_star_exact_agreement=f'{sum(agree)}/{len(agree)}')
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(json.dumps(out['summary']))


if __name__ == '__main__':
    main()
