"""Phân rã mức tăng toàn trình của E10 theo công đoạn (post hoc, chỉ đọc bản ghi thô, không đo mới).

    python stage_decomp.py RUN_DIR   -> RUN_DIR/stage_decomp_posthoc.json

E10 chỉ bấm giờ riêng công đoạn NMS (vi mô, tau 0,25) và toàn trình (forward -> decode -> lọc tau -> prepare -> NMS).
Với mỗi đơn vị (ảnh, phương pháp, seed, backend): dTotal = toàn trình tấn công - trung bình hai nhánh sạch cùng ảnh;
dNMS = NMS tấn công - trung bình NMS hai nhánh sạch; phần còn lại = dTotal - dNMS (forward + decode + lọc tau + prepare;
bản ghi thô KHÔNG tách được các công đoạn này với nhau). Mỗi ô tóm bằng trung vị các lần lặp; KTC bootstrap hai tầng ảnh x seed.
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


def main():
    run_dir = Path(sys.argv[1])
    cell = {'micro': defaultdict(list), 'pipeline': defaultdict(list)}
    for line in next(run_dir.glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['kind'] in cell:
            cell[r['kind']][(r['image_id'], r['arm'], r['seed'], r['backend'])].append((r['nms_ns'] if r['kind'] == 'micro' else r['total_ns']) / 1e6)
    med = {k: {kk: float(np.median(v)) for kk, v in d.items()} for k, d in cell.items()}
    ids = sorted({k[0] for k in med['pipeline']})
    methods = sorted({k[1] for k in med['pipeline'] if k[1] not in ('clean_a', 'clean_b')})
    backends = sorted({k[3] for k in med['pipeline']})
    out = dict(analysis='E10 stage decomposition (post hoc)', run_dir=str(run_dir), bootstrap_seed=SEED, cells={})
    for m in methods:
        seeds = sorted({k[2] for k in med['pipeline'] if k[1] == m})
        for be in backends:
            cl = lambda kind, i: np.mean([med[kind][(i, 'clean_a', None, be)], med[kind][(i, 'clean_b', None, be)]])  # noqa: E731
            dT = np.array([[med['pipeline'][(i, m, s, be)] - cl('pipeline', i) for s in seeds] for i in ids])
            dN = np.array([[med['micro'][(i, m, s, be)] - cl('micro', i) for s in seeds] for i in ids])
            rest = dT - dN
            ci = lambda M: [float(x) for x in stats.two_level_bootstrap(M, SEED)]  # noqa: E731
            out['cells'][f'{m}|{be}'] = dict(n_images=len(ids), seeds=seeds, d_total_ms=float(dT.mean()), d_total_ci95=ci(dT),
                                             d_nms_ms=float(dN.mean()), d_nms_ci95=ci(dN), d_rest_ms=float(rest.mean()), d_rest_ci95=ci(rest),
                                             rest_share_of_d_total=float(rest.mean() / dT.mean()))
    (run_dir / 'stage_decomp_posthoc.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for k, v in out['cells'].items():
        print(k, 'dT %.2f %s' % (v['d_total_ms'], [round(x, 2) for x in v['d_total_ci95']]), 'dNMS %.2f' % v['d_nms_ms'],
              'rest %.2f %s share %.2f' % (v['d_rest_ms'], [round(x, 2) for x in v['d_rest_ci95']], v['rest_share_of_d_total']))


if __name__ == '__main__':
    main()
