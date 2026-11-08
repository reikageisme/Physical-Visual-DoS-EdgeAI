"""Phân tích lại (post hoc, chỉ đọc) số ứng viên trước NMS của X2 (outputs/x2_architecture_20261020) cho các YOLO.

    python studies/x2_counts/analyze.py OUT_JSON

Nguồn: experiments/x2_architecture_boundary.py (sha256 aebe1f5f5ed2932b…, khớp manifest X2), 320 px, tau 0,01, patch nguồn
là 5 patch SG-GA cũ (seed 0–4, ngân sách 2500, từ X1-500). Chỉ dùng các detector có boundary pre_nms_head (YOLO); Faster R-CNN,
RetinaNet, SSDlite đếm sau NMS nội bộ nên bị loại (CLAUDE.md). Không dùng trường nms_ms (đo chưa ghim lõi).
Đơn vị: cặp (ảnh, seed); tỷ số theo ảnh = patched / max(clean, 1); KTC bootstrap hai tầng ảnh × seed, 20.000 lần, seed 20261021.
"""
import json
import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
from sponge_sim import stats  # noqa: E402

SRC = SIM.parent / 'outputs/x2_architecture_20261020'
YOLO = ['yolov8n', 'yolov8s', 'yolo11n', 'yolo11s', 'yolov12n']
SEED = 20261021


def main():
    out = dict(analysis='X2 YOLO pre-NMS candidate counts (post hoc, read-only)', source=str(SRC.relative_to(SIM.parent)).replace('\\', '/'),
               manifest=json.loads((SRC / 'manifest.json').read_text(encoding='utf-8'))['script_sha256'], detectors={})
    for d in YOLO:
        r = json.loads((SRC / f'{d}.json').read_text(encoding='utf-8'))
        assert r['boundary'] == 'pre_nms_head'
        seeds = [f'seed{s}' for s in range(5)]
        clean = np.array([x['clean']['active_candidates'] for x in r['records']], float)
        pat = np.array([[x[s]['active_candidates'] for s in seeds] for x in r['records']], float)
        ratio = pat / np.maximum(clean, 1)[:, None]
        out['detectors'][d] = dict(images=r['images'], weights_sha256=r['weights_sha256'], clean_median=float(np.median(clean)),
                                   patched_median=float(np.median(pat)), ratio_mean=float(ratio.mean()),
                                   ratio_ci95=[float(v) for v in stats.two_level_bootstrap(ratio, SEED)],
                                   pooled_mean_ratio_reported=r['pooled_mean_ratio'])
    Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding='utf-8')
    for d, v in out['detectors'].items():
        print(d, v['images'], v['clean_median'], v['patched_median'], round(v['ratio_mean'], 4), [round(x, 4) for x in v['ratio_ci95']], round(v['pooled_mean_ratio_reported'], 4))


if __name__ == '__main__':
    main()
