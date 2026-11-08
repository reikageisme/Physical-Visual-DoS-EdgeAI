"""Mô tả (post hoc, không đo lại): trung vị thời gian NMS vi mô và toàn trình của E10 theo nhóm nhánh (sạch / từng phương pháp) và backend.
Cung cấp nguồn cho các số mô tả trong văn bản (ms NMS và toàn trình trước/sau tấn công).

    python stage_medians.py RUN_DIR   -> RUN_DIR/stage_medians_posthoc.json
Mỗi ô (ảnh, nhánh, seed, backend) tóm bằng trung vị các lần lặp; sau đó lấy trung vị qua các ô của nhóm.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    run_dir = Path(sys.argv[1])
    cell = {'micro': defaultdict(list), 'pipeline': defaultdict(list)}
    for line in next(run_dir.glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['kind'] in cell:
            key = (r['image_id'], r['arm'], r['seed'], r['backend'])
            cell[r['kind']][key].append((r['nms_ns'] if r['kind'] == 'micro' else r['total_ns']) / 1e6)
    out = dict(analysis='E10 stage medians (post hoc, descriptive)', run_dir=str(run_dir), nms_ms={}, total_ms={})
    for kind, name in (('micro', 'nms_ms'), ('pipeline', 'total_ms')):
        groups = defaultdict(list)
        for (i, arm, s, be), v in cell[kind].items():
            grp = 'clean' if arm in ('clean_a', 'clean_b') else arm
            groups[f'{grp}|{be}'].append(float(np.median(v)))
        out[name] = {k: dict(median=float(np.median(v)), n_cells=len(v)) for k, v in sorted(groups.items())}
    (run_dir / 'stage_medians_posthoc.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for name in ('nms_ms', 'total_ms'):
        print(name, {k: round(v['median'], 3) for k, v in out[name].items()})


if __name__ == '__main__':
    main()
