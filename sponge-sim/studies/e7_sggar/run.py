"""E7: chạy lại thuật toán SG-GA ĐÃ CÓ với tập tối ưu 100 ảnh (thay vì 4), ở 320 px và 640 px, ngưỡng mục tiêu 0,01 và 0,25.
Xem prereg/PREREG_E7.md. KHÔNG phải thuật toán mới: gọi attacks_provenance.optimize (bản copy có header nguồn + SHA256)
của experiments/extended_validation.py:113-222; chỉ khác cấu hình (số ảnh, canvas, ngưỡng, thiết bị).

Một lệnh (trên máy GPU, từ thư mục gói):
    python sponge-sim/studies/e7_sggar/run.py --out runs/e7 --device cuda
Dừng/chạy lại an toàn: mỗi (điều kiện, seed) có resume.pt; điều kiện đã xong (complete.json) được bỏ qua.
Chỉ ghi vào --out. Không dùng mạng. Không đo độ trễ ở máy này (độ trễ đo trên laptop/điện thoại theo giao thức E5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
ROOT = SIM.parent                      # gốc gói (có core/, yolov8n.pt) hoặc gốc repo
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(ROOT))

SEED_DATA = 20261101
TRAIN_FILES = [
    '000000000009.jpg', '000000000030.jpg', '000000000034.jpg', '000000000042.jpg', '000000000049.jpg',
    '000000000061.jpg', '000000000064.jpg', '000000000071.jpg', '000000000072.jpg', '000000000073.jpg',
    '000000000077.jpg', '000000000078.jpg', '000000000086.jpg', '000000000089.jpg', '000000000092.jpg',
    '000000000109.jpg', '000000000110.jpg', '000000000127.jpg', '000000000136.jpg', '000000000138.jpg',
    '000000000143.jpg', '000000000144.jpg', '000000000149.jpg', '000000000151.jpg', '000000000164.jpg',
    '000000000165.jpg', '000000000192.jpg', '000000000196.jpg', '000000000201.jpg', '000000000208.jpg',
    '000000000241.jpg', '000000000247.jpg', '000000000250.jpg', '000000000257.jpg', '000000000260.jpg',
    '000000000263.jpg', '000000000283.jpg', '000000000307.jpg', '000000000308.jpg', '000000000309.jpg',
    '000000000312.jpg', '000000000315.jpg', '000000000321.jpg', '000000000322.jpg', '000000000328.jpg',
    '000000000332.jpg', '000000000338.jpg', '000000000349.jpg', '000000000357.jpg', '000000000359.jpg',
    '000000000360.jpg', '000000000368.jpg', '000000000370.jpg', '000000000382.jpg', '000000000384.jpg',
    '000000000395.jpg', '000000000397.jpg', '000000000404.jpg', '000000000415.jpg', '000000000419.jpg',
    '000000000428.jpg', '000000000436.jpg', '000000000438.jpg', '000000000443.jpg', '000000000446.jpg',
    '000000000450.jpg', '000000000459.jpg', '000000000471.jpg', '000000000472.jpg', '000000000474.jpg',
    '000000000486.jpg', '000000000488.jpg', '000000000491.jpg', '000000000502.jpg', '000000000508.jpg',
    '000000000510.jpg', '000000000514.jpg', '000000000520.jpg', '000000000529.jpg', '000000000531.jpg',
    '000000000532.jpg', '000000000536.jpg', '000000000542.jpg', '000000000544.jpg', '000000000560.jpg',
    '000000000562.jpg', '000000000569.jpg', '000000000572.jpg', '000000000575.jpg', '000000000581.jpg',
    '000000000584.jpg', '000000000595.jpg', '000000000597.jpg', '000000000605.jpg', '000000000625.jpg',
    '000000000626.jpg', '000000000634.jpg', '000000000636.jpg', '000000000641.jpg', '000000000650.jpg',
]
BUDGETS = [10000, 25000, 62500]        # số lần forward ảnh; 62500 = 625 ứng viên x 100 ảnh (tương đương q2500 của 4 ảnh)
CONDITIONS = {
    # tên: (canvas, kích thước patch, ngưỡng mục tiêu, phương pháp, kiểu vị trí)
    # Diện tích patch: 64^2/320^2 = 128^2/640^2 = 4,0%; 64^2/640^2 = 1,0%.
    # Nhóm chính (SG-GA = thuật toán đã có, đổi cấu hình):
    'C1_320_s64_t01': (320, 64, 0.01, 'sg_ga', 'saliency'),
    'C2_640_s128_t01': (640, 128, 0.01, 'sg_ga', 'saliency'),    # diện tích bằng C1
    'C3_320_s64_t25': (320, 64, 0.25, 'sg_ga', 'saliency'),      # tối ưu với ngưỡng 0,25
    # Baseline đã có sẵn trong repo, cùng quy mô 100 ảnh (không phải thuật toán mới):
    'B1_ga_center': (320, 64, 0.01, 'ga_center', 'center'),
    'B2_ga_random_location': (320, 64, 0.01, 'ga_random_location', 'random'),
    'B3_random_search': (320, 64, 0.01, 'random_search', 'saliency'),
    'B4_gradient_tanh': (320, 64, 0.01, 'gradient_tanh', 'saliency'),
    # Phụ:
    'C4_640_s64_t01': (640, 64, 0.01, 'sg_ga', 'saliency'),      # kích thước tuyệt đối bằng C1 (diện tích 1%)
}
DEFAULT_CONDITIONS = ['C1_320_s64_t01', 'C2_640_s128_t01', 'C3_320_s64_t25', 'B1_ga_center', 'B2_ga_random_location',
                      'B3_random_search', 'B4_gradient_tanh', 'C4_640_s64_t01']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def find_images_dir(arg):
    for c in ([Path(arg)] if arg else []) + [ROOT / 'data' / 'train100',
                                              ROOT / 'outputs' / 'datasets' / 'coco128' / 'images' / 'train2017']:
        if c.exists() and all((c / f).exists() for f in TRAIN_FILES):
            return c
    raise SystemExit('Không tìm thấy thư mục ảnh tối ưu chứa đủ 100 file (data/train100 hoặc coco128/train2017)')


def parse_seeds(text):
    if '-' in text:
        a, b = text.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--conditions', default=','.join(DEFAULT_CONDITIONS))
    ap.add_argument('--images-dir', default='')
    ap.add_argument('--weights', default='')
    ap.add_argument('--threads', type=int, default=0, help='CPU: số luồng torch (0 = mặc định)')
    ap.add_argument('--smoke', action='store_true', help='2 ảnh, ngân sách tí hon, 1 seed: chỉ kiểm lỗi')
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('Yêu cầu --device cuda nhưng không có CUDA')
    from core.victim_model import VictimModel
    from sponge_sim import attacks_provenance as ap_

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = Path(a.weights) if a.weights else ROOT / 'yolov8n.pt'
    images_dir = find_images_dir(a.images_dir)
    files = TRAIN_FILES[:2] if a.smoke else TRAIN_FILES
    budgets = [20, 40] if a.smoke else BUDGETS
    seeds = [0] if a.smoke else parse_seeds(a.seeds)
    names = [c for c in a.conditions.split(',') if c]

    victim = VictimModel(model_path=str(weights), device=a.device)
    # Ghi đè hai thuộc tính INSTANCE (không sửa file gốc): chạy float32 như lượt CPU gốc; xem attacks_provenance.py, ghi chú 3.
    victim.use_half = False
    victim.model = victim.model_float
    victim.model.eval()
    for p in victim.model.parameters():
        p.requires_grad_(False)

    manifest = dict(
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        experiment='E7', smoke=a.smoke, device=a.device,
        gpu=(torch.cuda.get_device_name(0) if a.device == 'cuda' else None),
        torch=torch.__version__, python=sys.version, platform=platform.platform(),
        git_head=git('rev-parse', 'HEAD'), git_tags_at_head=git('tag', '--points-at', 'HEAD').split(),
        weights_sha256=sha256_file(weights),
        script_sha256=sha256_file(Path(__file__)),
        attacks_provenance_sha256=sha256_file(SIM / 'src' / 'sponge_sim' / 'attacks_provenance.py'),
        victim_model_sha256=sha256_file(ROOT / 'core' / 'victim_model.py'),
        train_images=[dict(file=f, sha256=sha256_file(images_dir / f)) for f in files],
        data_selection=dict(seed=SEED_DATA, rule='random.Random(seed).sample(sorted(coco128/train2017 file names), 100)'),
        conditions={n: dict(zip(('canvas', 'size', 'threshold', 'method', 'location'), CONDITIONS[n])) for n in names},
        budgets=budgets, seeds=seeds,
        bundle_source=((ROOT / 'BUNDLE_SOURCE.txt').read_text(encoding='utf-8') if (ROOT / 'BUNDLE_SOURCE.txt').exists() else None),
        note_rng='Trên GPU, RNG CUDA khác RNG CPU: patch không trùng byte với bản CPU dù cùng thuật toán và seed.',
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')

    cache = {}
    t_start = time.perf_counter()
    done_units, total_units = 0, len(seeds) * len(names)
    for seed in seeds:                       # seed-major: dừng giữa chừng vẫn đủ mọi điều kiện cho các seed đã xong
        for name in names:
            canvas, size, thr, method, location = CONDITIONS[name]
            if canvas not in cache:
                cache[canvas] = [ap_.load_image_canvas(images_dir / f, canvas) for f in files]
            spec = dict(name=name, method=method, location=location, size=size, eot='full', draws=1,
                        budgets=budgets, canvas=canvas, threshold=thr, device=a.device, n_train=len(files))
            run_dir = out / name / f'seed{seed}'
            t0 = time.perf_counter()
            ap_.optimize(victim, cache[canvas], spec, seed, run_dir)
            done_units += 1
            el = time.perf_counter() - t_start
            print(f'[E7] {name} seed {seed} xong trong {time.perf_counter() - t0:.0f}s; '
                  f'{done_units}/{total_units}; ước còn {el / done_units * (total_units - done_units) / 60:.0f} phút',
                  flush=True)
    rows = []
    for ck in sorted(out.glob('*/seed*/checkpoint_q*.json')):
        d = json.loads(ck.read_text(encoding='utf-8'))
        rows.append(dict(path=str(ck.relative_to(out)).replace('\\', '/'), seed=d['seed'], queries=d['queries'],
                         canvas=d['canvas'], patch_sha256=d['patch_sha256'], locations=d['locations']))
    (out / 'patches_index.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    print('[E7] DONE', len(rows), 'checkpoint', flush=True)


if __name__ == '__main__':
    main()
