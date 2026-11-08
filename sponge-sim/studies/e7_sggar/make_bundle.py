"""Đóng gói độc lập cho máy GPU thuê: chỉ gồm mã đã audit + dữ liệu cần để chạy E7 (không có outputs/**, không có kết quả cũ).

    python sponge-sim/studies/e7_sggar/make_bundle.py --out e7_gpu_bundle.zip

Nội dung zip (cấu trúc phẳng, chạy từ gốc zip):
    sponge-sim/src/sponge_sim/{__init__,attacks_provenance}.py       mã sao chép có header nguồn + SHA256
    sponge-sim/studies/e7_sggar/run.py
    core/victim_model.py                                              mã gốc, KHÔNG sửa (byte-for-byte)
    yolov8n.pt                                                        trọng số (SHA256 ghi trong BUNDLE_SHA256.txt)
    data/train100/*.jpg                                               100 ảnh tối ưu (coco128/train2017)
    BUNDLE_SHA256.txt                                                 SHA256 mọi file trong gói
    RUN_ON_GPU.txt                                                    đúng một lệnh để chạy
Phụ thuộc trên máy GPU: python>=3.10, torch (CUDA), torchvision, ultralytics, kornia, opencv-python, pillow, numpy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import zipfile
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
ROOT = SIM.parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    spec = importlib.util.spec_from_file_location('e7run', SIM / 'studies' / 'e7_sggar' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = {
        'sponge-sim/src/sponge_sim/__init__.py': SIM / 'src' / 'sponge_sim' / '__init__.py',
        'sponge-sim/src/sponge_sim/attacks_provenance.py': SIM / 'src' / 'sponge_sim' / 'attacks_provenance.py',
        'sponge-sim/studies/e7_sggar/run.py': SIM / 'studies' / 'e7_sggar' / 'run.py',
        'core/victim_model.py': ROOT / 'core' / 'victim_model.py',
        'yolov8n.pt': ROOT / 'yolov8n.pt',
    }
    img_dir = ROOT / 'outputs' / 'datasets' / 'coco128' / 'images' / 'train2017'
    for f in mod.TRAIN_FILES:
        files[f'data/train100/{f}'] = img_dir / f
    lines = [f'{sha(p)}  {name}' for name, p in sorted(files.items())]
    readme = ('Chạy trên máy GPU (từ thư mục đã giải nén zip):\n'
              '  python sponge-sim/studies/e7_sggar/run.py --out runs/e7 --device cuda\n'
              'Kiểm gói trước: sha256sum -c BUNDLE_SHA256.txt\n'
              'Dừng (Ctrl+C) và chạy lại cùng lệnh sẽ tiếp tục từ checkpoint. Kết quả: runs/e7/ (mang cả thư mục về laptop).\n'
              'Nếu thiếu giờ: dừng sau khi log báo xong seed 4 (đủ cho bước đo; seed 5-9 là bổ sung). Cắt điều kiện theo thứ tự C4, B3, B2, B1, B4 bằng --conditions.\n')
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, p in sorted(files.items()):
            z.write(p, name)
        z.writestr('BUNDLE_SHA256.txt', '\n'.join(lines) + '\n')
        z.writestr('RUN_ON_GPU.txt', readme)
        git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        source = [f'git_head={git("rev-parse", "HEAD")}',
                  f'tags_at_head={git("tag", "--points-at", "HEAD")}',
                  f'prereg_e7_sha256={sha(SIM / "prereg" / "PREREG_E7.md")}']
        z.writestr('BUNDLE_SOURCE.txt', '\n'.join(source) + '\n')
    print(out, out.stat().st_size, 'bytes;', len(files), 'file')


if __name__ == '__main__':
    main()
