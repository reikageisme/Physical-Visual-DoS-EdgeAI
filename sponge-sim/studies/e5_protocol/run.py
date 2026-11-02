"""E5 — giao thức đo. Xem prereg/PREREG_E5.md.

Điều phối: `run.py --run-dir DIR [--smoke]` chạy mỗi profile trong một subprocess riêng
(`--worker PROFILE`), thứ tự profile theo hoán vị có seed. Worker ghim affinity TRƯỚC khi torch tạo
thread pool, xác minh bằng tải, rồi đo theo khối; ghi raw JSONL từng phép đo.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))

IMAGE_SEED = 20261021
IMAGE_IDS = [139, 785, 802, 872, 1000, 1503, 2157, 2299, 2431, 2587, 3501, 4395, 4795, 5037, 5477, 5503, 5586,
             5600, 6040, 6460, 6471, 7281, 7816, 7888, 8629, 8899, 10707, 11149, 11511, 12280, 12670, 13348, 13659,
             13774, 14007, 14888, 15746, 15956, 16249, 17029, 17115, 18519, 18833, 19786, 19924, 20059, 20571, 21167,
             21604, 21879, 22705, 24243, 25393, 28809, 28993, 30494, 31093, 31248, 31296, 32081, 32570, 32610, 32887,
             33368, 33759, 34071, 34257, 34873, 35197, 35279, 35963, 36494, 37689, 37751, 38070, 38118, 39551, 39769,
             40757, 41872, 42070, 42296, 42528, 42889, 44652, 46252, 46804, 46872, 47112, 49759, 49810, 50326, 50679,
             51976, 52412, 52413, 53529, 55072, 55299, 55528]
CONFIG = dict(
    experiment='E5', profiles=['P1', 'E1', 'E4'], profile_order_seed=5021, image_seed=IMAGE_SEED,
    n_images=100, repeats=10, warmup_per_image=1, block_size=10, labels=['A', 'B'], label_order_seed=5022,
    tau=.01, cap=None, iou=.45, interop_threads=1, intra_op_threads='= |allowed_cpus|',
    stages=['pre', 'forward', 'decode', 'tau', 'nms_torch_pure', 'nms_tv_nms', 'total'],
    total_definition='load_image -> forward -> decode -> tau -> torch_pure NMS, đo trực tiếp một lần gọi',
    reference='matmul 256x256 x20, trung vị 15 lần, đầu mỗi khối + cuối phiên; |Δ|>3% so với khối đầu -> cờ',
)
SMOKE = dict(n_images=4, repeats=2, block_size=2)


def worker(profile: str, run_dir: Path, cfg: dict):
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[profile]
    readback = hw.pin(cpus)                      # ghim TRƯỚC khi torch tạo thread pool
    import torch
    torch.set_num_interop_threads(cfg['interop_threads'])
    verify = hw.verify_pin(cpus)                 # dùng 2×|cpus| thread để thử vượt tập được phép
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    from sponge_sim.cohorts import REPO
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(REPO / 'yolov8n.pt')).model.eval().float()
    ids = IMAGE_IDS[:cfg['n_images']]
    writer = timing.JsonlWriter(run_dir / f'raw_{profile}.jsonl')
    writer.write(dict(kind='pin', profile=profile, allowed=cpus, readback=readback, verify=verify,
                      torch_threads=torch.get_num_threads(), interop=torch.get_num_interop_threads()))
    tp, tv = BACKENDS['torch_pure'], BACKENDS['tv_nms']
    rng = random.Random(cfg['label_order_seed'] + cpus[0])

    def total(p):
        im = pv.load_image(p)
        r = model(im)
        r = r[0] if isinstance(r, (tuple, list)) else r
        act, _ = pv.tau_filter(pv.decode(r[0]), cfg['tau'], cfg['cap'])
        return tp.run((act[:, :4], act[:, 4]), cfg['iou'])

    def one_rep(path):
        out = {}
        def t(name, fn, *a):
            r = timing.timed(fn, setup=lambda: a, repeats=1)[0]
            out[name] = r['ns']
            return r['result']
        with torch.no_grad():
            image = t('pre', pv.load_image, path)
            raw = t('forward', model, image)
            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
            decoded = t('decode', pv.decode, raw[0])
            active, n_tau = t('tau', pv.tau_filter, decoded, cfg['tau'], cfg['cap'])
            boxes, scores, classes = active[:, :4], active[:, 4], active[:, 5]
            t('nms_torch_pure', tp.run, tp.prepare(boxes, scores, classes), cfg['iou'])
            t('nms_tv_nms', tv.run, tv.prepare(boxes, scores, classes), cfg['iou'])
            t('total', total, path)
        return out, n_tau

    ref0 = timing.reference_ms()
    blocks = [ids[i:i + cfg['block_size']] for i in range(0, len(ids), cfg['block_size'])]
    for b, block in enumerate(blocks):
        writer.write(dict(kind='reference', profile=profile, block=b, **timing.drift(timing.reference_ms(), ref0),
                          wall_utc=time.time()))
        for image_id in block:
            path = data / 'images' / names[image_id]
            for _ in range(cfg['warmup_per_image']):
                one_rep(path)
            # A/A: nhãn A/B là CÙNG một điều kiện; thứ tự ABBA hoặc BAAB lặp lại, chọn ngẫu nhiên theo ảnh
            seq = ['A', 'B', 'B', 'A'] if rng.random() < .5 else ['B', 'A', 'A', 'B']
            seq = (seq * (2 * cfg['repeats'] // 4 + 1))[:2 * cfg['repeats']]
            for k, label in enumerate(seq):
                stages, n_tau = one_rep(path)
                writer.write(dict(kind='measure', profile=profile, block=b, image_id=image_id, label=label,
                                  seq_index=k, n_tau=n_tau, stages_ns=stages, wall_utc=time.time()))
    writer.write(dict(kind='reference', profile=profile, block='end', **timing.drift(timing.reference_ms(), ref0),
                      wall_utc=time.time()))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker')
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = {**CONFIG, **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    if a.worker:
        worker(a.worker, run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)  # không ghi đè run cũ
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg))
    order = cfg['profiles'][:]
    random.Random(cfg['profile_order_seed']).shuffle(order)
    for prof in order:
        t0 = time.time()
        cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker', prof]
        if a.smoke:
            cmd.append('--smoke')
        subprocess.run(cmd, check=True)
        print(f'[E5] profile {prof} xong sau {time.time() - t0:.1f}s', flush=True)
    print('[E5] DONE', flush=True)


if __name__ == '__main__':
    main()
