"""E9 — tiêm ứng viên vào TOÀN pipeline (forward thật + τ + NMS), 4 backend, nhiều ảnh. Xem prereg/PREREG_E9.md (Phần A).

Điều phối: `run.py --run-dir DIR [--smoke]`; worker là subprocess riêng, ghim trước khi torch tạo thread (như E1).
Mỗi ô (độ phân giải, ảnh, N, hình học, cap): 4 backend xen kẽ theo khối Latin; mỗi lần lặp ghi
  total_ns  = forward(ảnh) + τ + NMS trong MỘT lần gọi (ứng viên tiêm thay đầu ra đã giải mã, như E1 phần B);
  tau_ns, nms_ns = đo riêng ở lần gọi kế tiếp, ngoài vùng total (prepare() ngoài vùng đo), để có tỷ phần NMS.
Tập ứng viên tiêm giống hệt E1 (cùng seed, cùng canvas mặc định của inject.make) để so được với đường cong E1.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))

BACKEND_NAMES = ['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native']
CONFIG = dict(
    experiment='E9', profile='P1_cpu2', min_effective=0.5, inject_seed_base=1100,  # inject_seed_base giống E1
    cell_order_seed=9102, latin_seed=9103, n_images=20, images_per_block=2,
    canvases=[dict(canvas=320, anchors=2100, n=[0, 100, 300, 1000, 2100], warmup=1, repeats=5),
              dict(canvas=640, anchors=8400, n=[0, 300, 1000, 3000, 8400], warmup=1, repeats=3)],
    geometries=['disjoint', 'dense_cluster', 'mixed'], caps=[None, 300], nc=80, tau=.01, iou=.45,
    backends=BACKEND_NAMES, deadline_ms=100.0,
    baseline_copies=3,  # ô N = 0 (mẫu số của mọi tỷ số) đo 3 lần, rải ngẫu nhiên trong mỗi ảnh
)
SMOKE = dict(n_images=2)


def inject_seed(cfg, anchors, n, nc, geometry):
    """Như E1: seed tất định theo (A, N, nc, hình học)."""
    return cfg['inject_seed_base'] + zlib.crc32(f'{anchors}|{n}|{nc}|{geometry}'.encode()) % 100000


def cells(cfg, c):
    out = []
    for n, geom, cap in itertools.product(c['n'], cfg['geometries'], cfg['caps']):
        if cap is not None and n <= cap:
            continue
        if n == 0 and geom != 'disjoint':
            continue
        for copy in range(cfg['baseline_copies'] if n == 0 else 1):
            out.append(dict(canvas=c['canvas'], anchors=c['anchors'], n=n, geometry=geom, cap=cap, copy=copy))
    return out


def image_ids(n):
    spec = importlib.util.spec_from_file_location('e5run', SIM / 'studies' / 'e5_protocol' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IMAGE_IDS[:n]


def worker(run_dir, cfg):
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[cfg['profile']]
    readback = hw.pin(cpus)
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import attacks_provenance as ap_, inject, provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    from sponge_sim.cohorts import REPO
    w = timing.JsonlWriter(run_dir / f'raw_{cfg["profile"]}.jsonl')
    w.write(dict(kind='pin', profile=cfg['profile'], allowed=cpus, readback=readback, verify=verify,
                 torch_threads=torch.get_num_threads()))
    if not verify['passed']:
        w.close()
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_E9)')
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(REPO / 'yolov8n.pt')).model.eval().float()
    ids = image_ids(cfg['n_images'])
    # tập ứng viên tiêm: dựng một lần cho mỗi ô, dùng chung cho mọi ảnh
    injected = {}
    for c in cfg['canvases']:
        for g in cells(cfg, c):
            key = (g['anchors'], g['n'], g['geometry'])
            if key not in injected:
                seed = inject_seed(cfg, g['anchors'], g['n'], cfg['nc'], g['geometry'])
                injected[key] = (*inject.make(g['n'], g['geometry'], seed, anchors=g['anchors'], tau=cfg['tau'], nc=cfg['nc']), seed)
    n_cells = sum(len(cells(cfg, c)) for c in cfg['canvases']) * len(ids)
    orders = timing.latin_blocks(cfg['backends'], n_cells, cfg['latin_seed'])
    order_k, block = 0, -1
    ref0 = timing.reference_ms()
    for c in cfg['canvases']:
        for ii, image_id in enumerate(ids):
            if ii % cfg['images_per_block'] == 0:
                block += 1
                w.write(dict(kind='reference', block=block, **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
            image = ap_.load_image_canvas(data / 'images' / names[image_id], c['canvas'])
            cs = cells(cfg, c)
            random.Random(cfg['cell_order_seed'] + image_id).shuffle(cs)
            for g in cs:
                decoded, meta, seed = injected[(g['anchors'], g['n'], g['geometry'])]
                active, n_tau = pv.tau_filter(decoded, cfg['tau'], g['cap'])
                assert n_tau == g['n']
                boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
                order = orders[order_k]
                order_k += 1
                for rep in range(-c['warmup'], c['repeats']):
                    for name in order:
                        b = BACKENDS[name]

                        def full():
                            with torch.no_grad():
                                raw = model(image)
                            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
                            _ = raw.shape  # forward thật; đầu ra giải mã bị thay bằng ứng viên tiêm
                            act, _n = pv.tau_filter(decoded, cfg['tau'], g['cap'])
                            return b.run(b.prepare(act[:, :4], act[:, 4], act[:, 5], cfg['nc']), cfg['iou'])
                        tt_total = timing.timed(full)[0]
                        tt = timing.timed(pv.tau_filter, setup=lambda: (decoded, cfg['tau'], g['cap']))[0]
                        tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes, cfg['nc']), cfg['iou']))[0]
                        if rep < 0:
                            continue
                        w.write(dict(kind='cell', profile=cfg['profile'], block=block, image_id=image_id, rep=rep, backend=name,
                                     canvas=g['canvas'], anchors=g['anchors'], n=g['n'], geometry=g['geometry'], cap=g['cap'], copy=g['copy'],
                                     nc=cfg['nc'], total_ns=tt_total['ns'], tau_ns=tt['ns'], nms_ns=tn['ns'],
                                     nms_input=len(scores), kept=len(tn['result']),
                                     miss=tt_total['ns'] / 1e6 > cfg['deadline_ms'],
                                     inject_sha256=meta['sha256'], inject_seed=seed, wall_utc=tt_total['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.write(dict(kind='pin_end', verify=hw.verify_pin(cpus, min_effective=cfg['min_effective'])))  # chỉ ghi, không loại dữ liệu
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = {**CONFIG, **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    if a.worker:
        worker(run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)  # không ghi đè run cũ
    from sponge_sim import manifest
    n_cells = {c['canvas']: len(cells(cfg, c)) for c in cfg['canvases']}
    manifest.write(run_dir / 'manifest.json', manifest.build({**cfg, 'cells_per_image': n_cells}))
    t0 = time.time()
    cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker'] + (['--smoke'] if a.smoke else [])
    subprocess.run(cmd, check=True)
    print(f'[E9] xong sau {time.time() - t0:.1f}s', flush=True)
    print('[E9] DONE', flush=True)


if __name__ == '__main__':
    main()
