"""E1 — đối chứng dương + dose-response: tiêm ứng viên tại ranh giới decoded. Xem prereg/PREREG_E1.md.

Điều phối: `run.py --run-dir DIR [--smoke]`; mỗi profile một subprocess (`--worker PROFILE`), ghim trước khi torch tạo thread.
Phần A (vi benchmark): mỗi nhóm (A, N, geometry, nc, cap) đo xen kẽ 4 backend theo vòng (round-robin), thứ tự backend
theo hình vuông Latin; mỗi lần lặp đo riêng tau-filter và NMS (prepare() ngoài vùng đo).
Phần B (toàn trình, thứ cấp): forward ảnh sạch thật + ứng viên tiêm thay decoded -> tau -> NMS, đo trực tiếp một lần gọi.
"""
from __future__ import annotations

import argparse
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
    experiment='E1', profiles=['P1_cpu2', 'E1'], min_effective={'P1': 0.5, 'P1_cpu2': 0.5, 'E1': 0.5, 'E4': 2.8}, profile_order_seed=1021, group_order_seed=1022, latin_seed=1023,
    inject_seed_base=1100,
    grid=[  # (anchors, N levels, nc levels)
        dict(anchors=2100, label='head 320 (pipeline)', n=[0, 10, 30, 100, 300, 600, 1000, 1500, 2100], nc=[80, 1]),
        dict(anchors=8400, label='head 640 (tham chiếu, ngoài pipeline)', n=[0, 100, 300, 1000, 3000, 8400], nc=[80]),
        dict(anchors=10000, label='đầu giả 10^4 (vượt head thật; chỉ để phủ N đến 10^4)', n=[100, 300, 1000, 3000, 10000], nc=[80]),
    ],
    geometries=['disjoint', 'dense_cluster', 'mixed'], caps=[None, 300], tau=.01, iou=.45,
    backends=BACKEND_NAMES, repeats=30, warmup=3, groups_per_block=12,
    pipeline=dict(anchors=2100, n=[0, 100, 300, 1000, 2100], geometry='disjoint', nc=80, cap=None,
                  backends=['torch_pure', 'ultra_native'], image_id=139, repeats=20, warmup=3, deadline_ms=100.0),
)
SMOKE = dict(repeats=2, warmup=1, smoke_groups=6)
# Điện thoại (PREREG_E1.md, mục "E1 trên điện thoại"): lưới gọn, 2 backend, 10 lần lặp, bỏ phần B (toàn trình), cận dưới ghim 0,5.
PHONE = dict(
    grid=[dict(anchors=2100, label='head 320 (pipeline)', n=[100, 600, 2100], nc=[80]),
          dict(anchors=10000, label='đầu giả 10^4 (vượt head thật)', n=[100, 300, 1000, 3000, 10000], nc=[80])],
    caps=[None], backends=['torch_pure', 'tv_nms'], repeats=10, warmup=2, groups_per_block=6,
    pipeline=dict(CONFIG['pipeline'], n=[]), min_effective_default=0.5, device='phone')


def inject_seed(cfg, anchors, n, nc, geometry):
    """Seed tất định (không dùng hash() vì PYTHONHASHSEED ngẫu nhiên hóa chuỗi)."""
    return cfg['inject_seed_base'] + zlib.crc32(f'{anchors}|{n}|{nc}|{geometry}'.encode()) % 100000


def groups(cfg):
    out = []
    for g in cfg['grid']:
        for n, nc, geom, cap in itertools.product(g['n'], g['nc'], cfg['geometries'], cfg['caps']):
            if cap is not None and n <= cap:
                continue  # cap không tác động khi N <= cap
            if n == 0 and geom != 'disjoint':
                continue  # N=0: hình học không xác định, chỉ giữ một ô
            out.append(dict(anchors=g['anchors'], n=n, nc=nc, geometry=geom, cap=cap))
    return out


def worker(profile, run_dir, cfg):
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[profile]
    readback = hw.pin(cpus)
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'].get(profile, cfg.get('min_effective_default')))
    torch.set_num_threads(len(cpus))
    from sponge_sim import inject, provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    w = timing.JsonlWriter(run_dir / f'raw_{profile}.jsonl')
    w.write(dict(kind='pin', profile=profile, allowed=cpus, readback=readback, verify=verify,
                 torch_threads=torch.get_num_threads()))
    gs = groups(cfg)
    random.Random(cfg['group_order_seed']).shuffle(gs)
    if cfg.get('smoke_groups'):
        gs = gs[:cfg['smoke_groups']]
    orders = timing.latin_blocks(cfg['backends'], len(gs), cfg['latin_seed'])
    ref0 = timing.reference_ms()
    for gi, g in enumerate(gs):
        if gi % cfg['groups_per_block'] == 0:
            w.write(dict(kind='reference', block=gi // cfg['groups_per_block'],
                         **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
        seed = inject_seed(cfg, g['anchors'], g['n'], g['nc'], g['geometry'])
        decoded, meta = inject.make(g['n'], g['geometry'], seed, anchors=g['anchors'], tau=cfg['tau'], nc=g['nc'])
        active, n_tau = pv.tau_filter(decoded, cfg['tau'], g['cap'])
        assert n_tau == g['n']
        boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
        order = orders[gi]
        for rep in range(-cfg['warmup'], cfg['repeats']):
            for name in order:
                b = BACKENDS[name]
                tt = timing.timed(pv.tau_filter, setup=lambda: (decoded, cfg['tau'], g['cap']))[0]
                tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes, g['nc']), cfg['iou']))[0]
                if rep < 0:
                    continue
                w.write(dict(kind='micro', profile=profile, group=gi, rep=rep, backend=name, **g,
                             n_tau=n_tau, nms_input=len(scores), kept=len(tn['result']),
                             tau_ns=tt['ns'], nms_ns=tn['ns'], inject_sha256=meta['sha256'], inject_seed=seed,
                             wall_utc=tn['wall_utc']))
    # Phần B: toàn trình
    p = cfg['pipeline']
    if p['n']:
        from ultralytics import YOLO
        from sponge_sim.cohorts import REPO
        data = REPO / 'outputs/datasets/coco_val2017_extended'
        names = {im['id']: im['file_name'] for im in
                 json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
        model = YOLO(str(REPO / 'yolov8n.pt')).model.eval().float()
        image = pv.load_image(data / 'images' / names[p['image_id']])
        reps = p['repeats'] if not cfg.get('smoke_groups') else 2
        for n in p['n']:
            seed = inject_seed(cfg, p['anchors'], n, p['nc'], p['geometry'])
            decoded, meta = inject.make(n, p['geometry'], seed, anchors=p['anchors'], tau=cfg['tau'], nc=p['nc'])
            for rep in range(-p['warmup'], reps):
                for name in p['backends']:
                    b = BACKENDS[name]
                    def full():
                        with torch.no_grad():
                            raw = model(image)
                        raw = raw[0] if isinstance(raw, (tuple, list)) else raw
                        _ = raw.shape  # forward thật; decoded thật bị thay bằng ứng viên tiêm
                        act, _n = pv.tau_filter(decoded, cfg['tau'], p['cap'])
                        return b.run(b.prepare(act[:, :4], act[:, 4], act[:, 5], p['nc']), cfg['iou'])
                    r = timing.timed(full)[0]
                    if rep >= 0:
                        w.write(dict(kind='pipeline', profile=profile, rep=rep, backend=name, n=n, anchors=p['anchors'],
                                     geometry=p['geometry'], nc=p['nc'], total_ns=r['ns'],
                                     deadline_ms=p['deadline_ms'], miss=r['ns'] / 1e6 > p['deadline_ms'],
                                     inject_sha256=meta['sha256'], wall_utc=r['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker')
    ap.add_argument('--phone', action='store_true', help='lưới/backend gọn cho điện thoại; profile lấy từ SPONGE_PROFILES(_FILE)')
    ap.add_argument('--profiles', default='', help='ghi đè danh sách profile (phân cách bằng dấu phẩy)')
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = {**CONFIG, **(PHONE if a.phone else {}), **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    if a.profiles:
        cfg['profiles'] = a.profiles.split(',')
    if a.worker:
        worker(a.worker, run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build({**cfg, 'n_groups': len(groups(cfg))}))
    order = cfg['profiles'][:]
    random.Random(cfg['profile_order_seed']).shuffle(order)
    for prof in order:
        t0 = time.time()
        cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker', prof]
        if a.smoke:
            cmd.append('--smoke')
        if a.phone:
            cmd.append('--phone')
        if a.profiles:
            cmd += ['--profiles', a.profiles]
        subprocess.run(cmd, check=True)
        print(f'[E1] profile {prof} xong sau {time.time() - t0:.1f}s', flush=True)
    print('[E1] DONE', flush=True)


if __name__ == '__main__':
    main()
