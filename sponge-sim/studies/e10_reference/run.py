"""E10 — đo lại các nhiễu loạn toàn ảnh Phantom Sponges / Overload đã lưu (YOLOv5s, 640 px, letterbox gốc) dưới giao thức
ghim lõi, để có tập ứng viên THẬT ở vùng 10^3–10^4 và kiểm mô hình Amdahl tiến cứu. Xem prereg/PREREG_E10.md.

Mỗi đơn vị = (ảnh, nhánh). Nhánh: clean_a, clean_b (hai bản sạch xen kẽ = A/A trong phiên), overload s0–s4, phantom s0–s4.
Với mỗi đơn vị: forward (không đo) -> decode -> N tại τ 0,01 và 0,25; NMS riêng 4 backend ở τ 0,25 (1 khởi động + 3 lần);
toàn trình (forward + decode + τ 0,25 + NMS, một lần gọi) cho tv_nms, tv_batched, ultra_native (1 khởi động + 3 lần).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))

CONFIG = dict(experiment='E10', profile='P1_cpu2', min_effective=0.5, n_images=50, seeds=[0, 1, 2, 3, 4],
              methods=['overload', 'phantom_sponges'], tau_list=[.01, .25], tau_nms=.25, iou=.45, nc=80,
              micro_backends=['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native'], micro_warmup=1, micro_repeats=3,
              pipeline_backends=['tv_nms', 'tv_batched', 'ultra_native'], pipeline_warmup=1, pipeline_repeats=3,
              order_seed=10102, latin_seed=10103, images_per_block=5)
SMOKE = dict(n_images=2)


def image_ids(n):
    spec = importlib.util.spec_from_file_location('e5run', SIM / 'studies' / 'e5_protocol' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IMAGE_IDS[:n]


def worker(run_dir, cfg):
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[cfg['profile']]
    readback = hw.pin(cpus)
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    from sponge_sim import provenance as pv, timing, yolov5_ref
    from sponge_sim.backends import BACKENDS
    from sponge_sim.cohorts import REPO
    w = timing.JsonlWriter(run_dir / f'raw_{cfg["profile"]}.jsonl')
    w.write(dict(kind='pin', profile=cfg['profile'], allowed=cpus, readback=readback, verify=verify, torch_threads=torch.get_num_threads()))
    if not verify['passed']:
        w.close()
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_E10)')
    model, LoadImages = yolov5_ref.load()
    # nhiễu loạn đã lưu: kiểm SHA256 so với bản ghi lúc sinh (fail-closed)
    pert = {}
    ref = REPO / 'outputs/extended_q1_20261002/reference'
    for m in cfg['methods']:
        for s in cfg['seeds']:
            rec = json.loads((ref / f'{m}_seed{s}.json').read_text(encoding='utf-8'))
            path = REPO / rec['perturbation_file'].replace('\\', '/')
            if yolov5_ref.sha256(path) != rec['trained_float_perturbation_sha256']:
                raise SystemExit(f'SHA256 nhiễu loạn lệch bản ghi: {m} seed {s}')
            pert[(m, s)] = torch.from_numpy(np.load(path))
    w.write(dict(kind='perturbations', items={f'{m}|{s}': yolov5_ref.sha256(REPO / json.loads((ref / f'{m}_seed{s}.json').read_text(encoding='utf-8'))['perturbation_file'].replace('\\', '/'))
                                              for m, s in pert}))
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    ids = image_ids(cfg['n_images'])
    arms = [('clean_a', None, None), ('clean_b', None, None)] + [(m, m, s) for m in cfg['methods'] for s in cfg['seeds']]
    rng = random.Random(cfg['order_seed'])
    orders = timing.latin_blocks(cfg['micro_backends'], len(ids) * len(arms), cfg['latin_seed'])
    porders = timing.latin_blocks(cfg['pipeline_backends'], len(ids) * len(arms), cfg['latin_seed'] + 1)
    k = 0
    ref0 = timing.reference_ms()
    for ii, image_id in enumerate(ids):
        if ii % cfg['images_per_block'] == 0:
            w.write(dict(kind='reference', block=ii // cfg['images_per_block'], **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
        loader = LoadImages(str(data / 'images' / names[image_id]), img_size=640, stride=32, auto=False)
        _, im, _, _, _ = next(iter(loader))
        clean = torch.from_numpy(im).float()[None] / 255
        units = arms[:]
        rng.shuffle(units)
        for name, m, s in units:
            image = clean if m is None else (clean + pert[(m, s)][None]).clamp(0, 1)
            with torch.no_grad():
                raw = model(image)[0][0]
            decoded = yolov5_ref.decode(raw)
            n_by_tau = {str(t): int((decoded[:, 4] > t).sum()) for t in cfg['tau_list']}
            active, _ = pv.tau_filter(decoded, cfg['tau_nms'], None)
            boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
            base = dict(profile=cfg['profile'], image_id=image_id, arm=name, method=m, seed=s, block=ii // cfg['images_per_block'], n_by_tau=n_by_tau)
            for rep in range(-cfg['micro_warmup'], cfg['micro_repeats']):
                for be in orders[k]:
                    b = BACKENDS[be]
                    tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes, cfg['nc']), cfg['iou']))[0]
                    if rep >= 0:
                        w.write(dict(kind='micro', **base, rep=rep, backend=be, nms_input=len(scores), kept=len(tn['result']), nms_ns=tn['ns']))
            for rep in range(-cfg['pipeline_warmup'], cfg['pipeline_repeats']):
                for be in porders[k]:
                    b = BACKENDS[be]

                    def full():
                        with torch.no_grad():
                            r = model(image)[0][0]
                        d = yolov5_ref.decode(r)
                        a, _n = pv.tau_filter(d, cfg['tau_nms'], None)
                        return b.run(b.prepare(a[:, :4], a[:, 4], a[:, 5], cfg['nc']), cfg['iou'])
                    tt = timing.timed(full)[0]
                    if rep >= 0:
                        w.write(dict(kind='pipeline', **base, rep=rep, backend=be, total_ns=tt['ns']))
            k += 1
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.write(dict(kind='pin_end', verify=hw.verify_pin(cpus, min_effective=cfg['min_effective'])))
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
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg, extra_files=[SIM / 'src' / 'sponge_sim' / 'yolov5_ref.py']))
    t0 = time.time()
    subprocess.run([sys.executable, '-X', 'utf8', '-B', __file__, '--run-dir', str(run_dir), '--worker'] + (['--smoke'] if a.smoke else []), check=True)
    print(f'[E10] xong sau {time.time() - t0:.1f}s', flush=True)
    print('[E10] DONE', flush=True)


if __name__ == '__main__':
    main()
