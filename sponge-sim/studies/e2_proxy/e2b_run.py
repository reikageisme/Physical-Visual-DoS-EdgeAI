"""E2b — proxy N_tau trên ảnh thật + patch thật, đo theo giao thức E5. Xem prereg/PREREG_E2b.md.

Mỗi ảnh (100 ảnh của E5) × điều kiện {clean, sg_ga q2500 seed 0..9}: tái sinh decoded bằng forward mới, insert patch tại
`location` đã lưu trong record X1-500; KIỂM active_candidates khớp record (fail-closed: lệch -> dừng).
Sau đó đo: tau-filter + NMS cho 4 backend × cap {None, 100} (round-robin, prepare() ngoài vùng đo) và toàn trình
(load -> insert -> forward -> decode -> tau -> NMS torch_pure / tv_nms) đo trực tiếp.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(SIM / 'studies' / 'e5_protocol'))
from run import IMAGE_IDS  # noqa: E402  (cùng 100 ảnh với E5)

CONFIG = dict(
    experiment='E2b', profile='P1_cpu2', min_effective=0.5, cohort='X1-500', attack='sg_ga', q=2500, seeds=list(range(10)),
    image_ids='= E5 IMAGE_IDS (100)', tau=.01, caps=[None, 100], iou=.45,
    backends=['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native'], micro_repeats=10, micro_warmup=2,
    pipeline_backends=['torch_pure', 'tv_nms'], pipeline_repeats=5, pipeline_warmup=1,
    condition_order_seed=2201, latin_seed=2202, images_per_block=10,
)
SMOKE = dict(n_images=2, seeds=[0, 1], micro_repeats=2, pipeline_repeats=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = {**CONFIG, **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[cfg['profile']]
    readback = hw.pin(cpus)
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import cohorts, manifest, provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg))
    data = cohorts.REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(cohorts.REPO / 'yolov8n.pt')).model.eval().float()
    w = timing.JsonlWriter(run_dir / f'raw_{cfg["profile"]}.jsonl')
    w.write(dict(kind='pin', profile=cfg['profile'], allowed=cpus, readback=readback, verify=verify,
                 torch_threads=torch.get_num_threads()))
    if not verify['passed']:
        w.write(dict(kind='abort', reason='verify ghim FAIL'))
        raise SystemExit('verify ghim FAIL -> dừng theo PREREG_E2b')
    clean_recs = {s: cohorts.records(cohorts.read_case('X1-500', 'clean', s, 0)) for s in cfg['seeds']}
    atk_recs = {s: cohorts.records(cohorts.read_case('X1-500', cfg['attack'], s, cfg['q'])) for s in cfg['seeds']}
    patches = {s: torch.from_numpy(np.load(cohorts.COHORTS['X1-500'] /
                                           f'optimization/{cfg["attack"]}_seed{s}/patch_q{cfg["q"]}.npy'))
               for s in cfg['seeds']}
    conds = [('clean', None)] + [(cfg['attack'], s) for s in cfg['seeds']]
    ids = IMAGE_IDS[:cfg.get('n_images', len(IMAGE_IDS))]
    rng = random.Random(cfg['condition_order_seed'])
    orders = timing.latin_blocks(cfg['backends'], len(ids) * len(conds), cfg['latin_seed'])
    ref0 = timing.reference_ms()
    k = 0

    def make_image(path, cond, seed, image_id):
        img = pv.load_image(path)
        if cond != 'clean':
            img = pv.insert(img, patches[seed][None], [tuple(atk_recs[seed][image_id]['location'])])
        return img

    for bi, image_id in enumerate(ids):
        if bi % cfg['images_per_block'] == 0:
            w.write(dict(kind='reference', block=bi // cfg['images_per_block'],
                         **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
        path = data / 'images' / names[image_id]
        order = conds[:]
        rng.shuffle(order)
        for cond, seed in order:
            img = make_image(path, cond, seed, image_id)
            with torch.no_grad():
                raw = model(img)
            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
            decoded = pv.decode(raw[0])
            expected = (clean_recs[0] if cond == 'clean' else atk_recs[seed])[image_id]['active_candidates']
            _, n_tau = pv.tau_filter(decoded, cfg['tau'], None)
            if n_tau != expected:  # fail-closed
                w.write(dict(kind='abort', reason='active_candidates lệch record', image_id=image_id, cond=cond,
                             seed=seed, got=n_tau, expected=expected))
                raise SystemExit(f'lệch N: ảnh {image_id} {cond} seed {seed}: {n_tau} != {expected}')
            base = dict(profile=cfg['profile'], image_id=image_id, cond=cond, seed=seed, n_tau=n_tau, block=bi // cfg['images_per_block'])
            for cap in cfg['caps']:
                active, _ = pv.tau_filter(decoded, cfg['tau'], cap)
                boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
                for rep in range(-cfg['micro_warmup'], cfg['micro_repeats']):
                    for name in orders[k]:
                        b = BACKENDS[name]
                        tt = timing.timed(pv.tau_filter, setup=lambda: (decoded, cfg['tau'], cap))[0]
                        tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes), cfg['iou']))[0]
                        if rep >= 0:
                            w.write(dict(kind='micro', **base, cap=cap, rep=rep, backend=name, nms_input=len(scores),
                                         kept=len(tn['result']), tau_ns=tt['ns'], nms_ns=tn['ns'], wall_utc=tn['wall_utc']))
            k += 1
            for rep in range(-cfg['pipeline_warmup'], cfg['pipeline_repeats']):
                for name in cfg['pipeline_backends']:
                    b = BACKENDS[name]
                    def full():
                        im = make_image(path, cond, seed, image_id)
                        with torch.no_grad():
                            r = model(im)
                        r = r[0] if isinstance(r, (tuple, list)) else r
                        act, _ = pv.tau_filter(pv.decode(r[0]), cfg['tau'], None)
                        return b.run(b.prepare(act[:, :4], act[:, 4], act[:, 5]), cfg['iou'])
                    r = timing.timed(full)[0]
                    if rep >= 0:
                        w.write(dict(kind='pipeline', **base, rep=rep, backend=name, total_ns=r['ns'], wall_utc=r['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.close()
    print('[E2b] DONE', flush=True)


if __name__ == '__main__':
    main()
