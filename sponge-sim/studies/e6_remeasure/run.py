"""E6: đo lại patch THẬT bằng giao thức ghim core (laptop; điện thoại nếu có giờ). Xem prereg/PREREG_E6.md.

Mỗi đơn vị = (ảnh giữ-ra-ngoài, nhánh, seed). Nhánh 'clean' không có patch. Với mỗi đơn vị:
  forward thật -> decode -> N tại hai ngưỡng (0,01 và 0,25) -> đo riêng lọc tau + NMS cho 4 backend (xen kẽ vòng tròn, prepare() ngoài vùng đo)
  -> đo toàn trình (chèn patch -> forward -> decode -> tau -> NMS) cho torch_pure và tv_nms.
Chạy: python studies/e6_remeasure/run.py --run-dir results/e6/<id> --profile P1_cpu2 --e7-dir <thư mục runs/e7 đã mang về> [--arms ...] [--smoke]
Fail-closed: với nhánh clean (và R_sg) ở 320 px, N tại ngưỡng 0,01 phải bằng bản ghi lịch sử đã khóa; lệch thì dừng.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))
sys.path.insert(0, str(SIM / 'studies' / 'e5_protocol'))
from run import IMAGE_IDS  # noqa: E402  (cùng 100 ảnh với E5)

# nhánh: (nguồn, tên điều kiện/thư mục, canvas, kiểu vị trí, kích thước patch, hậu tố checkpoint cuối)
FINAL_Q = 62500  # có thể ghi đè bằng --final-q
ARMS = {
    'C1': ('e7', 'C1_320_s64_t01', 320, 'saliency', 64),
    'C2': ('e7', 'C2_640_s128_t01', 640, 'saliency', 128),
    'C3': ('e7', 'C3_320_s64_t25', 320, 'saliency', 64),
    'C4': ('e7', 'C4_640_s64_t01', 640, 'saliency', 64),
    'B1': ('e7', 'B1_ga_center', 320, 'center', 64),
    'B2': ('e7', 'B2_ga_random_location', 320, 'random', 64),
    'B3': ('e7', 'B3_random_search', 320, 'saliency', 64),
    'B4': ('e7', 'B4_gradient_tanh', 320, 'saliency', 64),
    'clean640': ('none', 'clean640', 640, 'none', 0),  # PREREG_E6_addendum_clean640: ảnh sạch ở canvas 640
    'R_sg': ('old', 'sg_ga', 320, 'saliency', 64),               # patch cũ (4 ảnh tối ưu), chỉ tham khảo
    'R_tanh': ('old', 'tanh_whitebox_patch', 320, 'saliency', 64),
}
OLD_Q = {'R_sg': 2500, 'R_tanh': 400}
PRIMARY = ['clean', 'C1', 'C2', 'C3']
SECONDARY = ['C4', 'B1', 'B2', 'B3', 'B4', 'R_sg', 'R_tanh']
CONFIG = dict(
    experiment='E6', tau_list=[0.01, 0.25], iou=.45, backends=['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native'],
    micro_repeats=10, micro_warmup=2, pipeline_backends=['torch_pure', 'tv_nms'], pipeline_repeats=5, pipeline_warmup=1,
    seeds=[0, 1, 2, 3, 4], n_images_primary=100, n_images_secondary=50, order_seed=3301, latin_seed=3302, images_per_block=10,
    min_effective=0.5,
)


def sha256_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def patch_path(arm, seed, e7_dir, repo, final_q=FINAL_Q):
    src, name, canvas, _, _ = ARMS[arm]
    if src == 'e7':
        suf = f'q{final_q}' + (f'_c{canvas}' if canvas != 320 else '')
        return Path(e7_dir) / name / f'seed{seed}' / f'patch_{suf}.npy'
    return repo / f'outputs/x1_500img_20261018/optimization/{name}_seed{seed}/patch_q{OLD_Q[arm]}.npy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--profile', default='P1_cpu2')
    ap.add_argument('--e7-dir', default='')
    ap.add_argument('--secondary-offset', type=int, default=0, help='E6-A2: vị trí bắt đầu tập ảnh của nhánh phụ')
    ap.add_argument('--only-secondary-images', action='store_true', help='E6-A2: chỉ đo các ảnh của nhánh phụ')
    ap.add_argument('--skip-clean320', action='store_true', help='E6-A1: không đo lại nhánh clean 320 px')
    ap.add_argument('--arms', default=','.join(PRIMARY[1:] + SECONDARY))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--final-q', type=int, default=FINAL_Q, help='checkpoint E7 dùng làm patch cuối (mặc định 62500; smoke dùng 40)')
    a = ap.parse_args()
    final_q = a.final_q
    cfg = dict(CONFIG, smoke=a.smoke, profile=a.profile)
    if a.smoke:
        cfg.update(n_images_primary=2, n_images_secondary=2, seeds=[0], micro_repeats=2, pipeline_repeats=1)
    arms = [x for x in a.arms.split(',') if x]
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[a.profile]
    readback = hw.pin(cpus)
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import attacks_provenance as ap_, cohorts, manifest, provenance as pv, timing
    from sponge_sim.backends import BACKENDS
    repo = cohorts.REPO
    if not verify['passed']:
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_E6)')

    patches, patch_sha = {}, {}
    for arm in arms:
        if ARMS[arm][0] == 'none':
            continue
        for s in cfg['seeds']:
            path = patch_path(arm, s, a.e7_dir, repo, final_q)
            if not path.exists():
                raise SystemExit(f'Thiếu patch {arm} seed {s}: {path}')
            arr = np.load(path)
            patches[(arm, s)] = torch.from_numpy(arr)
            patch_sha[f'{arm}|{s}'] = dict(path=str(path), sha256=sha256_array(arr))
    idx = Path(a.e7_dir) / 'patches_index.json' if a.e7_dir else None
    idx_sha = {}
    if idx and idx.exists():  # đối chiếu SHA256 với bản ghi lúc sinh patch (GPU): lệch thì dừng
        for r in json.loads(idx.read_text(encoding='utf-8')):
            if r['queries'] == final_q:
                key = Path(r['path']).parts
                idx_sha[(key[0], int(r['seed']))] = r['patch_sha256']
        for arm in arms:
            if ARMS[arm][0] == 'e7':
                for s in cfg['seeds']:
                    want = idx_sha.get((ARMS[arm][1], s))
                    got = patch_sha[f'{arm}|{s}']['sha256']
                    if want and want != got:
                        raise SystemExit(f'SHA256 patch lệch bản ghi GPU: {arm} seed {s}')

    config = dict(cfg, arms=arms, patch_sha=patch_sha)
    manifest.write(run_dir / 'manifest.json', manifest.build(config))
    data = repo / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(repo / 'yolov8n.pt')).model.eval().float()
    clean_hist = {s: cohorts.records(cohorts.read_case('X1-500', 'clean', 0, 0)) for s in [0]}
    rsg_hist = {s: cohorts.records(cohorts.read_case('X1-500', 'sg_ga', s, 2500)) for s in cfg['seeds']} if 'R_sg' in arms else {}
    w = timing.JsonlWriter(run_dir / f'raw_{a.profile}.jsonl')
    w.write(dict(kind='pin', profile=a.profile, allowed=cpus, readback=readback, verify=verify, torch_threads=torch.get_num_threads()))

    ids_primary = IMAGE_IDS[:cfg['n_images_primary']]
    off = a.secondary_offset  # E6-A2: nhánh phụ trên ảnh 51-100 để bảng cân (PREREG_E6_A2)
    ids_secondary = set(ids_primary[off:off + cfg['n_images_secondary']])
    if a.only_secondary_images:
        ids_primary = [i for i in ids_primary if i in ids_secondary]
    rng = random.Random(cfg['order_seed'])
    orders = timing.latin_blocks(cfg['backends'], 40000, cfg['latin_seed'])
    order_k = 0
    ref0 = timing.reference_ms()
    cache_img = {}
    for bi, image_id in enumerate(ids_primary):
        if bi % cfg['images_per_block'] == 0:
            w.write(dict(kind='reference', block=bi // cfg['images_per_block'], **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
        units = [] if a.skip_clean320 else [('clean', None)]
        units += [('clean640', None)] if 'clean640' in arms else []
        for arm in arms:
            if ARMS[arm][0] == 'none':
                continue
            if arm in SECONDARY and image_id not in ids_secondary:
                continue
            units += [(arm, s) for s in cfg['seeds']]
        rng.shuffle(units)
        for arm, seed in units:
            canvas = 320 if arm == 'clean' else ARMS[arm][2]
            key = (image_id, canvas)
            if key not in cache_img:
                cache_img.clear()
                cache_img[key] = ap_.load_image_canvas(data / 'images' / names[image_id], canvas)
            base = cache_img[key]
            if arm in ('clean', 'clean640'):
                img, loc = base, None
            else:
                _, _, _, location, size = ARMS[arm]
                loc = ap_.find_locations([base], size, location, (seed if location != 'saliency' else 0) + image_id, canvas=canvas)[0]
                img = ap_.insert(base, patches[(arm, seed)][None], [tuple(loc)])
            with torch.no_grad():
                raw = model(img)
            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
            decoded = pv.decode(raw[0])
            n_by_tau = {t: int((decoded[:, 4] > t).sum()) for t in cfg['tau_list']}
            if canvas == 320 and (arm == 'clean' or arm == 'R_sg'):  # fail-closed vs bản ghi lịch sử đã khóa
                hist = clean_hist[0] if arm == 'clean' else rsg_hist[seed]
                if image_id in hist and n_by_tau[0.01] != hist[image_id]['active_candidates']:
                    w.write(dict(kind='abort', reason='N lệch bản ghi lịch sử', image_id=image_id, arm=arm, seed=seed,
                                 got=n_by_tau[0.01], expected=hist[image_id]['active_candidates']))
                    raise SystemExit(f'lệch N: ảnh {image_id} {arm} seed {seed}: {n_by_tau[0.01]} != {hist[image_id]["active_candidates"]}')
            base_row = dict(profile=a.profile, image_id=image_id, arm=arm, seed=seed, canvas=canvas, location=loc,
                            block=bi // cfg['images_per_block'], n_by_tau={str(t): n for t, n in n_by_tau.items()})
            for tau in cfg['tau_list']:
                active, _ = pv.tau_filter(decoded, tau, None)
                boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
                order = orders[order_k]
                order_k += 1
                for rep in range(-cfg['micro_warmup'], cfg['micro_repeats']):
                    for name in order:
                        b = BACKENDS[name]
                        tt = timing.timed(pv.tau_filter, setup=lambda: (decoded, tau, None))[0]
                        tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes), cfg['iou']))[0]
                        if rep >= 0:
                            w.write(dict(kind='micro', **base_row, tau=tau, rep=rep, backend=name, nms_input=len(scores),
                                         kept=len(tn['result']), tau_ns=tt['ns'], nms_ns=tn['ns'], wall_utc=tn['wall_utc']))
            for rep in range(-cfg['pipeline_warmup'], cfg['pipeline_repeats']):
                for name in cfg['pipeline_backends']:
                    b = BACKENDS[name]
                    def full():
                        im = base if arm in ('clean', 'clean640') else ap_.insert(base, patches[(arm, seed)][None], [tuple(loc)])
                        with torch.no_grad():
                            r = model(im)
                        r = r[0] if isinstance(r, (tuple, list)) else r
                        act, _ = pv.tau_filter(pv.decode(r[0]), 0.01, None)
                        return b.run(b.prepare(act[:, :4], act[:, 4], act[:, 5]), cfg['iou'])
                    r = timing.timed(full)[0]
                    if rep >= 0:
                        w.write(dict(kind='pipeline', **base_row, rep=rep, backend=name, total_ns=r['ns'], wall_utc=r['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.close()
    print('[E6] DONE', flush=True)


if __name__ == '__main__':
    main()
