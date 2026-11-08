"""E11: đo lại bảng patch (E6) với bước chèn patch NGOÀI vùng bấm giờ ở mọi nhánh. Xem prereg/PREREG_E11.md.

Sửa lỗi của E6 (studies/e6_remeasure/run.py:198-205): hàm full() của E6 gọi ap_.insert trong vùng đo cho nhánh có patch
nhưng không cho nhánh sạch. Ở đây ảnh đầu vào (đã chèn patch hoặc sạch) được dựng MỘT lần trước khi đo, và mọi nhánh
đo cùng một hàm `pipeline_fn` (forward -> decode -> lọc tau -> NMS). Lý do chọn "đưa ra ngoài" thay vì "chèn giả ở nhánh sạch":
bước chèn là thao tác dựng kịch bản của thí nghiệm, không thuộc pipeline của bộ phát hiện (ngoài đời, ảnh có patch đến từ camera).

Thiết kế cân: mọi nhánh (C1-C4, B1-B4) trên CÙNG 100 ảnh và CÙNG seed patch 0-4. Sàn nhiễu trong phiên: hai nhánh sạch độc lập
ở mỗi canvas (clean_a/clean_b ở 320 px, clean640_a/clean640_b ở 640 px).
Chạy: python studies/e11_patch_remeasure/run.py --run-dir results/e11/<id> --e7-dir results/e7_gpu/final/runs/e7 [--smoke]
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

FINAL_Q = 62500
# nhánh: (tên thư mục E7, canvas, kiểu vị trí, kích thước patch); None = nhánh sạch
ARMS = {
    'clean_a': (None, 320, 'none', 0), 'clean_b': (None, 320, 'none', 0),
    'clean640_a': (None, 640, 'none', 0), 'clean640_b': (None, 640, 'none', 0),
    'C1': ('C1_320_s64_t01', 320, 'saliency', 64),
    'C2': ('C2_640_s128_t01', 640, 'saliency', 128),
    'C3': ('C3_320_s64_t25', 320, 'saliency', 64),
    'C4': ('C4_640_s64_t01', 640, 'saliency', 64),
    'B1': ('B1_ga_center', 320, 'center', 64),
    'B2': ('B2_ga_random_location', 320, 'random', 64),
    'B3': ('B3_random_search', 320, 'saliency', 64),
    'B4': ('B4_gradient_tanh', 320, 'saliency', 64),
}
CLEAN = [a for a, v in ARMS.items() if v[0] is None]
PATCHED = [a for a, v in ARMS.items() if v[0] is not None]
CONFIG = dict(
    experiment='E11', tau_list=[0.01, 0.25], iou=.45, backends=['torch_pure', 'tv_nms', 'tv_batched', 'ultra_native'],
    micro_repeats=10, micro_warmup=2, pipeline_backends=['torch_pure', 'tv_nms'], pipeline_repeats=5, pipeline_warmup=1,
    seeds=[0, 1, 2, 3, 4], n_images=100, order_seed=4401, latin_seed=4402, images_per_block=10, min_effective=0.5,
    insertion='outside_timed_region',
)


def sha256_array(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def build_input(arm, base, patch, loc, insert):
    """Dựng ảnh đầu vào NGOÀI vùng đo. Nhánh sạch: chính ảnh gốc."""
    if ARMS[arm][0] is None:
        return base
    return insert(base, patch[None], [tuple(loc)])


def pipeline_fn(model, img, backend, iou, pv, torch):
    """Hàm được bấm giờ: GIỐNG HỆT nhau cho mọi nhánh; không chứa bước chèn."""
    def full():
        with torch.no_grad():
            r = model(img)
        r = r[0] if isinstance(r, (tuple, list)) else r
        act, _ = pv.tau_filter(pv.decode(r[0]), 0.01, None)
        return backend.run(backend.prepare(act[:, :4], act[:, 4], act[:, 5]), iou)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--profile', default='P1_cpu2')
    ap.add_argument('--e7-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = dict(CONFIG, smoke=a.smoke, profile=a.profile)
    if a.smoke:
        cfg.update(n_images=2, seeds=[0], micro_repeats=2, pipeline_repeats=1)
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
    from run import IMAGE_IDS  # cùng 100 ảnh với E5/E6
    repo = cohorts.REPO
    if not verify['passed']:
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_E11)')

    # patch đóng băng: SHA256 phải khớp bản ghi lúc sinh trên GPU (patches_index.json), thiếu hoặc lệch thì dừng
    want = {}
    for r in json.loads((Path(a.e7_dir) / 'patches_index.json').read_text(encoding='utf-8')):
        if r['queries'] == FINAL_Q:
            want[(Path(r['path']).parts[0], int(r['seed']))] = r['patch_sha256']
    patches, patch_sha = {}, {}
    for arm in PATCHED:
        name, canvas, _, _ = ARMS[arm]
        for s in cfg['seeds']:
            path = Path(a.e7_dir) / name / f'seed{s}' / (f'patch_q{FINAL_Q}' + (f'_c{canvas}' if canvas != 320 else '') + '.npy')
            arr = np.load(path)
            got = sha256_array(arr)
            if want.get((name, s)) != got:
                raise SystemExit(f'SHA256 patch không khớp bản ghi GPU: {arm} seed {s} ({got} != {want.get((name, s))})')
            patches[(arm, s)] = torch.from_numpy(arr)
            patch_sha[f'{arm}|{s}'] = dict(path=str(path), sha256=got)

    manifest.write(run_dir / 'manifest.json', manifest.build(dict(cfg, arms=list(ARMS), patch_sha=patch_sha)))
    data = repo / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    model = YOLO(str(repo / 'yolov8n.pt')).model.eval().float()
    clean_hist = cohorts.records(cohorts.read_case('X1-500', 'clean', 0, 0))
    w = timing.JsonlWriter(run_dir / f'raw_{a.profile}.jsonl')
    w.write(dict(kind='pin', profile=a.profile, allowed=cpus, readback=readback, verify=verify, torch_threads=torch.get_num_threads()))

    rng = random.Random(cfg['order_seed'])
    orders = timing.latin_blocks(cfg['backends'], 40000, cfg['latin_seed'])
    porders = timing.latin_blocks(cfg['pipeline_backends'], 40000, cfg['latin_seed'] + 1)
    order_k = 0
    ref0 = timing.reference_ms()
    for bi, image_id in enumerate(IMAGE_IDS[:cfg['n_images']]):
        if bi % cfg['images_per_block'] == 0:
            w.write(dict(kind='reference', block=bi // cfg['images_per_block'], **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
        bases = {c: ap_.load_image_canvas(data / 'images' / names[image_id], c) for c in (320, 640)}
        units = [(arm, None) for arm in CLEAN] + [(arm, s) for arm in PATCHED for s in cfg['seeds']]
        rng.shuffle(units)
        for arm, seed in units:
            _, canvas, location, size = ARMS[arm]
            base = bases[canvas]
            loc = None
            if seed is not None:
                loc = ap_.find_locations([base], size, location, (seed if location != 'saliency' else 0) + image_id, canvas=canvas)[0]
            img = build_input(arm, base, patches.get((arm, seed)), loc, ap_.insert)
            with torch.no_grad():
                raw = model(img)
            raw = raw[0] if isinstance(raw, (tuple, list)) else raw
            decoded = pv.decode(raw[0])
            n_by_tau = {t: int((decoded[:, 4] > t).sum()) for t in cfg['tau_list']}
            if arm in ('clean_a', 'clean_b') and image_id in clean_hist and n_by_tau[0.01] != clean_hist[image_id]['active_candidates']:
                w.write(dict(kind='abort', reason='N lệch bản ghi lịch sử', image_id=image_id, arm=arm, got=n_by_tau[0.01]))
                raise SystemExit(f'lệch N: ảnh {image_id} {arm}')
            base_row = dict(profile=a.profile, image_id=image_id, arm=arm, seed=seed, canvas=canvas, location=loc,
                            block=bi // cfg['images_per_block'], n_by_tau={str(t): n for t, n in n_by_tau.items()})
            for tau in cfg['tau_list']:
                active, _ = pv.tau_filter(decoded, tau, None)
                boxes, scores, classes = active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()
                order = orders[order_k]
                for rep in range(-cfg['micro_warmup'], cfg['micro_repeats']):
                    for name in order:
                        b = BACKENDS[name]
                        tt = timing.timed(pv.tau_filter, setup=lambda: (decoded, tau, None))[0]
                        tn = timing.timed(b.run, setup=lambda: (b.prepare(boxes, scores, classes), cfg['iou']))[0]
                        if rep >= 0:
                            w.write(dict(kind='micro', **base_row, tau=tau, rep=rep, backend=name, nms_input=len(scores),
                                         kept=len(tn['result']), tau_ns=tt['ns'], nms_ns=tn['ns'], wall_utc=tn['wall_utc']))
                order_k += 1
            for rep in range(-cfg['pipeline_warmup'], cfg['pipeline_repeats']):
                for name in porders[order_k]:
                    r = timing.timed(pipeline_fn(model, img, BACKENDS[name], cfg['iou'], pv, torch))[0]
                    if rep >= 0:
                        w.write(dict(kind='pipeline', **base_row, rep=rep, backend=name, total_ns=r['ns'], wall_utc=r['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.close()
    print('[E11] DONE', flush=True)


if __name__ == '__main__':
    main()
