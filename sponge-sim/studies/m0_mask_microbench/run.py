"""M0 — micro-benchmark chi phí dựng mask theo K (cổng go/no-go). Xem prereg/PREREG_M0_mask_microbench.md.

Đo công đoạn dựng mask của model phân đoạn như hàm của K (số đối tượng giữ lại) và `retina_masks`.
- Công đoạn dựng mask dùng `ultralytics.utils.ops.process_mask` (upsample=False, mask ở độ phân giải proto)
  và `process_mask_native` (retina: phóng lên full-res). Chi phí phụ thuộc HÌNH DẠNG tensor (K, proto, full-res),
  không phụ thuộc GIÁ TRỊ -> proto/hệ số/hộp tổng hợp tất định cho cùng thời gian như proto thật, và chạy được
  smoke/kiểm thử KHÔNG cần trọng số seg.
- total đo trong MỘT lần gọi `full()` = forward(ảnh) thật + dựng mask trên K tiêm. KHÔNG gồm decode/NMS (nhỏ, ~2-4%),
  nên mẫu số hơi nhỏ -> tỷ phần mask là CẬN TRÊN: NO-GO dưới ngưỡng là kết luận bảo thủ (xem PREREG, quy tắc quyết định).
- forward đo riêng (mẫu số tham chiếu). mask đo riêng (prepare ngoài vùng đo), như E9 đo total/nms riêng.

Điều phối: `run.py --run-dir DIR [--smoke]`; worker là subprocess ghim trước khi torch tạo thread (như E9).
Mọi phép đo thời gian chạy một mình (quy tắc 9). CHƯA tag, CHƯA chạy đầy đủ khi chưa duyệt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))

CONFIG = dict(
    experiment='M0', profile='P1_cpu2', min_effective=0.5,
    seg_weights=['yolov8n-seg.pt'],   # chính; thêm 'yolo11n-seg.pt' nếu có trọng số (trục model). YOLO26-seg/SAM: giai đoạn sau.
    canvas=640, mask_dim=32,           # proto spatial = canvas // 4 (stride proto của YOLO-seg)
    K=[0, 10, 50, 100, 300], baseline_k=10, retinas=[False, True],
    n_images=20, images_per_block=2,
    coeff_seed=5100, box_seed=5101, cell_order_seed=5103,
    warmup=1, repeats=5, baseline_copies=3,  # K=0 đo nhiều lần, rải trong mỗi ảnh (A/A của total)
    deadline_ms=100.0,
)
SMOKE = dict(n_images=2, repeats=2)


def make_masks(kmax: int, proto_hw: int, mask_dim: int, canvas: int, coeff_seed: int, box_seed: int):
    """Proto + hệ số mask + hộp tổng hợp, tất định theo seed. Hộp phủ nhiều diện tích (chi phí cắt phụ thuộc diện tích).
    Trả proto [mask_dim, proto_hw, proto_hw], coeffs [kmax, mask_dim], boxes [kmax, 4] (toạ độ canvas), SHA256."""
    import torch
    rc = np.random.default_rng(coeff_seed)
    protos = torch.from_numpy(rc.standard_normal((mask_dim, proto_hw, proto_hw)).astype(np.float32))
    coeffs = torch.from_numpy(rc.standard_normal((kmax, mask_dim)).astype(np.float32))
    rb = np.random.default_rng(box_seed)
    cx, cy = rb.uniform(.2, .8, kmax) * canvas, rb.uniform(.2, .8, kmax) * canvas
    side = rb.uniform(.05, .6, kmax) * canvas
    x1, y1 = np.clip(cx - side / 2, 0, canvas), np.clip(cy - side / 2, 0, canvas)
    x2, y2 = np.clip(cx + side / 2, 0, canvas), np.clip(cy + side / 2, 0, canvas)
    boxes = torch.from_numpy(np.stack([x1, y1, x2, y2], 1).astype(np.float32))
    sha = hashlib.sha256(protos.numpy().tobytes() + coeffs.numpy().tobytes() + boxes.numpy().tobytes()).hexdigest()
    return protos, coeffs, boxes, sha


def mask_fn(retina: bool):
    """Hàm dựng mask đúng hai tuỳ chọn triển khai của Ultralytics."""
    from ultralytics.utils import ops
    if retina:   # retina_masks=True: mask full-res (đòn bẩy chính)
        return lambda protos, mc, boxes, shape: ops.process_mask_native(protos, mc, boxes, shape)
    return lambda protos, mc, boxes, shape: ops.process_mask(protos, mc, boxes, shape, upsample=False)


def image_ids(n):
    spec = importlib.util.spec_from_file_location('e5run', SIM / 'studies' / 'e5_protocol' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IMAGE_IDS[:n]


def seg_weight_sha(cfg):
    from sponge_sim import manifest
    from sponge_sim.cohorts import REPO
    out = {}
    for w in cfg['seg_weights']:
        p = REPO / w
        if not p.exists():
            raise SystemExit(f'Thiếu trọng số seg: {p}. Tải về trước (thao tác mạng, người dùng tự làm — quy tắc 7).')
        out[w] = manifest.sha256(p)
    return out


def worker(run_dir, cfg):
    from sponge_sim import hw
    cpus = hw.profiles(hw.cpu_topology())[cfg['profile']]
    readback = hw.pin(cpus)
    import torch
    torch.set_num_interop_threads(1)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    from ultralytics import YOLO
    from sponge_sim import timing
    from sponge_sim import attacks_provenance as ap_
    from sponge_sim.cohorts import REPO
    canvas, proto_hw = cfg['canvas'], cfg['canvas'] // 4
    shape = (canvas, canvas)
    protos, coeffs, boxes, msha = make_masks(max(cfg['K']), proto_hw, cfg['mask_dim'], canvas, cfg['coeff_seed'], cfg['box_seed'])
    fns = {r: mask_fn(r) for r in cfg['retinas']}
    w = timing.JsonlWriter(run_dir / f'raw_{cfg["profile"]}.jsonl')
    w.write(dict(kind='pin', profile=cfg['profile'], allowed=cpus, readback=readback, verify=verify,
                 torch_threads=torch.get_num_threads(), mask_sha256=msha, seg_weights_sha256=cfg['seg_weights_sha256']))
    if not verify['passed']:
        w.close()
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_M0)')
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    ids = image_ids(cfg['n_images'])
    # cell = (model, K, retina); K=0 nhân baseline_copies (A/A của total)
    base_cells = []
    for model_name in cfg['seg_weights']:
        for retina in cfg['retinas']:
            for k in cfg['K']:
                for copy in range(cfg['baseline_copies'] if k == 0 else 1):
                    base_cells.append(dict(model=model_name, k=k, retina=retina, copy=copy))
    ref0 = timing.reference_ms()
    block = -1
    for model_name in cfg['seg_weights']:
        net = YOLO(str(REPO / model_name)).model.eval().float()
        for ii, image_id in enumerate(ids):
            if ii % cfg['images_per_block'] == 0:
                block += 1
                w.write(dict(kind='reference', block=block, model=model_name,
                             **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
            image = ap_.load_image_canvas(data / 'images' / names[image_id], canvas)
            cs = [c for c in base_cells if c['model'] == model_name]
            random.Random(cfg['cell_order_seed'] + image_id).shuffle(cs)
            for g in cs:
                k, retina, fn = g['k'], g['retina'], fns[g['retina']]
                mc, bx = coeffs[:k].contiguous(), boxes[:k].contiguous()

                def full():
                    with torch.no_grad():
                        raw = net(image)
                    _ = raw[0] if isinstance(raw, (tuple, list)) else raw  # forward thật; đầu ra bỏ, mask dùng K tiêm
                    return fn(protos, mc, bx, shape)
                def fwd():
                    with torch.no_grad():
                        raw = net(image)
                    return raw[0] if isinstance(raw, (tuple, list)) else raw
                for rep in range(-cfg['warmup'], cfg['repeats']):
                    tt = timing.timed(full)[0]
                    tm = timing.timed(fn, setup=lambda: (protos, mc, bx, shape))[0]
                    tf = timing.timed(fwd)[0]
                    if rep < 0:
                        continue
                    w.write(dict(kind='cell', profile=cfg['profile'], block=block, model=model_name,
                                 image_id=image_id, rep=rep, k=k, retina=retina, copy=g['copy'],
                                 total_ns=tt['ns'], mask_ns=tm['ns'], forward_ns=tf['ns'],
                                 miss=tt['ns'] / 1e6 > cfg['deadline_ms'], wall_utc=tt['wall_utc']))
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
    cfg['seg_weights_sha256'] = seg_weight_sha(cfg)
    if a.worker:
        worker(run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)  # không ghi đè run cũ
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg))
    t0 = time.time()
    cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker'] + (['--smoke'] if a.smoke else [])
    subprocess.run(cmd, check=True)
    print(f'[M0] xong sau {time.time() - t0:.1f}s', flush=True)
    print('[M0] DONE', flush=True)


if __name__ == '__main__':
    main()
