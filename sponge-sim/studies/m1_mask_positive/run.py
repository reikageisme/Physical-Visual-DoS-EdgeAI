"""M1 — đối chứng dương đầy đủ: độ trễ toàn trình theo K đối tượng giữ lại. Xem prereg/PREREG_M1_mask_positive_control.md.

Khác M0: total = forward thật + decode+NMS THẬT (ops.non_max_suppression, tự suy nm=32 từ bề rộng tensor) + dựng mask trên K tiêm.
Chạy NMS thật của Ultralytics để có chi phí decode+NMS thật, đầu ra bị bỏ; chỉ công đoạn dựng mask dùng K tiêm (hộp + 32 hệ số
tổng hợp, tất định). Proto tổng hợp cùng hình dạng proto thật nên cùng thời gian. KHÔNG tự viết lại decode seg.

make_masks/mask_fn/image_ids: copy từ studies/m0_mask_microbench/run.py (04/11/2026) để M1 tự chứa.
Điều phối như E9/M0: worker subprocess ghim trước khi torch tạo thread. Mọi phép đo chạy một mình (quy tắc 9).
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
    experiment='M1', profile='P1_cpu2', min_effective=0.5,
    seg_weights=['yolov8n-seg.pt'],   # thêm 'yolo11n-seg.pt' nếu có trọng số (trục model)
    canvas=640, mask_dim=32, nc=80, conf=0.25, iou=0.45, max_det=300,
    K=[0, 10, 50, 100, 300], baseline_k=10, retinas=[False, True],
    n_images=50, images_per_block=2, cooldown_s=0,   # M1b: nghỉ giữa block hạ nhiệt (ngoài vùng đo), chống throttle
    device='cpu',                                    # M2: 'cuda' cho trục thiết bị GPU (đồng bộ CUDA quanh vùng đo)
    coeff_seed=6100, box_seed=6101, cell_order_seed=6103,
    warmup=1, repeats=5, baseline_copies=3, deadline_ms=100.0,
)
SMOKE = dict(n_images=2, repeats=2)


def make_masks(kmax, proto_hw, mask_dim, canvas, coeff_seed, box_seed):
    """COPY từ m0_mask_microbench/run.py:make_masks. Proto+hệ số+hộp tổng hợp tất định."""
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


def mask_fn(retina):
    """COPY từ m0_mask_microbench/run.py:mask_fn."""
    from ultralytics.utils import ops
    if retina:
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
            raise SystemExit(f'Thiếu trọng số seg: {p}. Tải về trước (quy tắc 7).')
        out[w] = manifest.sha256(p)
    return out


def worker(run_dir, cfg):
    import psutil
    from sponge_sim import hw
    device = cfg.get('device', 'cpu')
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('device=cuda nhưng CUDA không khả dụng trên máy này')
    torch.set_num_interop_threads(1)
    profs = hw.profiles(hw.cpu_topology())
    cpus = profs.get(cfg['profile']) or sorted(psutil.Process().cpu_affinity())  # GPU box có thể không có profile này
    readback = hw.pin(cpus)
    verify = hw.verify_pin(cpus, min_effective=cfg['min_effective'])
    torch.set_num_threads(len(cpus))
    sync = torch.cuda.synchronize if device == 'cuda' else None   # chặn đồng hồ tới khi kernel CUDA xong
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression   # 8.4.90: NMS ở utils.nms, process_mask ở utils.ops
    from sponge_sim import timing
    from sponge_sim import attacks_provenance as ap_
    from sponge_sim.cohorts import REPO
    canvas, proto_hw = cfg['canvas'], cfg['canvas'] // 4
    shape = (canvas, canvas)
    protos, coeffs, boxes, msha = make_masks(max(cfg['K']), proto_hw, cfg['mask_dim'], canvas, cfg['coeff_seed'], cfg['box_seed'])
    protos, coeffs, boxes = protos.to(device), coeffs.to(device), boxes.to(device)
    fns = {r: mask_fn(r) for r in cfg['retinas']}
    nms = lambda preds: non_max_suppression(preds, cfg['conf'], cfg['iou'], nc=cfg['nc'], max_det=cfg['max_det'])
    pred_width = 4 + cfg['nc'] + cfg['mask_dim']
    w = timing.JsonlWriter(run_dir / f'raw_{cfg["profile"]}.jsonl')
    w.write(dict(kind='pin', profile=cfg['profile'], device=device, allowed=cpus, readback=readback, verify=verify,
                 torch_threads=torch.get_num_threads(), mask_sha256=msha, seg_weights_sha256=cfg['seg_weights_sha256']))
    if device == 'cpu' and not verify['passed']:   # trên GPU, ghim lõi host không quyết định -> chỉ ghi, không dừng
        w.close()
        raise SystemExit('verify ghim FAIL -> dừng (PREREG_M1)')
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    ids = image_ids(cfg['n_images'])
    base_cells = []
    for model_name in cfg['seg_weights']:
        for retina in cfg['retinas']:
            for k in cfg['K']:
                for copy in range(cfg['baseline_copies'] if k == 0 else 1):
                    base_cells.append(dict(model=model_name, k=k, retina=retina, copy=copy))
    ref0 = timing.reference_ms()
    block = -1
    for model_name in cfg['seg_weights']:
        net = YOLO(str(REPO / model_name)).model.eval().float().to(device)
        for ii, image_id in enumerate(ids):
            if ii % cfg['images_per_block'] == 0:
                if ii > 0 and cfg.get('cooldown_s'):
                    time.sleep(cfg['cooldown_s'])   # hạ nhiệt NGOÀI vùng đo; reference kế tiếp đọc trạng thái đã nguội
                block += 1
                w.write(dict(kind='reference', block=block, model=model_name, cooldown_s=cfg.get('cooldown_s', 0),
                             **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
            image = ap_.load_image_canvas(data / 'images' / names[image_id], canvas).to(device)
            with torch.no_grad():   # preds thật cho phép đo decode+NMS riêng (dựng ngoài vùng đo)
                raw0 = net(image)
            preds0 = _find_preds(raw0, pred_width).detach()
            n_real = int(len(nms(preds0.clone())[0]))
            cs = [c for c in base_cells if c['model'] == model_name]
            random.Random(cfg['cell_order_seed'] + image_id).shuffle(cs)
            for g in cs:
                k, retina, fn = g['k'], g['retina'], fns[g['retina']]
                mc, bx = coeffs[:k].contiguous(), boxes[:k].contiguous()

                def full():
                    with torch.no_grad():
                        raw = net(image)
                    preds = _find_preds(raw, pred_width)
                    _ = nms(preds)                       # decode+NMS thật, đầu ra bỏ
                    return fn(protos, mc, bx, shape)      # dựng mask trên K tiêm
                for rep in range(-cfg['warmup'], cfg['repeats']):
                    tt = timing.timed(full, sync=sync)[0]
                    tf = timing.timed(lambda im: _fwd(net, im, torch), setup=lambda: (image,), sync=sync)[0]
                    tn = timing.timed(nms, setup=lambda: (preds0.clone(),), sync=sync)[0]
                    tm = timing.timed(fn, setup=lambda: (protos, mc, bx, shape), sync=sync)[0]
                    if rep < 0:
                        continue
                    w.write(dict(kind='cell', profile=cfg['profile'], device=device, block=block, model=model_name,
                                 image_id=image_id, rep=rep, k=k, retina=retina, copy=g['copy'], n_real_kept=n_real,
                                 total_ns=tt['ns'], forward_ns=tf['ns'], nms_ns=tn['ns'], mask_ns=tm['ns'],
                                 miss=tt['ns'] / 1e6 > cfg['deadline_ms'], wall_utc=tt['wall_utc']))
    w.write(dict(kind='reference', block='end', **timing.drift(timing.reference_ms(), ref0), wall_utc=time.time()))
    w.write(dict(kind='pin_end', verify=hw.verify_pin(cpus, min_effective=cfg['min_effective'])))
    w.close()


def _flat(o):
    import torch
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, (tuple, list)):
        for e in o:
            yield from _flat(e)
    elif isinstance(o, dict):
        for e in o.values():
            yield from _flat(e)


def _find_preds(raw, width):
    """YOLOv8-seg forward trả cấu trúc lồng: raw[0]=(preds[1,W,A], proto[1,32,mh,mw]), raw[1]=dict.
    Lấy tensor preds [1, W, A] với W = 4 + nc + mask_dim (bền với thay đổi bố cục)."""
    for t in _flat(raw):
        if t.dim() == 3 and t.shape[1] == width:
            return t
    raise RuntimeError(f'không tìm thấy tensor preds [1,{width},A] trong đầu ra forward')


def _fwd(net, image, torch):
    with torch.no_grad():
        raw = net(image)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--n-images', type=int, default=None)      # M1b: giảm số ảnh
    ap.add_argument('--cooldown-s', type=float, default=None)  # M1b: nghỉ giữa block (giây)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default=None)  # M2: trục thiết bị
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = {**CONFIG, **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    if a.n_images is not None:
        cfg['n_images'] = a.n_images
    if a.cooldown_s is not None:
        cfg['cooldown_s'] = a.cooldown_s
    if a.device is not None:
        cfg['device'] = a.device
    cfg['seg_weights_sha256'] = seg_weight_sha(cfg)
    if a.worker:
        worker(run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg))
    t0 = time.time()
    extra = (['--smoke'] if a.smoke else []) + (['--n-images', str(a.n_images)] if a.n_images is not None else []) \
        + (['--cooldown-s', str(a.cooldown_s)] if a.cooldown_s is not None else []) \
        + (['--device', a.device] if a.device is not None else [])
    cmd = [sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker'] + extra
    subprocess.run(cmd, check=True)
    print(f'[M1] xong sau {time.time() - t0:.1f}s', flush=True)
    print('[M1] DONE', flush=True)


if __name__ == '__main__':
    main()
