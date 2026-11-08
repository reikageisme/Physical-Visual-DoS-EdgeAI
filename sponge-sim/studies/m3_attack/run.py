"""M3 — tấn công thật: PGD hộp trắng đẩy số đối tượng giữ lại K. Xem prereg/PREREG_M3_pgd_attack.md.

Mục tiêu: nhiễu đối kháng L∞ làm K (sau NMS) tăng gần max_det=300 trên YOLOv8n-seg. Bỏ GA.
Hàm mục tiêu khả vi là PROXY (số ứng viên mềm vượt ngưỡng); K kết luận luôn là số ĐẾM THẬT bằng non_max_suppression.
CẦN GPU để chạy đủ (device=cuda). Smoke CPU 1 ảnh/ít vòng chỉ kiểm cơ chế, KHÔNG kết luận hiệu lực.

KHUNG để chạy/tinh chỉnh trên GPU: proxy đơn giản (không phân tán IoU) có thể chưa đẩy đủ K — nếu vậy ghi kết quả âm và
thử proxy có số hạng phân tán (ghi DEVIATIONS). _find_preds/_flat/image_ids copy từ m1_mask_positive/run.py (06/11/2026).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM / 'src'))

CONFIG = dict(
    experiment='M3', device='cuda', seg_weights=['yolov8n-seg.pt'],
    canvas=640, nc=80, mask_dim=32, conf=0.25, iou=0.45, max_det=300,
    eps=16 / 255, alpha=2 / 255, steps=100, temp=0.02, eval_every=10,
    n_images=20, seed=7100, save_deltas=False,   # delta lớn (~4,7 MB/ảnh): mặc định không lưu, chỉ lưu số K
    k_star_ref=50,   # K* của M1 (full-res) để tính tỷ lệ ảnh đạt; mô tả
)
SMOKE = dict(n_images=1, steps=6, eval_every=2)


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
    for t in _flat(raw):
        if t.dim() == 3 and t.shape[1] == width:
            return t
    raise RuntimeError(f'không tìm thấy tensor preds [1,{width},A]')


def soft_candidate_count(preds, nc, conf, temp):
    """Proxy khả vi cho số ứng viên vượt ngưỡng: Σ sigmoid((max_điểm_lớp − τ)/T). Điểm lớp đã sigmoid trong forward."""
    import torch
    c = preds[:, 4:4 + nc, :].amax(dim=1)       # [1, A]
    return torch.sigmoid((c - conf) / temp).sum()


def image_ids(n):
    spec = importlib.util.spec_from_file_location('e5run', SIM / 'studies' / 'e5_protocol' / 'run.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.IMAGE_IDS[:n]


def attack_image(net, image, cfg, nms, pred_width):
    import torch
    eps, alpha = cfg['eps'], cfg['alpha']
    delta = torch.zeros_like(image, requires_grad=True)

    def k_real(x):
        with torch.no_grad():
            return int(len(nms(_find_preds(net(x), pred_width))[0]))
    k_clean = k_real(torch.clamp(image, 0, 1))
    best_k, best_delta, traj = k_clean, None, []
    for step in range(cfg['steps']):
        x = torch.clamp(image + delta, 0, 1)
        loss = soft_candidate_count(_find_preds(net(x), pred_width), cfg['nc'], cfg['conf'], cfg['temp'])
        loss.backward()
        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-eps, eps)
            delta.data = torch.clamp(image + delta, 0, 1) - image   # giữ ảnh hợp lệ [0,1]
            delta.grad.zero_()
        if step % cfg['eval_every'] == 0 or step == cfg['steps'] - 1:
            k = k_real(torch.clamp(image + delta, 0, 1))
            traj.append(dict(step=step, loss=float(loss.item()), k=k))
            if k > best_k:
                best_k, best_delta = k, delta.detach().clone()
    return dict(k_clean=k_clean, k_best=best_k, trajectory=traj), best_delta


def worker(run_dir, cfg):
    import torch
    from sponge_sim import timing
    from sponge_sim import attacks_provenance as ap_
    from sponge_sim.cohorts import REPO
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    device = cfg['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('device=cuda nhưng CUDA không khả dụng')
    torch.manual_seed(cfg['seed'])
    canvas = cfg['canvas']
    pred_width = 4 + cfg['nc'] + cfg['mask_dim']
    nms = lambda preds: non_max_suppression(preds, cfg['conf'], cfg['iou'], nc=cfg['nc'], max_det=cfg['max_det'])
    data = REPO / 'outputs/datasets/coco_val2017_extended'
    names = {im['id']: im['file_name'] for im in
             json.loads((data / 'instances_subset.json').read_text(encoding='utf-8'))['images']}
    ids = image_ids(cfg['n_images'])
    w = timing.JsonlWriter(run_dir / 'attacks.jsonl')
    deltas = {}
    for model_name in cfg['seg_weights']:
        net = YOLO(str(REPO / model_name)).model.eval().float().to(device)
        for p in net.parameters():
            p.requires_grad_(False)   # chỉ tối ưu nhiễu, không tối ưu trọng số
        for image_id in ids:
            image = ap_.load_image_canvas(data / 'images' / names[image_id], canvas).to(device)
            res, best_delta = attack_image(net, image, cfg, nms, pred_width)
            w.write(dict(kind='attack', model=model_name, image_id=image_id, device=device,
                         eps=cfg['eps'], alpha=cfg['alpha'], steps=cfg['steps'], temp=cfg['temp'], **res))
            if cfg['save_deltas'] and best_delta is not None:
                deltas[f'{model_name}|{image_id}'] = best_delta.cpu()
    if cfg['save_deltas'] and deltas:
        torch.save(deltas, run_dir / 'best_deltas.pt')   # để đưa vào M1/M2 đo toàn trình thật
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    ap.add_argument('--save-deltas', action='store_true')
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = {**CONFIG, **(SMOKE if a.smoke else {}), 'smoke': a.smoke}
    if a.device is not None:
        cfg['device'] = a.device
    if a.save_deltas:
        cfg['save_deltas'] = True
    if a.worker:
        worker(run_dir, cfg)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    from sponge_sim import manifest
    manifest.write(run_dir / 'manifest.json', manifest.build(cfg))
    t0 = time.time()
    extra = (['--smoke'] if a.smoke else []) + (['--device', a.device] if a.device else []) + (['--save-deltas'] if cfg['save_deltas'] else [])
    subprocess.run([sys.executable, '-X', 'utf8', __file__, '--run-dir', str(run_dir), '--worker'] + extra, check=True)
    print(f'[M3] xong sau {time.time() - t0:.1f}s', flush=True)
    print('[M3] DONE', flush=True)


if __name__ == '__main__':
    main()
