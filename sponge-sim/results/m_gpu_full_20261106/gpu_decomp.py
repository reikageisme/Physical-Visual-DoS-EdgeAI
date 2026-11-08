"""Phân rã cơ chế: trên ảnh clean vs ảnh bị tấn công, đo RIÊNG forward / NMS / mask, và đếm ứng viên trước NMS (N_tau) + K sau NMS.
Trả lời: phần chậm 16-20x do NMS (ngập ứng viên) hay do mask? Ghi C:\\sponge\\gpu_decomp.json."""
import json, time, statistics
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops, ASSETS

DEV = 'cuda'
NC, MASK_DIM, CANVAS = 80, 32, 640
CONF, IOU, MAXDET = 0.25, 0.45, 300
PRED_W = 4 + NC + MASK_DIM


def flat(o):
    if torch.is_tensor(o):
        yield o
    elif isinstance(o, (tuple, list)):
        for e in o:
            yield from flat(e)
    elif isinstance(o, dict):
        for e in o.values():
            yield from flat(e)


def find_preds(raw):
    for t in flat(raw):
        if t.dim() == 3 and t.shape[1] == PRED_W:
            return t


def find_proto(raw):
    for t in flat(raw):
        if t.dim() == 4 and t.shape[1] == MASK_DIM:
            return t


def nms(preds):
    return non_max_suppression(preds, CONF, IOU, nc=NC, max_det=MAXDET)


def load_img(path):
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((CANVAS, CANVAS))
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)[None] / 255.0).to(DEV)


def soft_count(preds):
    return torch.sigmoid((preds[:, 4:4 + NC, :].amax(dim=1) - CONF) / 0.02).sum()


def attack(net, image, steps=100, eps=16 / 255, alpha=2 / 255):
    delta = torch.zeros_like(image, requires_grad=True)
    for s in range(steps):
        x = (image + delta).clamp(0, 1)
        loss = soft_count(find_preds(net(x)))
        loss.backward()
        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign()).clamp_(-eps, eps)
            delta.data = (image + delta).clamp(0, 1) - image
            delta.grad.zero_()
    return (image + delta).clamp(0, 1).detach()


def med_ms(fn, reps=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter_ns(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(ts)


def decompose(net, img, tag):
    with torch.no_grad():
        raw = net(img)
        preds0 = find_preds(raw).detach()
        proto0 = find_proto(raw).detach()
        dets = nms(preds0.clone())[0]
        ntau = int((preds0[:, 4:4 + NC, :].amax(dim=1) > CONF).sum().item())
        k = int(len(dets))
        mc, bx = dets[:, 6:6 + MASK_DIM].contiguous(), dets[:, :4].contiguous()
    fwd = med_ms(lambda: net(img))
    nmst = med_ms(lambda: nms(preds0.clone()))
    if k:
        maskf = med_ms(lambda: ops.process_mask(proto0[0], mc, bx, (CANVAS, CANVAS), upsample=False))
        maskt = med_ms(lambda: ops.process_mask_native(proto0[0], mc, bx, (CANVAS, CANVAS)))
    else:
        maskf = maskt = 0.0
    r = {'N_tau': ntau, 'K': k, 'forward_ms': round(fwd, 3), 'nms_ms': round(nmst, 3),
         'mask_proto_ms': round(maskf, 3), 'mask_full_ms': round(maskt, 3)}
    print(tag, r, flush=True)
    return r


def main():
    print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)
    net = YOLO('yolov8n-seg.pt').model.eval().float().to(DEV)
    for p in net.parameters():
        p.requires_grad_(False)
    out = {}
    for path in sorted(ASSETS.glob('*.jpg'))[:2]:
        clean = load_img(path)
        adv = attack(net, clean)
        out[path.name] = {'clean': decompose(net, clean, path.name + ' CLEAN'),
                          'attacked': decompose(net, adv, path.name + ' ATTACKED')}
    json.dump(out, open(r'C:\sponge\gpu_decomp.json', 'w'), indent=1)
    print('WRITTEN C:\\sponge\\gpu_decomp.json', flush=True)


if __name__ == '__main__':
    main()
