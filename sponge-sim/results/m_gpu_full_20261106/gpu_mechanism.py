"""Cơ chế: chi phí NMS trên ảnh bị tấn công với (a) ultralytics non_max_suppression mặc định,
(b) chặn ứng viên trước NMS (max_nms=300), (c) torchvision.ops.batched_nms (nhân C++ nhanh).
Trả lời: 126 ms là thật hay do cài đặt ultralytics? Chặn tiền-NMS có cứu không? Ghi C:\\sponge\\gpu_mech.json."""
import json, time, statistics
import numpy as np, torch, torchvision
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ASSETS

DEV = 'cuda'; NC, MASK_DIM, CANVAS = 80, 32, 640; CONF, IOU = 0.25, 0.45; PRED_W = 4 + NC + MASK_DIM


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


def load_img(p):
    from PIL import Image
    im = Image.open(p).convert('RGB').resize((CANVAS, CANVAS))
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)[None] / 255.0).to(DEV)


def soft_count(preds):
    return torch.sigmoid((preds[:, 4:4 + NC, :].amax(dim=1) - CONF) / 0.02).sum()


def attack(net, image, steps=100, eps=16 / 255, alpha=2 / 255):
    d = torch.zeros_like(image, requires_grad=True)
    for _ in range(steps):
        loss = soft_count(find_preds(net((image + d).clamp(0, 1))))
        loss.backward()
        with torch.no_grad():
            d.add_(alpha * d.grad.sign()).clamp_(-eps, eps); d.data = (image + d).clamp(0, 1) - image; d.grad.zero_()
    return (image + d).clamp(0, 1).detach()


def med_ms(fn, reps=25, warmup=6):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter_ns(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter_ns() - t0) / 1e6)
    return round(statistics.median(ts), 3)


def tv_candidates(preds):
    p = preds[0].transpose(0, 1)                      # [A, 116]
    box, sc_all = p[:, :4], p[:, 4:4 + NC]
    sc, cls = sc_all.max(1)
    keep = sc > CONF
    b, s, c = box[keep], sc[keep], cls[keep]
    xy, wh = b[:, :2], b[:, 2:]
    boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    return boxes.contiguous(), s.contiguous(), c.contiguous()


def main():
    print('torch', torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0), flush=True)
    net = YOLO('yolov8n-seg.pt').model.eval().float().to(DEV)
    for pr in net.parameters():
        pr.requires_grad_(False)
    out = {}
    for path in sorted(ASSETS.glob('*.jpg'))[:2]:
        clean = load_img(path); adv = attack(net, clean)
        row = {}
        for tag, im in (('clean', clean), ('attacked', adv)):
            with torch.no_grad():
                preds = find_preds(net(im)).detach()
            ntau = int((preds[:, 4:4 + NC, :].amax(1) > CONF).sum())
            ultra = med_ms(lambda: non_max_suppression(preds.clone(), CONF, IOU, nc=NC, max_det=300))
            capped = med_ms(lambda: non_max_suppression(preds.clone(), CONF, IOU, nc=NC, max_det=300, max_nms=300))
            bx, s, c = tv_candidates(preds)
            tv = med_ms(lambda: torchvision.ops.batched_nms(bx, s, c, IOU))
            row[tag] = {'N_tau': ntau, 'ultra_nms_ms': ultra, 'ultra_capped_maxnms300_ms': capped,
                        'torchvision_nms_ms': tv, 'n_cand_tv': int(len(s))}
            print(path.name, tag, row[tag], flush=True)
        out[path.name] = row
    json.dump(out, open(r'C:\sponge\gpu_mech.json', 'w'), indent=1)
    print('WRITTEN', flush=True)


if __name__ == '__main__':
    main()
