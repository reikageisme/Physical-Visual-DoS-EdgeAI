"""Phần còn lại: (A) đo độ trễ TOÀN TRÌNH THẬT clean vs ảnh bị tấn công (khép vòng attack->latency);
(B) thử YOLO26-seg (không NMS): tấn công có đẩy K không, mask có tăng không. Ghi C:\\sponge\\gpu_extra.json."""
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


def find_preds(raw, w=PRED_W):
    for t in flat(raw):
        if t.dim() == 3 and t.shape[1] == w:
            return t
    return None


def find_proto(raw):
    for t in flat(raw):
        if t.dim() == 4 and t.shape[1] == MASK_DIM:
            return t
    return None


def nms(preds):
    return non_max_suppression(preds, CONF, IOU, nc=NC, max_det=MAXDET)


def load_img(path):
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((CANVAS, CANVAS))
    return (torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)[None] / 255.0).to(DEV)


def soft_count(preds):
    c = preds[:, 4:4 + NC, :].amax(dim=1)
    return torch.sigmoid((c - CONF) / 0.02).sum()


def attack(net, image, steps=100, eps=16 / 255, alpha=2 / 255, w=PRED_W):
    delta = torch.zeros_like(image, requires_grad=True)
    for s in range(steps):
        x = (image + delta).clamp(0, 1)
        loss = soft_count(find_preds(net(x), w))
        loss.backward()
        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign()).clamp_(-eps, eps)
            delta.data = (image + delta).clamp(0, 1) - image
            delta.grad.zero_()
    return (image + delta).clamp(0, 1).detach()


def real_total_ms(net, img, retina, w=PRED_W, reps=30, warmup=8):
    """Toàn trình thật: forward + NMS + dựng mask trên detection THẬT (proto thật). K là số thật."""
    mfn = (lambda p, mc, b: ops.process_mask_native(p, mc, b, (CANVAS, CANVAS))) if retina \
        else (lambda p, mc, b: ops.process_mask(p, mc, b, (CANVAS, CANVAS), upsample=False))
    def pipe():
        with torch.no_grad():
            raw = net(img)
            preds, proto = find_preds(raw, w), find_proto(raw)
            dets = nms(preds)[0]
            if len(dets):
                mfn(proto[0], dets[:, 6:6 + MASK_DIM], dets[:, :4])
    for _ in range(warmup):
        pipe()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter_ns(); pipe(); torch.cuda.synchronize()
        ts.append((time.perf_counter_ns() - t0) / 1e6)
    with torch.no_grad():
        raw = net(img); k = int(len(nms(find_preds(raw, w))[0]))
    return statistics.median(ts), k


def partA(out):
    net = YOLO('yolov8n-seg.pt').model.eval().float().to(DEV)
    for p in net.parameters():
        p.requires_grad_(False)
    res = []
    for path in sorted(ASSETS.glob('*.jpg'))[:2]:
        clean = load_img(path)
        adv = attack(net, clean)
        row = {'name': path.name}
        for retina in (False, True):
            ct, ck = real_total_ms(net, clean, retina)
            at, ak = real_total_ms(net, adv, retina)
            row[f'retina{retina}'] = {'clean_ms': round(ct, 3), 'clean_K': ck, 'attacked_ms': round(at, 3),
                                      'attacked_K': ak, 'ratio': round(at / ct, 3)}
        res.append(row)
        print('A', path.name, res[-1], flush=True)
    out['attack_to_latency'] = res


def partB(out):
    info = {}
    model_name = None
    for cand in ['yolo26n-seg.pt', 'yoloe-26-seg.pt', 'yolov10n-seg.pt']:
        try:
            net = YOLO(cand).model.eval().float().to(DEV)
            model_name = cand
            break
        except Exception as e:
            info[cand] = 'unavailable: ' + repr(e)[:120]
    if model_name is None:
        out['nms_free'] = {'status': 'KHÔNG tải được model seg không-NMS', 'tried': info}
        print('B: no NMS-free seg model available', info, flush=True)
        return
    # dò bề rộng preds (nc + nm có thể khác)
    img = torch.rand(1, 3, CANVAS, CANVAS, device=DEV)
    with torch.no_grad():
        raw = net(img)
    preds = None
    for t in flat(raw):
        if t.dim() == 3 and t.shape[1] > NC:
            preds = t; break
    w = preds.shape[1] if preds is not None else PRED_W
    for p in net.parameters():
        p.requires_grad_(False)
    res = []
    for path in sorted(ASSETS.glob('*.jpg'))[:2]:
        clean = load_img(path)
        with torch.no_grad():
            kc = int(len(nms(find_preds(net(clean), w))[0])) if find_preds(net(clean), w) is not None else -1
        adv = attack(net, clean, w=w)
        with torch.no_grad():
            ka = int(len(nms(find_preds(net(adv), w))[0])) if find_preds(net(adv), w) is not None else -1
        res.append({'name': path.name, 'k_clean': kc, 'k_best': ka})
        print('B', model_name, res[-1], flush=True)
    out['nms_free'] = {'status': 'ok', 'model': model_name, 'pred_width': int(w), 'm3': res}


def main():
    print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)
    out = {}
    partA(out)
    try:
        partB(out)
    except Exception as e:
        out['nms_free'] = {'status': 'error', 'err': repr(e)[:200]}
        print('B error', repr(e)[:200], flush=True)
    json.dump(out, open(r'C:\sponge\gpu_extra.json', 'w'), indent=1)
    print('WRITTEN C:\\sponge\\gpu_extra.json', flush=True)


if __name__ == '__main__':
    main()
