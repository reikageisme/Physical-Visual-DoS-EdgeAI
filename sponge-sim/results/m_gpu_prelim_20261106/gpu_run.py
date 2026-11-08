"""M1-GPU + M3 tự chứa cho máy thuê (RTX 5060 Ti). In JSON kết quả ra C:\\sponge\\gpu_results.json.

M1-GPU: đo forward + NMS thật + dựng mask theo K, ĐỒNG BỘ CUDA quanh vùng đo (kernel GPU bất đồng bộ).
  proto/hệ số/hộp tổng hợp (chi phí mask phụ thuộc hình dạng, không phụ thuộc giá trị); ảnh forward tổng hợp.
M3: PGD hộp trắng L∞ đẩy K (số đối tượng sau NMS) trên ảnh thật; hàm mục tiêu proxy khả vi, K đếm THẬT bằng NMS.
Logic khớp studies/m1_mask_positive + studies/m3_attack đã test trên CPU; đây là bản chạy GPU.
"""
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
WEIGHTS = 'yolov8n-seg.pt'


def sync():
    torch.cuda.synchronize()


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
    raise RuntimeError('no preds tensor')


def make_masks(kmax, proto_hw=CANVAS // 4):
    rc = np.random.default_rng(6100)
    protos = torch.from_numpy(rc.standard_normal((MASK_DIM, proto_hw, proto_hw)).astype(np.float32)).to(DEV)
    coeffs = torch.from_numpy(rc.standard_normal((kmax, MASK_DIM)).astype(np.float32)).to(DEV)
    rb = np.random.default_rng(6101)
    cx, cy = rb.uniform(.2, .8, kmax) * CANVAS, rb.uniform(.2, .8, kmax) * CANVAS
    side = rb.uniform(.05, .6, kmax) * CANVAS
    x1, y1 = np.clip(cx - side / 2, 0, CANVAS), np.clip(cy - side / 2, 0, CANVAS)
    x2, y2 = np.clip(cx + side / 2, 0, CANVAS), np.clip(cy + side / 2, 0, CANVAS)
    boxes = torch.from_numpy(np.stack([x1, y1, x2, y2], 1).astype(np.float32)).to(DEV)
    return protos, coeffs, boxes


def mask_fn(retina):
    if retina:
        return lambda p, mc, b: ops.process_mask_native(p, mc, b, (CANVAS, CANVAS))
    return lambda p, mc, b: ops.process_mask(p, mc, b, (CANVAS, CANVAS), upsample=False)


def nms(preds):
    return non_max_suppression(preds, CONF, IOU, nc=NC, max_det=MAXDET)


def timed(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(reps):
        sync(); t0 = time.perf_counter_ns(); fn(); sync(); out.append(time.perf_counter_ns() - t0)
    return statistics.median(out) / 1e6  # ms


def m1_gpu(net):
    img = torch.rand(1, 3, CANVAS, CANVAS, device=DEV)
    protos, coeffs, boxes = make_masks(300)
    res = {'device': torch.cuda.get_device_name(0), 'stages_ms': {}, 'frac_mask': {}, 'total_ratio': {}}
    with torch.no_grad():
        for retina in (False, True):
            fn = mask_fn(retina)
            base_total = None
            for k in (0, 10, 50, 100, 300):
                mc, b = coeffs[:k].contiguous(), boxes[:k].contiguous()
                def full():
                    preds = find_preds(net(img)); nms(preds); fn(protos, mc, b)
                def fwd():
                    find_preds(net(img))
                def nn():
                    preds = find_preds(net(img)); nms(preds)
                def mk():
                    fn(protos, mc, b)
                tot = timed(full); fw = timed(fwd); mkt = timed(mk)
                tag = f'retina{retina}|K{k}'
                res['stages_ms'][tag] = {'total': tot, 'forward': fw, 'mask': mkt}
                if k > 0:
                    res['frac_mask'][tag] = mkt / tot
                if k == 10:
                    base_total = tot
                if base_total:
                    res['total_ratio'][tag] = tot / base_total
    return res


def load_img(path):
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((CANVAS, CANVAS))
    t = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
    return t.to(DEV)


def soft_count(preds):
    c = preds[:, 4:4 + NC, :].amax(dim=1)
    return torch.sigmoid((c - CONF) / 0.02).sum()


def m3_attack(net, steps=100, eps=16 / 255, alpha=2 / 255):
    imgs = sorted(ASSETS.glob('*.jpg'))[:4]
    out = {'steps': steps, 'eps': eps, 'images': []}
    for p in imgs:
        image = load_img(p)
        delta = torch.zeros_like(image, requires_grad=True)
        def kreal(x):
            with torch.no_grad():
                return int(len(nms(find_preds(net(x)))[0]))
        k_clean = kreal(image.clamp(0, 1)); best = k_clean; traj = []
        for s in range(steps):
            x = (image + delta).clamp(0, 1)
            loss = soft_count(find_preds(net(x)))
            loss.backward()
            with torch.no_grad():
                delta.add_(alpha * delta.grad.sign()).clamp_(-eps, eps)
                delta.data = (image + delta).clamp(0, 1) - image
                delta.grad.zero_()
            if s % 10 == 0 or s == steps - 1:
                k = kreal((image + delta).clamp(0, 1)); traj.append([s, k]); best = max(best, k)
        out['images'].append({'name': p.name, 'k_clean': k_clean, 'k_best': best, 'traj': traj})
    ks = [i['k_best'] for i in out['images']]
    out['k_best_median'] = statistics.median(ks)
    out['frac_reach_50'] = sum(k >= 50 for k in ks) / len(ks)
    return out


def main():
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    net = YOLO(WEIGHTS).model.eval().float().to(DEV)
    for pr in net.parameters():
        pr.requires_grad_(False)
    t0 = time.time()
    r1 = m1_gpu(net); print('M1-GPU done', round(time.time() - t0, 1), 's', flush=True)
    t1 = time.time()
    r3 = m3_attack(net); print('M3 done', round(time.time() - t1, 1), 's', flush=True)
    result = {'m1_gpu': r1, 'm3': r3, 'torch': torch.__version__}
    with open(r'C:\sponge\gpu_results.json', 'w') as f:
        json.dump(result, f, indent=1)
    # tóm tắt
    print('--- M1-GPU frac_mask & total_ratio ---', flush=True)
    for k in r1['total_ratio']:
        fm = r1['frac_mask'].get(k)
        print(k, 'total_ratio=%.3f' % r1['total_ratio'][k], ('frac_mask=%.3f' % fm) if fm else '', flush=True)
    print('--- M3 attack (K_clean -> K_best) ---', flush=True)
    for i in r3['images']:
        print(i['name'], i['k_clean'], '->', i['k_best'], flush=True)
    print('M3 k_best_median', r3['k_best_median'], 'frac_reach_50', r3['frac_reach_50'], flush=True)
    print('RESULTS_WRITTEN C:\\sponge\\gpu_results.json', flush=True)


if __name__ == '__main__':
    main()
