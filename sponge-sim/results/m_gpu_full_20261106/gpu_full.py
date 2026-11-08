"""Bộ đầy đủ GPU giai đoạn 1: M1-GPU chặt (A/A+MDE+K*) + M3 đa ảnh, cho 2 model seg. Ghi C:\\sponge\\gpu_full.json."""
import json, time, statistics
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils import ops

DEV = 'cuda'
NC, MASK_DIM, CANVAS = 80, 32, 640
CONF, IOU, MAXDET = 0.25, 0.45, 300
PRED_W = 4 + NC + MASK_DIM
MODELS = ['yolov8n-seg.pt', 'yolo11n-seg.pt']
KS = [0, 10, 50, 100, 300]
BASE_K = 10
RNG = np.random.default_rng(20261106)


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
    raise RuntimeError('no preds')


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


def block_medians(fn, blocks=12, reps=15, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(blocks):
        xs = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter_ns(); fn(); torch.cuda.synchronize()
            xs.append(time.perf_counter_ns() - t0)
        out.append(statistics.median(xs) / 1e6)
    return out  # list các trung vị khối (ms)


def boot_ci(x, stat=np.median, n=5000, seed=1):
    x = np.asarray(x); r = np.random.default_rng(seed)
    v = [stat(x[r.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def mde_paired(sd, n, a=.05, pw=.8):
    from scipy import stats as sps
    df = n - 1
    return float((sps.t.ppf(1 - a / 2, df) + sps.t.ppf(pw, df)) * sd / np.sqrt(n))


def m1_gpu(net):
    img = torch.rand(1, 3, CANVAS, CANVAS, device=DEV)
    protos, coeffs, boxes = make_masks(300)
    res = {'stages_ms': {}, 'total_ratio': {}, 'frac_mask': {}, 'mde_total_pct': {}, 'k_star': {}}
    with torch.no_grad():
        for retina in (False, True):
            fn = mask_fn(retina)
            blockcache = {}
            for k in KS:
                mc, b = coeffs[:k].contiguous(), boxes[:k].contiguous()
                def full():
                    preds = find_preds(net(img)); nms(preds); fn(protos, mc, b)
                def mk():
                    fn(protos, mc, b)
                bm = block_medians(full)
                blockcache[k] = bm
                res['stages_ms'][f'r{retina}|K{k}'] = {'total_med': statistics.median(bm),
                                                        'mask_med': statistics.median(block_medians(mk))}
            base = np.asarray(blockcache[BASE_K])
            # A/A: tách đôi các khối baseline, log tỷ số trung vị -> MDE
            aa = []
            for _ in range(200):
                idx = RNG.permutation(len(base))
                a1, a2 = base[idx[:len(base) // 2]], base[idx[len(base) // 2:]]
                aa.append(np.log(np.median(a1) / np.median(a2)))
            mde = mde_paired(float(np.std(aa, ddof=1)), len(base))
            res['mde_total_pct'][f'r{retina}'] = mde * 100
            kstar = None
            for k in KS:
                if k <= BASE_K:
                    continue
                logs = np.log(np.asarray(blockcache[k]) / base)  # ghép theo khối
                ratio = float(np.exp(np.median(logs)))
                lo, hi = boot_ci(logs, np.median)
                ci = [float(np.exp(lo)), float(np.exp(hi))]
                det = ci[0] > 1 + mde
                res['total_ratio'][f'r{retina}|K{k}'] = {'ratio': ratio, 'ci95': ci, 'detected': det}
                res['frac_mask'][f'r{retina}|K{k}'] = res['stages_ms'][f'r{retina}|K{k}']['mask_med'] / res['stages_ms'][f'r{retina}|K{k}']['total_med']
                if det and kstar is None:
                    kstar = k
            res['k_star'][f'r{retina}'] = kstar if kstar is not None else '>300'
    return res


def coco_images(n):
    from ultralytics.data.utils import check_det_dataset
    from pathlib import Path
    d = check_det_dataset('coco128-seg.yaml')
    base = Path(d['path'])
    imgs = sorted((base / 'images' / 'train2017').glob('*.jpg'))[:n]
    return imgs


def load_img(path):
    from PIL import Image
    im = Image.open(path).convert('RGB').resize((CANVAS, CANVAS))
    t = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1)[None] / 255.0
    return t.to(DEV)


def soft_count(preds):
    c = preds[:, 4:4 + NC, :].amax(dim=1)
    return torch.sigmoid((c - CONF) / 0.02).sum()


def m3(net, imgs, steps=100, eps=16 / 255, alpha=2 / 255):
    res = []
    for p in imgs:
        image = load_img(p)
        delta = torch.zeros_like(image, requires_grad=True)
        def kreal(x):
            with torch.no_grad():
                return int(len(nms(find_preds(net(x)))[0]))
        kc = kreal(image.clamp(0, 1)); best = kc
        for s in range(steps):
            x = (image + delta).clamp(0, 1)
            loss = soft_count(find_preds(net(x)))
            loss.backward()
            with torch.no_grad():
                delta.add_(alpha * delta.grad.sign()).clamp_(-eps, eps)
                delta.data = (image + delta).clamp(0, 1) - image
                delta.grad.zero_()
            if s % 10 == 0 or s == steps - 1:
                best = max(best, kreal((image + delta).clamp(0, 1)))
        res.append({'name': p.name, 'k_clean': kc, 'k_best': best})
    return res


def main():
    print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)
    imgs = coco_images(30)
    print('coco128 images:', len(imgs), flush=True)
    out = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(0), 'models': {}}
    for mname in MODELS:
        print('=== model', mname, flush=True)
        net = YOLO(mname).model.eval().float().to(DEV)
        for pr in net.parameters():
            pr.requires_grad_(False)
        t0 = time.time(); r1 = m1_gpu(net); print(' M1-GPU', round(time.time() - t0, 1), 's', flush=True)
        t1 = time.time(); r3 = m3(net, imgs); print(' M3', round(time.time() - t1, 1), 's', flush=True)
        ks = [i['k_best'] for i in r3]
        out['models'][mname] = {'m1_gpu': r1, 'm3': r3,
                                'm3_k_best_median': statistics.median(ks),
                                'm3_k_clean_median': statistics.median([i['k_clean'] for i in r3]),
                                'm3_frac_reach_kstar': sum(k >= (r1['k_star']['rTrue'] if isinstance(r1['k_star']['rTrue'], int) else 300) for k in ks) / len(ks),
                                'm3_frac_reach_100': sum(k >= 100 for k in ks) / len(ks),
                                'm3_frac_reach_300': sum(k >= 300 for k in ks) / len(ks)}
    with open(r'C:\sponge\gpu_full.json', 'w') as f:
        json.dump(out, f, indent=1)
    for mname, m in out['models'].items():
        print('===', mname, flush=True)
        print(' MDE%', {k: round(v, 3) for k, v in m['m1_gpu']['mde_total_pct'].items()}, 'K*', m['m1_gpu']['k_star'], flush=True)
        for k, v in m['m1_gpu']['total_ratio'].items():
            print('  ', k, 'ratio=%.3f' % v['ratio'], 'ci=[%.3f,%.3f]' % tuple(v['ci95']), 'det=%s' % v['detected'], flush=True)
        print(' M3 K_clean_med=%s K_best_med=%s reach100=%.2f reach300=%.2f' % (
            m['m3_k_clean_median'], m['m3_k_best_median'], m['m3_frac_reach_100'], m['m3_frac_reach_300']), flush=True)
    print('WRITTEN C:\\sponge\\gpu_full.json', flush=True)


if __name__ == '__main__':
    main()
