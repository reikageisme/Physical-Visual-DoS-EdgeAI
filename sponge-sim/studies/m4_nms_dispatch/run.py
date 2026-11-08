"""Paired NMS backend diagnostic; see PREREG_M4_nms_dispatch.md.

Existing PGD is reused verbatim from gpu_mechanism.py:attack via importlib;
its full-file SHA256 is recorded. No timing from historical runs is overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('C:/sponge'))
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--provenance', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import nms as unms, ops
    dispatch_before = 'torchvision' in sys.modules
    import torchvision
    import numpy as np
    import psutil

    cfg = dict(experiment='M4', smoke=a.smoke, n_images=2 if a.smoke else 20,
               repeats=3 if a.smoke else 15, warmup=2 if a.smoke else 5,
               seed=20261106, conf=.25, iou=.45, max_det=300, nc=80, canvas=640,
               steps=100, eps=16/255, alpha=2/255,
               mask_proto_upsample=False, nms_cap=30000, device='cuda')
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    paths = sorted((a.root / 'datasets/coco128-seg/images/train2017').glob('*.jpg'))[:cfg['n_images']]
    assert len(paths) == cfg['n_images'], 'dataset incomplete'
    weights = a.root / 'yolov8n-seg.pt'
    source = a.root / 'gpu_mechanism.py'
    spec = importlib.util.spec_from_file_location('historical_mechanism', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    net = YOLO(str(weights)).model.eval().float().cuda()
    for p in net.parameters():
        p.requires_grad_(False)
    a.run_dir.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert provenance['script_sha256'] == sha(__file__), 'code differs from tag'
    assert provenance['historical_source_sha256'] == sha(source), 'historical attack source differs'
    manifest = dict(config=cfg, provenance=provenance, script_sha256=sha(__file__),
                    historical_source_sha256=sha(source), weights_sha256=sha(weights),
                    images={p.name: sha(p) for p in paths}, torch=torch.__version__,
                    torchvision=torchvision.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), affinity=psutil.Process().cpu_affinity(),
                    threads=torch.get_num_threads(), dispatch_before_torchvision_import=dispatch_before,
                    ultralytics_nms_sha256=sha(inspect.getfile(unms)),
                    config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    (a.run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (a.run_dir / 'ultralytics_nms_source.py').write_text(inspect.getsource(unms), encoding='utf-8')
    fh = (a.run_dir / 'raw.jsonl').open('x', encoding='utf-8')

    def write(**row):
        fh.write(json.dumps(dict(wall_utc=time.time(), **row)) + '\n')
        fh.flush()

    def timed(fn, setup=lambda: ()):
        args = setup()  # cloning is outside timing in EVERY condition
        was = gc.isenabled()
        gc.disable()
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            result = fn(*args)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
        finally:
            if was:
                gc.enable()
        return t1-t0, t0, result

    tv_nms = torchvision.ops.nms
    def suppress(preds, backend):
        # Explicit substitution preserves the SAME Ultralytics preprocessing,
        # class offsets, max_nms, and output slicing. Restore even on error.
        torchvision.ops.nms = tv_nms if backend == 'tv' else unms.TorchNMS.nms
        try:
            return unms.non_max_suppression(preds, .25, .45, nc=80, max_det=300)[0]
        finally:
            torchvision.ops.nms = tv_nms

    def proto(raw):
        return next(t for t in old.flat(raw) if t.dim() == 4 and t.shape[1] == 32)[0]

    def full(im, retina):
        raw = net(im)
        det = suppress(old.find_preds(raw), 'tv')
        return mask(proto(raw), det, retina)

    def mask(pr, det, retina):
        if not len(det):
            return None
        if retina:
            return ops.process_mask_native(pr, det[:, 6:], det[:, :4], (640, 640))
        return ops.process_mask(pr, det[:, 6:], det[:, :4], (640, 640), upsample=False)

    ref_image = old.load_img(paths[0])
    with torch.no_grad():
        for _ in range(10):
            net(ref_image)
    ref0 = None
    def reference(block):
        nonlocal ref0
        with torch.no_grad():
            samples = [timed(lambda: net(ref_image))[0] for _ in range(9)]
        median = float(np.median(samples))
        ref0 = median if ref0 is None else ref0
        write(kind='reference', block=block, samples_ns=samples, ratio=median/ref0,
              flagged=abs(median/ref0-1) > .03)

    reference('start')
    for ix, path in enumerate(paths):
        reference(ix)
        clean = old.load_img(path)
        adv = old.attack(net, clean, steps=100)
        torch.save(dict(clean=clean.cpu(), attacked=adv.cpu()), a.run_dir / f'{path.stem}_inputs.pt')
        write(kind='inputs', image=path.name, sha256=sha(a.run_dir / f'{path.stem}_inputs.pt'))
        jobs = []
        for arm, im in [('clean', clean), ('attacked', adv)]:
            with torch.no_grad():
                raw = net(im)
                pred = old.find_preds(raw).detach()
                pr = proto(raw).detach()
                dt = suppress(pred.clone(), 'tv')
                dp = suppress(pred.clone(), 'pure')
                write(kind='counts', image=path.name, arm=arm,
                      n_tau=int((pred[:, 4:84].amax(1) > .25).sum()),
                      k_tv=len(dt), k_pure=len(dp), outputs_equal=bool(torch.equal(dt, dp)),
                      prediction_sha256=hashlib.sha256(pred.cpu().numpy().tobytes()).hexdigest())
            for backend in ('tv', 'pure'):
                jobs.append((arm, f'nms_{backend}', lambda p, b=backend: suppress(p, b), lambda p=pred: (p.clone(),)))
            for retina in (False, True):
                jobs.append((arm, f'mask_{retina}', lambda p=pr, d=dt, r=retina: mask(p, d, r), lambda: ()))
                jobs.append((arm, f'total_tv_{retina}', lambda x=im, r=retina: full(x, r), lambda: ()))
            jobs.append((arm, 'forward', lambda x=im: net(x), lambda: ()))
            for label in ('A', 'B'):
                jobs.append((arm, f'aa_total_tv_{label}', lambda x=im: full(x, True), lambda: ()))
        with torch.no_grad():
            for _, _, fn, setup in jobs:
                for _ in range(cfg['warmup']):
                    fn(*setup())
            for rep in range(cfg['repeats']):
                order = list(range(len(jobs)))
                random.Random(cfg['seed'] + 100*ix + rep).shuffle(order)
                for pos, j in enumerate(order):
                    arm, stage, fn, setup = jobs[j]
                    ns, start, _ = timed(fn, setup)
                    write(kind='timing', image=path.name, block=ix, arm=arm, stage=stage,
                          rep=rep, position=pos, ns=ns, t_start_ns=start)
        print(f'IMAGE_DONE {ix+1}/{len(paths)} {path.name}', flush=True)
    reference('end')
    write(kind='done')
    fh.close()
    print('M4_DONE', flush=True)


if __name__ == '__main__':
    main()
