"""Prospective M2 CPU/GPU synthetic mask workload confirmation. Standalone Windows worker."""
import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu','cuda'], required=True)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    import psutil
    cpus = [2] if a.threads == 1 else [2,3,4,5]
    if a.device == 'cpu':
        psutil.Process().cpu_affinity(cpus)
        assert psutil.Process().cpu_affinity() == cpus
    import torch
    import torchvision  # fixed compiled backend; import BEFORE Ultralytics calls
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils import ops
    root = Path('C:/sponge')
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261106)
    old = module('historical', root/'gpu_mechanism.py')
    fixture = module('fixture', root/'gpu_full.py')
    hw = module('hardware', root/'m2_hw.py')
    old.DEV = a.device
    fixture.DEV = a.device
    paths = sorted((root/'datasets/coco128-seg/images/train2017').glob('*.jpg'))[:2 if a.smoke else 20]
    cfg = dict(device=a.device,threads=a.threads,smoke=a.smoke,n_images=len(paths),
               repeats=2 if a.smoke else 5, K=[0,10,50,100,300], retinas=[False,True],baseline_k=10,
               coeff_seed=6100,box_seed=6101,canvas=640,nc=80,conf=.25,iou=.45,max_det=300,
               order_seed=20261106,warmup=2,steady_forward_warmup=50,mask_proto_upsample=False)
    prov = json.loads((root/'confirm_provenance.json').read_text(encoding='utf-8-sig'))
    assert sha(__file__) == prov['m2_sha256']
    a.run_dir.mkdir(parents=True,exist_ok=False)
    net = YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().to(a.device)
    pr,mc,bx = fixture.make_masks(300)
    sync = torch.cuda.synchronize if a.device=='cuda' else lambda: None
    manifest = dict(config=cfg,provenance=prov,weights_sha256=sha(root/'yolov8n-seg.pt'),
                    fixture_sha256=sha(root/'gpu_full.py'),historical_sha256=sha(root/'gpu_mechanism.py'),
                    topology=hw.cpu_topology(),power=hw.power_snapshot(),
                    images={p.name:sha(p) for p in paths},affinity=psutil.Process().cpu_affinity(),
                    torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda)
    (a.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fh = (a.run_dir/'raw.jsonl').open('x')
    def write(**r):
        fh.write(json.dumps(dict(wall_utc=time.time(),**r))+'\n'); fh.flush()
    def timed(fn):
        was=gc.isenabled(); gc.disable()
        try:
            sync(); t0=time.perf_counter_ns(); result=fn(); sync(); t1=time.perf_counter_ns()
        finally:
            if was: gc.enable()
        return t1-t0,t0,result
    def mask(k,r):
        if r: return ops.process_mask_native(pr,mc[:k],bx[:k],(640,640))
        return ops.process_mask(pr,mc[:k],bx[:k],(640,640),upsample=False)
    def nms(raw):
        return non_max_suppression(old.find_preds(raw),.25,.45,nc=80,max_det=300)
    ref_image=old.load_img(paths[0])
    with torch.no_grad():
        for _ in range(50): net(ref_image)
    ref0=None
    def ref(block):
        nonlocal ref0
        with torch.no_grad():
            samples=[timed(lambda:net(ref_image))[0] for _ in range(15)]
        med=float(np.median(samples)); ref0=med if ref0 is None else ref0
        write(kind='reference',block=block,ns=samples,ratio=med/ref0,flagged=abs(med/ref0-1)>.03)
    if a.device=='cpu':
        write(kind='pin',verify=hw.verify_pin(cpus,threads=a.threads,min_effective=.5 if a.threads==1 else 2.0))
    ref('start')
    with torch.no_grad():
        for ix,path in enumerate(paths):
            ref(ix); image=old.load_img(path)
            def full(k,r):
                raw=net(image); nms(raw); return mask(k,r)
            cells=[(k,r,c) for k in cfg['K'] for r in cfg['retinas'] for c in range(2 if k==0 else 1)]
            for k,r,c in cells:
                for _ in range(2): full(k,r)
            for rep in range(cfg['repeats']):
                order=cells[:]; random.Random(20261106+ix*100+rep).shuffle(order)
                for k,r,c in order:
                    tn,ts,_=timed(lambda:full(k,r)); tm,_,_=timed(lambda:mask(k,r))
                    write(kind='cell',image=path.name,block=ix,k=k,retina=r,copy=c,rep=rep,total_ns=tn,mask_ns=tm,t_start_ns=ts)
            print(f'M2_IMAGE {ix+1}/{len(paths)}',flush=True)
    ref('end')
    if a.device=='cpu':
        write(kind='pin_end',verify=hw.verify_pin(cpus,threads=a.threads,min_effective=.5 if a.threads==1 else 2.0))
    write(kind='done'); fh.close(); print('M2_DONE',flush=True)


if __name__=='__main__': main()
