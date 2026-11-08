"""Controls-only host-affinity calibration. Never loads attacked inputs."""
import argparse
import gc
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(a):
    import psutil
    proc=psutil.Process()
    original=proc.cpu_affinity()
    cpus=dict(all=original,one=[2],four=[2,3,4,5])[a.worker]
    proc.cpu_affinity(cpus)
    assert proc.cpu_affinity()==cpus
    import torch
    import torchvision
    import numpy as np
    from ultralytics import YOLO
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261106)
    root=Path('C:/sponge')
    spec=importlib.util.spec_from_file_location('historical',root/'gpu_mechanism.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    image_path=root/'datasets/coco128-seg/images/train2017/000000000009.jpg'
    image=old.load_img(image_path)
    net=YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().cuda()
    run=a.run_dir/a.worker;run.mkdir(exist_ok=False)
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert prov['script_sha256']==sha(__file__)
    cfg=dict(profile=a.worker,affinity=cpus,threads=1,blocks=20,repeats=31,warmup=200,
             per_reference_warmup=5,order_seed=20261106,canvas=640,drift_limit=.03,max_flags=2,max_mde_frac=.01)
    (run/'manifest.json').write_text(json.dumps(dict(config=cfg,provenance=prov,
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        image_sha256=sha(image_path),weights_sha256=sha(root/'yolov8n-seg.pt'),
        torch=torch.__version__,torchvision=torchvision.__version__,original_affinity=original),indent=2))
    fh=(run/'raw.jsonl').open('x')
    def write(**r):
        fh.write(json.dumps(dict(wall_utc=time.time(),**r))+'\n');fh.flush()
    def wall():
        enabled=gc.isenabled();gc.disable()
        try:
            torch.cuda.synchronize();start=time.perf_counter_ns();result=net(image)
            torch.cuda.synchronize();end=time.perf_counter_ns()
        finally:
            if enabled:gc.enable()
        return end-start,start
    first=None
    with torch.no_grad():
        for _ in range(200):net(image)
        for block in ['start',*range(20),'end']:
            for _ in range(5):net(image)
            reference=[wall()[0] for _ in range(31)]
            med=float(np.median(reference));first=med if first is None else first
            write(kind='reference',block=block,samples_ns=reference,ratio=med/first,flagged=abs(med/first-1)>.03)
            if isinstance(block,int):
                samples=[]
                for rep in range(31):
                    labels=['A','B'];random.Random(20261106+100*block+rep).shuffle(labels)
                    for label in labels:
                        ns,start=wall();samples.append(dict(label=label,rep=rep,ns=ns,t_start_ns=start))
                e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
                device_ms=[]
                for _ in range(15):
                    torch.cuda.synchronize();e0.record();result=net(image);e1.record();e1.synchronize()
                    device_ms.append(float(e0.elapsed_time(e1)))
                write(kind='aa',block=block,samples=samples,device_forward_ms=device_ms)
        write(kind='done')
    fh.close();print('CALIBRATION_DONE',a.worker,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--provenance',type=Path,required=True)
    ap.add_argument('--worker',choices=['all','one','four']);a=ap.parse_args()
    if a.worker:worker(a)
    else:
        a.run_dir.mkdir(parents=True,exist_ok=False)
        order=['all','one','four'];random.Random(20261106).shuffle(order)
        (a.run_dir/'order.json').write_text(json.dumps(order))
        for profile in order:
            subprocess.run([sys.executable,'-u',__file__,'--run-dir',str(a.run_dir),'--provenance',str(a.provenance),'--worker',profile],check=True)
