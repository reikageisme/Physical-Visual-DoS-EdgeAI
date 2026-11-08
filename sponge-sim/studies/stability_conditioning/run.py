"""One controls-only matched-load conditioning diagnostic; no attacked inputs."""
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


class Telemetry:
    """Read-only NVML. Unsupported telemetry is recorded, not fabricated."""
    def __init__(self):
        import ctypes
        self.c=ctypes;self.error=None
        try:
            self.lib=ctypes.WinDLL('nvml.dll')
            assert self.lib.nvmlInit_v2()==0
            self.handle=ctypes.c_void_p()
            assert self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle))==0
        except Exception as e:self.error=str(e)

    def read(self):
        if self.error:return dict(error=self.error)
        out={}
        for key,name,extra in [('sm_mhz','nvmlDeviceGetClockInfo',[1]),('memory_mhz','nvmlDeviceGetClockInfo',[2]),
                               ('pstate','nvmlDeviceGetPerformanceState',[]),('temperature_c','nvmlDeviceGetTemperature',[0])]:
            value=self.c.c_uint()
            rc=getattr(self.lib,name)(self.handle,*extra,self.c.byref(value))
            out[key]=value.value if rc==0 else None
            if rc:out[key+'_error']=rc
        return out


def worker(a):
    import psutil
    proc=psutil.Process()
    original=proc.cpu_affinity()
    base_lock=json.loads(a.base_lock.read_text(encoding='utf-8-sig'))
    cpus=base_lock['affinity']
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
    fspec=importlib.util.spec_from_file_location('fixture',root/'gpu_full.py')
    fixture=importlib.util.module_from_spec(fspec);fspec.loader.exec_module(fixture)
    protos,coeffs,boxes=fixture.make_masks(300)
    from ultralytics.utils import ops
    image_path=root/'datasets/coco128-seg/images/train2017/000000000009.jpg'
    image=old.load_img(image_path)
    net=YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().cuda()
    run=a.run_dir/a.worker;run.mkdir(exist_ok=False)
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert prov['script_sha256']==sha(__file__)
    warmup=int(a.worker[1:])
    cfg=dict(profile=a.worker,affinity=cpus,threads=1,blocks=20,repeats=31,warmup=200,
             per_reference_warmup=warmup,condition_cycles=10,order_seed=20261106,canvas=640,
             drift_limit=.03,max_flags=2,max_mde_frac=.01)
    (run/'manifest.json').write_text(json.dumps(dict(config=cfg,provenance=prov,
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        image_sha256=sha(image_path),weights_sha256=sha(root/'yolov8n-seg.pt'),fixture_sha256=sha(root/'gpu_full.py'),
        base_lock_sha256=sha(a.base_lock),
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
    telemetry=Telemetry()
    with torch.no_grad():
        for _ in range(200):net(image)
        for block in ['start',*range(20),'end']:
            for _ in range(10):
                net(image);ops.process_mask_native(protos,coeffs,boxes,(640,640))
            torch.cuda.synchronize()
            state_load=telemetry.read()
            for _ in range(warmup):net(image)
            torch.cuda.synchronize()
            state_warm=telemetry.read()
            reference=[wall()[0] for _ in range(31)]
            med=float(np.median(reference));first=med if first is None else first
            write(kind='reference',block=block,samples_ns=reference,ratio=med/first,flagged=abs(med/first-1)>.03,
                  after_load=state_load,after_warmup=state_warm,after_reference=telemetry.read())
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
    ap.add_argument('--base-lock',type=Path,required=True)
    ap.add_argument('--worker',choices=['w5','w50','w200']);a=ap.parse_args()
    if a.worker:worker(a)
    else:
        a.run_dir.mkdir(parents=True,exist_ok=False)
        order=['w5','w50','w200'];random.Random(20261106).shuffle(order)
        (a.run_dir/'order.json').write_text(json.dumps(order))
        for profile in order:
            subprocess.run([sys.executable,'-u',__file__,'--run-dir',str(a.run_dir),'--provenance',str(a.provenance),
                            '--base-lock',str(a.base_lock),'--worker',profile],check=True)
