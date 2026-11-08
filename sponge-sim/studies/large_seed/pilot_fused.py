"""Fusion-only runtime QA; preserve score/gradient agreement before speed selection."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(a):
    import torch
    import numpy as np
    from ultralytics import YOLO
    import engine
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path('C:/sponge');spec=importlib.util.spec_from_file_location('old',root/'gpu_mechanism.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    ids=['000000000009.jpg','000000000025.jpg','000000000030.jpg','000000000034.jpg']
    image=torch.cat([old.load_img(root/'datasets/coco128-seg/images/train2017'/ids[j%4]) for j in range(a.batch)])
    net=YOLO(str(root/a.model)).model.eval().float().cuda()
    if a.model=='yolo26n-seg.pt':net.model[-1].end2end=True
    mode='native' if a.model=='yolo26n-seg.pt' else 'many'
    for p in net.parameters():p.requires_grad_(False)
    zero=torch.zeros_like(image)
    unfused_step=engine.eager(net,image,zero,mode,steps=1)
    with torch.no_grad():
        before=engine.prediction(net(image),mode).detach()
        before_scores=before[:,:,4].sort(1).values if mode=='native' else before[:,4:84]
    net.fuse(verbose=False)
    fused_step=engine.eager(net,image,zero,mode,steps=1)
    with torch.no_grad():
        after=engine.prediction(net(image),mode).detach()
        after_scores=after[:,:,4].sort(1).values if mode=='native' else after[:,4:84]
    fusion_cosine=float(torch.nn.functional.cosine_similarity((unfused_step-image).flatten(),(fused_step-image).flatten(),dim=0))
    score_error=float((before_scores-after_scores).abs().max())
    fusion_eligible=bool(fusion_cosine>.999 and score_error<1e-4)
    run=a.run_dir/f'{a.model}_b{a.batch}';run.mkdir(exist_ok=False)
    prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert sha(__file__)==prov['pilot_sha256'] and sha(root/'engine.py')==prov['engine_sha256']
    rows=[]
    d=torch.zeros_like(image)
    for kind in ['eager','graph']:
        try:
            torch.cuda.reset_peak_memory_stats()
            fn=(lambda x,y,steps=100:engine.eager(net,x,y,mode,steps)) if kind=='eager' else engine.GraphedPGD(net,image,mode)
            first=fn(image,d,steps=1)
            reference=engine.eager(net,image,d,mode,steps=1)
            cosine=float(torch.nn.functional.cosine_similarity((first-image).flatten(),(reference-image).flatten(),dim=0))
            torch.cuda.synchronize();t0=time.monotonic();result=fn(image,d,steps=100);torch.cuda.synchronize();elapsed=time.monotonic()-t0
            bound=float((result-image).abs().max());finite=bool(torch.isfinite(result).all())
            row=dict(engine=kind,seconds=elapsed,seconds_per_image=elapsed/a.batch,batch=a.batch,
                     cosine_one_step=cosine,finite=finite,max_delta=bound,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
            row['eligible']=bool(fusion_eligible and finite and bound<=16/255+1e-6 and cosine>.999 and row['peak_reserved_bytes']<=12*1024**3)
            rows.append(row);del fn
        except Exception as e:
            rows.append(dict(engine=kind,eligible=False,error=repr(e)))
    out=dict(model=a.model,mode=mode,provenance=prov,batch=a.batch,results=rows,fused=True,
             fusion_qa=dict(cosine=fusion_cosine,max_score_error=score_error,eligible=fusion_eligible),
             weights_sha256=sha(root/a.model),torch=torch.__version__)
    (run/'pilot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--provenance',type=Path,required=True)
    ap.add_argument('--model');ap.add_argument('--batch',type=int);a=ap.parse_args()
    if a.model:worker(a)
    else:
        a.run_dir.mkdir(parents=True,exist_ok=False)
        for model in ['yolov8n-seg.pt','yolo11n-seg.pt','yolo26n-seg.pt']:
            for batch in [4,8]:
                subprocess.run([sys.executable,'-u',__file__,'--run-dir',str(a.run_dir),'--provenance',str(a.provenance),
                                '--model',model,'--batch',str(batch)],check=True)
