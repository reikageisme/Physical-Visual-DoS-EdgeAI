"""Nested1000/1500, five starts, both source models and both target26 objectives.

Every batch retains count-level detections; exact adversarial inputs retained
on an independently locked64-image audit subset. No latency claim.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

TASKS={'v8':('yolov8n-seg.pt','many'),'v11':('yolo11n-seg.pt','many'),
       'y26_many':('yolo26n-seg.pt','many'),'y26_native':('yolo26n-seg.pt','native')}


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def worker(a):
    import torch
    import torchvision
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    import engine
    root=Path('C:/sponge');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg=json.loads(a.cohort.read_text());lock=json.loads(a.lock.read_text());prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert sha(__file__)==prov['runner_sha256'] and sha(root/'engine.py')==prov['engine_sha256']
    assert sha(a.cohort)==prov['cohort_sha256'] and sha(a.lock)==prov['lock_sha256']
    model,mode=TASKS[a.task];runtime=lock['selected'][model]
    assert runtime['batch']==4 and not runtime['fused']
    samples=cfg['images'][:1000] if a.stage==1000 else cfg['images'][1000:1500]
    if a.smoke:samples=samples[:4]
    data=root/'coco_val_confirm/images/val2017'
    spec=importlib.util.spec_from_file_location('historical',root/'gpu_mechanism.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for row in samples:assert sha(data/row['file_name'])==row['sha256']
    net=YOLO(str(root/model)).model.eval().float().cuda()
    if model=='yolo26n-seg.pt':net.model[-1].end2end=mode=='native'
    for p in net.parameters():p.requires_grad_(False)
    target=net if model=='yolo26n-seg.pt' else YOLO(str(root/'yolo26n-seg.pt')).model.eval().float().cuda()
    if target is not net:
        target.model[-1].end2end=True
        for p in target.parameters():p.requires_grad_(False)
    old_names=['000000000009.jpg','000000000025.jpg','000000000030.jpg','000000000034.jpg']
    warm=torch.cat([old.load_img(root/'datasets/coco128-seg/images/train2017'/n) for n in old_names])
    attack=engine.GraphedPGD(net,warm,mode) if runtime['engine']=='graph' else lambda x,d,steps=100:engine.eager(net,x,d,mode,steps)
    zero=torch.zeros_like(warm)
    probe=attack(warm,zero,steps=1)
    reference=engine.eager(net,warm,zero,mode,steps=1)
    agreement=torch.nn.functional.cosine_similarity((probe-warm).flatten(1),(reference-warm).flatten(1),dim=1)
    assert bool((agreement>.999).all()),'runtime objective QA failed'
    output=a.run_dir/f'stage{a.stage}'/a.task;output.mkdir(parents=True,exist_ok=True)
    manifest=dict(task=a.task,model=model,objective=mode,stage=a.stage,smoke=a.smoke,runtime=runtime,provenance=prov,
        config=dict(steps=100,eps=16/255,alpha=2/255,temp=.02,conf=.25,iou=.45,nc=80,batch4=True,
                    seeds=cfg['seeds'],snapshot_policy='all detections/scores; exact adv inputs only locked64 audit IDs'),
        weights={n:sha(root/n) for n in set([model,'yolo26n-seg.pt'])},images=samples,
        torch=torch.__version__,torchvision=torchvision.__version__,startup_gradient_cosines=agreement.cpu().tolist())
    if (output/'manifest.json').exists():
        previous=json.loads((output/'manifest.json').read_text())
        assert all(previous[k]==v for k,v in manifest.items() if k!='startup_gradient_cosines')
        with (output/'resume_checks.jsonl').open('a') as f:
            f.write(json.dumps(dict(wall_utc=time.time(),startup_gradient_cosines=agreement.cpu().tolist()))+'\n')
    else:(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    kernel=torchvision.ops.nms;variants=[('baseline',300,30000),('maxdet50',50,30000),('prenms300',300,300),('prenms1000',300,1000)]
    def evaluate_many(pred):
        arrays=[];records=[]
        for variant,maxdet,cap in variants:
            dets=[];ks=[];ns=[]
            for j in range(len(pred)):
                calls=[]
                def spy(boxes,scores,iou):calls.append(len(boxes));return kernel(boxes,scores,iou)
                torchvision.ops.nms=spy
                try:d=non_max_suppression(pred[j:j+1].clone(),.25,.45,nc=80,max_det=maxdet,max_nms=cap)[0]
                finally:torchvision.ops.nms=kernel
                assert len(d)<=maxdet and max(calls,default=0)<=cap
                pad=torch.zeros(300,6,device='cuda');pad[:len(d)]=d[:,:6]
                dets.append(pad);ks.append(len(d));ns.append(max(calls,default=0))
            arrays.append(torch.stack(dets).cpu().numpy());records.append(dict(variant=variant,k=ks,kernel_n=ns))
        return np.stack(arrays,axis=1),records
    def eval_all(x):
        if model=='yolo26n-seg.pt':net.model[-1].end2end=False
        many=engine.prediction(net(x),'many')
        dets,records=evaluate_many(many)
        scores=many[:,4:84].amax(1).cpu().numpy()
        target.model[-1].end2end=True
        native=engine.prediction(target(x),'native').detach().cpu().numpy()
        assert np.isfinite(native).all()
        if model=='yolo26n-seg.pt':net.model[-1].end2end=mode=='native'
        return dict(detections=dets,score_max=scores,native=native),records
    t0=time.monotonic()
    for seed in cfg['seeds']:
        seed_dir=output/f'seed{seed}';seed_dir.mkdir(exist_ok=True)
        for offset in range(0,len(samples),4):
            rows=samples[offset:offset+4];names=[r['file_name'] for r in rows]
            assert len(rows)==4
            stem=f'batch{offset:04d}'
            attempts=sorted(seed_dir.glob(stem+'_attempt*'))
            done=[p for p in attempts if (p/'DONE').exists()]
            if done:continue  # first completed attempt remains canonical; no effect-based choice
            dest=seed_dir/f'{stem}_attempt{len(attempts)+1:02d}';dest.mkdir(exist_ok=False)
            images=torch.cat([old.load_img(data/n) for n in names])
            if model=='yolo26n-seg.pt':net.model[-1].end2end=mode=='native'
            delta=engine.initial_delta(images,names,seed,model)
            initial_norms=[float(d.norm()) for d in delta]
            assert all(v==0 for v in initial_norms) if seed==0 else all(v>0 for v in initial_norms)
            adv=attack(images,delta,steps=100)
            bound=float((adv-images).abs().max())
            assert bound<=16/255+1e-6 and float(adv.min())>=0 and float(adv.max())<=1 and torch.isfinite(adv).all()
            with torch.no_grad():
                clean,cr=eval_all(images);attacked,ar=eval_all(adv)
            np.savez_compressed(dest/'outputs.npz',image_names=np.asarray(names),
                clean_detections=clean['detections'],attacked_detections=attacked['detections'],
                clean_score_max=clean['score_max'],attacked_score_max=attacked['score_max'],
                clean_native=clean['native'],attacked_native=attacked['native'])
            audit=[j for j,row in enumerate(rows) if row['audit']]
            if audit:
                np.savez_compressed(dest/'audit_adv.npz',image_names=np.asarray([names[j] for j in audit]),adv=adv[audit].cpu().numpy())
            result=dict(task=a.task,model=model,objective=mode,stage=a.stage,seed=seed,offset=offset,images=names,
                max_delta=bound,clean_variants=cr,attacked_variants=ar,
                initial_delta_l2=initial_norms,
                clean_native_k=[int((v[:,4]>.25).sum()) for v in clean['native']],
                attacked_native_k=[int((v[:,4]>.25).sum()) for v in attacked['native']],
                input_sha256=[hashlib.sha256(adv[j].cpu().numpy().tobytes()).hexdigest() for j in range(4)],
                output_sha256=sha(dest/'outputs.npz'),audit_sha256=sha(dest/'audit_adv.npz') if audit else None)
            (dest/'record.json').write_text(json.dumps(result,indent=2));(dest/'DONE').write_text('complete\n')
            if offset%100==0 or offset+4>=len(samples):
                print(f'LARGE_PROGRESS stage={a.stage} task={a.task} seed={seed} images={min(offset+4,len(samples))}/{len(samples)} elapsed={time.monotonic()-t0:.1f}',flush=True)
        if not (seed_dir/'DONE').exists():(seed_dir/'DONE').write_text('complete\n')
    if not (output/'DONE').exists():(output/'DONE').write_text('complete\n')
    print('LARGE_TASK_DONE',a.stage,a.task,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--cohort',type=Path,required=True)
    ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--provenance',type=Path,required=True)
    ap.add_argument('--stage',type=int,choices=[1000,1500]);ap.add_argument('--task',choices=list(TASKS));ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    if a.task:worker(a)
    else:
        a.run_dir.mkdir(parents=True,exist_ok=True)
        for stage in ([1000] if a.smoke else [1000,1500]):
            for task in ['y26_many','y26_native','v8','v11']:
                cmd=[sys.executable,'-u',__file__,'--run-dir',str(a.run_dir),'--cohort',str(a.cohort),'--lock',str(a.lock),
                     '--provenance',str(a.provenance),'--stage',str(stage),'--task',task]
                if a.smoke:cmd.append('--smoke')
                subprocess.run(cmd,check=True)
            if not (a.run_dir/f'stage{stage}'/'DONE').exists():(a.run_dir/f'stage{stage}'/'DONE').write_text('complete\n')
            print('LARGE_STAGE_DONE',stage,flush=True)
        if not (a.run_dir/'DONE').exists():(a.run_dir/'DONE').write_text('complete\n')
        print('LARGE_ALL_DONE',flush=True)
