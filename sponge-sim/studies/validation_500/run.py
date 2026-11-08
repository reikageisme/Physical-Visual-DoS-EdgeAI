"""Prospective fixed COCO-val500 attack/count validation; no latency inference."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--provenance',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    import torch
    import torchvision
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    root=Path('C:/sponge');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261106)
    cohort=json.loads(a.cohort.read_text(encoding='utf-8-sig'));prov=json.loads(a.provenance.read_text(encoding='utf-8-sig'))
    assert prov['script_sha256']==sha(__file__) and prov['cohort_sha256']==sha(a.cohort)
    samples=cohort['images'][:4] if a.smoke else cohort['images']
    assert len(samples)==(4 if a.smoke else 500)
    data=root/'coco_val_confirm/images/val2017'
    for row in samples:assert sha(data/row['file_name'])==row['sha256']
    old=load_module('historical',root/'gpu_mechanism.py')
    target=YOLO(str(root/'yolo26n-seg.pt')).model.eval().float().cuda()
    assert getattr(target.model[-1],'one2one_cv2',None) is not None
    target.model[-1].end2end=True;assert target.model[-1].end2end
    cfg=dict(n_images=len(samples),smoke=a.smoke,batch_size=4,steps=100,eps=16/255,alpha=2/255,temp=.02,
             canvas=640,nc=80,conf=.25,iou=.45,max_det=300,seed=20261106,target_end2end=True,
             models=['yolov8n-seg.pt','yolo11n-seg.pt'],target='yolo26n-seg.pt',
             variants=[['baseline',300,30000],['maxdet50',50,30000],['prenms300',300,300],['prenms1000',300,1000]])
    a.run_dir.mkdir(parents=True,exist_ok=False)
    (a.run_dir/'manifest.json').write_text(json.dumps(dict(config=cfg,provenance=prov,
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        cohort=cohort,historical_source_sha256=sha(root/'gpu_mechanism.py'),
        weights={n:sha(root/n) for n in cfg['models']+[cfg['target']]},
        torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),indent=2))
    fh=(a.run_dir/'raw.jsonl').open('x')
    def write(**r):
        fh.write(json.dumps(r)+'\n');fh.flush()
    kernel=torchvision.ops.nms
    def count(pred,cap,maxdet):
        calls=[]
        def spy(boxes,scores,iou):
            calls.append(len(boxes));return kernel(boxes,scores,iou)
        torchvision.ops.nms=spy
        try:det=non_max_suppression(pred.clone(),.25,.45,nc=80,max_det=maxdet,max_nms=cap)[0]
        finally:torchvision.ops.nms=kernel
        assert len(calls)<=1 and max(calls,default=0)<=cap and len(det)<=maxdet
        return dict(n_tau=int((pred[:,4:84].amax(1)>.25).sum()),kernel_n=max(calls,default=0),k=len(det))
    def target_count(image):
        raw=target(image)
        native=next(t for t in old.flat(raw) if t.dim()==3 and tuple(t.shape[1:])==(300,38))
        assert torch.isfinite(native).all()
        return native
    t_start=time.monotonic()
    for model_name in cfg['models']:
        net=YOLO(str(root/model_name)).model.eval().float().cuda()
        assert not net.training
        for p in net.parameters():p.requires_grad_(False)
        delta_dir=a.run_dir/model_name;delta_dir.mkdir()
        for start in range(0,len(samples),4):
            batch=samples[start:start+4];names=[r['file_name'] for r in batch]
            clean=torch.cat([old.load_img(data/n) for n in names],0)
            # Existing objective is a batch sum in eval mode: no cross-image
            # train-time BN or shared perturbation. FP32 batched kernels can
            # differ numerically from batch1; batch4 is fixed prospectively.
            adv=old.attack(net,clean,steps=100)
            delta=(adv-clean).detach()
            assert float(delta.abs().max())<=16/255+1e-6
            assert float(adv.min())>=0 and float(adv.max())<=1 and torch.isfinite(adv).all()
            artifact=delta_dir/f'batch_{start:04d}.npz'
            np.savez_compressed(artifact,delta=delta.cpu().numpy(),image_names=np.asarray(names))
            write(kind='snapshot',model=model_name,start=start,file=str(artifact.relative_to(a.run_dir)),sha256=sha(artifact))
            with torch.no_grad():
                cp=old.find_preds(net(clean));advp=old.find_preds(net(adv))
                transfer={}
                if model_name=='yolov8n-seg.pt':
                    transfer['clean']=target_count(clean).detach().cpu().numpy()
                    transfer['attacked']=target_count(adv).detach().cpu().numpy()
                    native_file=delta_dir/f'native26_{start:04d}.npz'
                    np.savez_compressed(native_file,**transfer,image_names=np.asarray(names))
                    write(kind='native_snapshot',start=start,file=str(native_file.relative_to(a.run_dir)),sha256=sha(native_file))
                for j,name in enumerate(names):
                    for arm,preds in [('clean',cp),('attacked',advp)]:
                        for variant,maxdet,cap in cfg['variants']:
                            result=count(preds[j:j+1],cap,maxdet)
                            write(kind='counts',model=model_name,image=name,arm=arm,variant=variant,**result)
                        if transfer:
                            det=transfer[arm][j]
                            write(kind='native26',source=model_name,image=name,arm=arm,k=int((det[:,4]>.25).sum()))
            if start%20==0 or start+4>=len(samples):
                print(f'VAL500_PROGRESS {model_name} {min(start+4,len(samples))}/{len(samples)} elapsed_s={time.monotonic()-t_start:.1f}',flush=True)
        del net
    write(kind='done');fh.close();print('VAL500_DONE',flush=True)


if __name__=='__main__':main()
