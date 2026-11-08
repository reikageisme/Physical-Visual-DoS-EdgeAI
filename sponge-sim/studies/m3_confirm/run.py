"""M3 prospective rerun and native YOLO26 transfer counts (no NMS on YOLO26)."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    root=Path('C:/sponge'); torch.manual_seed(20261106)
    spec=importlib.util.spec_from_file_location('old',root/'gpu_mechanism.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    paths=sorted((root/'datasets/coco128-seg/images/train2017').glob('*.jpg'))[:2 if a.smoke else 20]
    prov=json.loads((root/'confirm_provenance.json').read_text(encoding='utf-8-sig')); assert sha(__file__)==prov['m3_sha256']
    source=YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().cuda()
    target=YOLO(str(root/'yolo26n-seg.pt')).model.eval().float().cuda()
    assert getattr(target.model[-1],'one2one_cv2',None) is not None, 'missing native one-to-one head'
    target.model[-1].end2end=True
    assert bool(target.model[-1].end2end), 'not an end-to-end model'
    for p in source.parameters(): p.requires_grad_(False)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    manifest=dict(provenance=prov,smoke=a.smoke,images={p.name:sha(p) for p in paths},
                  weights={p:sha(root/p) for p in ['yolov8n-seg.pt','yolo26n-seg.pt']},
                  historical_sha256=sha(root/'gpu_mechanism.py'),torch=torch.__version__,torchvision=torchvision.__version__,
                  config=dict(steps=100,eps=16/255,alpha=2/255,conf=.25,iou=.45,canvas=640,max_det=300,seed=20261106,target_end2end=True))
    (a.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (a.run_dir/'raw.jsonl').open('x') as f:
        for ix,path in enumerate(paths):
            clean=old.load_img(path); adv=old.attack(source,clean,steps=100)
            output={}
            with torch.no_grad():
                for arm,image in [('clean',clean),('attacked',adv)]:
                    d8=non_max_suppression(old.find_preds(source(image)),.25,.45,nc=80,max_det=300)[0]
                    raw26=target(image)
                    ds=[t for t in old.flat(raw26) if t.dim()==3 and t.shape[1]==300 and t.shape[2]==38]
                    assert len(ds)>=1, 'unexpected native output; stop instead of applying NMS'
                    detections=ds[0][0]
                    assert torch.isfinite(detections).all()
                    k26=int((detections[:,4]>.25).sum())
                    output[arm]=dict(k_v8=len(d8),k26_native=k26,shape=list(ds[0].shape),
                                     native_sha256=hashlib.sha256(detections.cpu().numpy().tobytes()).hexdigest())
                    torch.save(detections.cpu(),a.run_dir/f'{path.stem}_{arm}_native26.pt')
            torch.save(dict(clean=clean.cpu(),attacked=adv.cpu()),a.run_dir/f'{path.stem}_inputs.pt')
            f.write(json.dumps(dict(kind='attack',image=path.name,wall_utc=time.time(),**output))+'\n'); f.flush()
            print(f'M3_IMAGE {ix+1}/{len(paths)} {output}',flush=True)
        f.write(json.dumps(dict(kind='done'))+'\n')
    print('M3_DONE',flush=True)


if __name__=='__main__': main()
