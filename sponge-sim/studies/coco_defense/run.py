"""Download official validation images/segmentation labels; evaluate fixed defense configurations."""
import argparse
import functools
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def setup():
    from ultralytics.utils import ASSETS_URL, YAML
    import ultralytics
    root=Path('C:/sponge/coco_val_confirm'); root.mkdir(exist_ok=True)
    urls=dict(images='http://images.cocodataset.org/zips/val2017.zip',labels=ASSETS_URL+'/coco2017labels-segments.zip')
    downloads={}
    for key,url in urls.items():
        path=root/f'{key}.zip'; print('DOWNLOAD',url,flush=True)
        if not zipfile.is_zipfile(path):
            with urllib.request.urlopen(url,timeout=60) as src, path.open('xb') as dst:
                while chunk:=src.read(1<<20): dst.write(chunk)
        downloads[key]=dict(url=url,sha256=sha(path),bytes=path.stat().st_size)
        with zipfile.ZipFile(path) as z:
            bad=z.testzip(); assert bad is None,bad
            for name in z.namelist():
                parts=Path(name).parts
                if key=='images' and name.endswith('.jpg'):
                    dest=root/'images/val2017'/Path(name).name
                elif key=='labels' and '/labels/val2017/' in '/'+name and name.endswith('.txt'):
                    dest=root/'labels/val2017'/Path(name).name
                else: continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                content=z.read(name)
                if dest.exists():
                    assert dest.read_bytes()==content, f'existing file mismatch: {dest}'
                else:
                    with dest.open('xb') as f: f.write(content)
    images=sorted((root/'images/val2017').glob('*.jpg')); assert len(images)==5000,len(images)
    names=YAML.load(Path(ultralytics.__file__).parent/'cfg/datasets/coco.yaml')['names']
    cfg=dict(path=str(root),train='images/val2017',val='images/val2017',names=names)
    YAML.save(root/'val.yaml',cfg)
    manifest=dict(downloads=downloads,images={p.name:sha(p) for p in images},
                  labels={p.name:sha(p) for p in sorted((root/'labels/val2017').glob('*.txt'))})
    (root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('COCO_SETUP_DONE',len(images),len(manifest['labels']),flush=True)


def evaluate(run,smoke):
    import numpy as np
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import nms
    root=Path('C:/sponge'); data=root/'coco_val_confirm'; run.mkdir(parents=True,exist_ok=False)
    prov=json.loads((root/'confirm_provenance.json').read_text(encoding='utf-8-sig'))
    assert sha(__file__)==prov['coco_sha256']
    configs=[('baseline',300,30000),('maxdet50',50,30000),('prenms300',300,300),('prenms1000',300,1000)]
    manifest=dict(provenance=prov,smoke=smoke,dataset_manifest_sha256=sha(data/'dataset_manifest.json'),
                  config=dict(models=['yolov8n-seg.pt','yolo11n-seg.pt'],variants=configs,imgsz=640,batch=8,conf=.001,iou=.7,workers=0),
                  weights={n:sha(root/n) for n in ['yolov8n-seg.pt','yolo11n-seg.pt']},torch=torch.__version__,torchvision=torchvision.__version__)
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset=str(data/'val.yaml')
    if smoke:
        from ultralytics.utils import YAML
        ys=YAML.load(data/'val.yaml'); paths=sorted((data/'images/val2017').glob('*.jpg'))[:16]
        (run/'smoke_images.txt').write_text('\n'.join(map(str,paths)))
        ys['val']=str((run/'smoke_images.txt').resolve()); YAML.save(run/'smoke.yaml',ys); dataset=str(run/'smoke.yaml')
    original=nms.non_max_suppression
    for model_name in manifest['config']['models']:
        for name,maxdet,cap in configs:
            outdir=run/f'{model_name}_{name}'
            holder=[]
            class RawValidator(SegmentationValidator):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    holder.append(self)
                def get_stats(self):
                    # Preserve detection-level metric inputs before metrics.process clears them.
                    raw={k:np.concatenate(v,axis=0) if len(v) else np.array([]) for k,v in self.metrics.stats.items()}
                    self.save_dir.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(self.save_dir/'metric_inputs.npz',**raw)
                    return super().get_stats()
            nms.non_max_suppression=functools.partial(original,max_nms=cap)
            try:
                model=YOLO(str(root/model_name))
                result=model.val(validator=RawValidator,data=dataset,imgsz=640,batch=8,device=0,workers=0,
                                 conf=.001,iou=.7,max_det=maxdet,plots=False,save_json=False,save_txt=False,
                                 project=str(run),name=outdir.name,exist_ok=False,verbose=False)
                validator=holder[-1]
                assert validator.seen==(16 if smoke else 5000),validator.seen
                record=dict(model=model_name,variant=name,max_det=maxdet,max_nms=cap,seen=validator.seen,
                            results=result.results_dict,metric_inputs_sha256=sha(validator.save_dir/'metric_inputs.npz'))
                with (run/'results.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                print('COCO_VARIANT_DONE',model_name,name,record['results'],flush=True)
            finally: nms.non_max_suppression=original
    (run/'DONE').write_text('complete\n'); print('COCO_EVAL_DONE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--setup',action='store_true'); ap.add_argument('--run-dir',type=Path); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.setup: setup()
    else: evaluate(a.run_dir,a.smoke)
