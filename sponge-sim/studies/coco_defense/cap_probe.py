"""Prospective workload-cap count check on locked M3 inputs; no latency claims."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    root=Path('C:/sponge');src=root/'m3_native_full_20261106'
    spec=importlib.util.spec_from_file_location('old',root/'gpu_mechanism.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    files=sorted(src.glob('*_inputs.pt'))[:2 if a.smoke else 20];assert len(files)==(2 if a.smoke else 20)
    prov=json.loads((root/'cap_provenance.json').read_text(encoding='utf-8-sig'));assert sha(__file__)==prov['script_sha256']
    net=YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().cuda();a.run_dir.mkdir(parents=True,exist_ok=False)
    manifest=dict(provenance=prov,smoke=a.smoke,inputs={p.name:sha(p) for p in files},weights_sha256=sha(root/'yolov8n-seg.pt'),
                  config=dict(conf=.25,iou=.45,nc=80,variants=[['baseline',300,30000],['maxdet50',50,30000],['prenms300',300,300],['prenms1000',300,1000]]))
    (a.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    kernel=torchvision.ops.nms
    with torch.no_grad(),(a.run_dir/'raw.jsonl').open('x') as f:
        for p in files:
            data=torch.load(p,weights_only=True)
            for arm in ['clean','attacked']:
                preds=old.find_preds(net(data[arm].cuda()))
                for variant,maxdet,cap in manifest['config']['variants']:
                    calls=[]
                    def spy(boxes,scores,iou):
                        calls.append(len(boxes));return kernel(boxes,scores,iou)
                    torchvision.ops.nms=spy
                    try:det=non_max_suppression(preds.clone(),.25,.45,nc=80,max_det=maxdet,max_nms=cap)[0]
                    finally:torchvision.ops.nms=kernel
                    assert len(calls)<=1 and max(calls,default=0)<=cap and len(det)<=maxdet
                    f.write(json.dumps(dict(kind='counts',image=p.stem,arm=arm,variant=variant,kernel_n=max(calls,default=0),k=len(det)))+'\n');f.flush()
        f.write(json.dumps(dict(kind='done'))+'\n')
    print('CAP_PROBE_DONE',len(files),flush=True)
