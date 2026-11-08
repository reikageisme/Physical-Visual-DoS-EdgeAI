"""Separate source transfer, target-head choice, and target-specific optimization."""
import hashlib
import importlib.util
import json
from pathlib import Path


if __name__=='__main__':
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    import engine
    root=Path('C:/sponge');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run=root/'diagnose26_20261106';run.mkdir(exist_ok=False)
    ids=['000000000009.jpg','000000000025.jpg','000000000030.jpg','000000000034.jpg']
    spec=importlib.util.spec_from_file_location('old',root/'gpu_mechanism.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    images=torch.cat([old.load_img(root/'datasets/coco128-seg/images/train2017'/n) for n in ids])
    source=YOLO(str(root/'yolov8n-seg.pt')).model.eval().float().cuda()
    target=YOLO(str(root/'yolo26n-seg.pt')).model.eval().float().cuda()
    for net in [source,target]:
        for p in net.parameters():p.requires_grad_(False)
    head=target.model[-1];assert head.one2one_cv2 is not None
    manifest=dict(images=ids,budget=dict(steps=100,eps=16/255,alpha=2/255),
        weights={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['yolov8n-seg.pt','yolo26n-seg.pt']},
        engine_sha256=hashlib.sha256((root/'engine.py').read_bytes()).hexdigest(),
        provenance=json.loads((root/'diagnose26_provenance.json').read_text(encoding='utf-8-sig')))
    assert manifest['provenance']['script_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for mode in ['many','native']:
        head.end2end=mode=='native';x=images.detach().clone().requires_grad_(True)
        loss=engine.objective(target(x),mode)
        grad=torch.autograd.grad(loss,x)[0]
        rows.append(dict(kind='gradient',head=mode,loss=float(loss),finite=bool(torch.isfinite(grad).all()),
                         grad_l2=[float(g.norm()) for g in grad],grad_max=[float(g.abs().max()) for g in grad]))
    zero=torch.zeros_like(images)
    conditions=[('clean',images)]
    conditions.append(('transfer_from_v8',engine.eager(source,images,zero,'many',100)))
    head.end2end=False;conditions.append(('self26_many',engine.eager(target,images,zero,'many',100)))
    head.end2end=True;conditions.append(('self26_native',engine.eager(target,images,zero,'native',100)))
    with torch.no_grad():
        for name,x in conditions:
            assert float((x-images).abs().max())<=16/255+1e-6
            head.end2end=False;many=engine.prediction(target(x),'many')
            km=[len(non_max_suppression(many[j:j+1].clone(),.25,.45,nc=80,max_det=300)[0]) for j in range(4)]
            n=[int((many[j,4:84].amax(0)>.25).sum()) for j in range(4)]
            head.end2end=True;native=engine.prediction(target(x),'native')
            for j,image in enumerate(ids):
                scores=native[j,:,4].sort(descending=True).values
                rows.append(dict(kind='condition',condition=name,image=image,many_n=n[j],many_k=km[j],
                    native_k=int((scores>.25).sum()),native_score_order_stats=[float(scores[k]) for k in [0,9,49,99,299]]))
            torch.save(dict(inputs=x.cpu(),native=native.cpu(),many=many.cpu()),run/f'{name}.pt')
    with (run/'raw.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
        f.write(json.dumps(dict(kind='done'))+'\n')
    print(json.dumps(rows),flush=True)
