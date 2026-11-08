"""Validate saved perturbations and replay source counts; never rewrite measurements."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    import torch
    import torchvision
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    root=Path('C:/sponge');torch.set_num_threads(1);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('historical',root/'gpu_mechanism.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    manifest=json.loads((a.run/'manifest.json').read_text())
    rows=[json.loads(s) for s in (a.run/'raw.jsonl').read_text().splitlines()];assert rows[-1]['kind']=='done'
    expected={(r['model'],r['image'],r['arm'],r['variant']):r['k'] for r in rows if r['kind']=='counts'}
    native={(r['image'],r['arm']):r['k'] for r in rows if r['kind']=='native26'}
    for name,digest in manifest['weights'].items():assert sha(root/name)==digest
    hashes=0;native_checks=0;budget_checks=0;replayed=0;mismatches=[];maximum_delta=0.
    for r in rows:
        if r['kind'] not in ['snapshot','native_snapshot']:continue
        p=a.run/r['file'];assert sha(p)==r['sha256'];hashes+=1
        if r['kind']=='native_snapshot':
            d=np.load(p,allow_pickle=False)
            for j,image in enumerate(d['image_names']):
                for arm in ['clean','attacked']:
                    assert int((d[arm][j,:,4]>.25).sum())==native[str(image),arm]
                    native_checks+=1
    with torch.no_grad():
        for model in manifest['config']['models']:
            net=YOLO(str(root/model)).model.eval().float().cuda()
            records=[r for r in rows if r['kind']=='snapshot' and r['model']==model]
            for record in records:
                data=np.load(a.run/record['file'],allow_pickle=False)
                names=[str(n) for n in data['image_names']]
                clean=torch.cat([old.load_img(root/'coco_val_confirm/images/val2017'/n) for n in names])
                delta=torch.from_numpy(data['delta']).cuda()
                bound=float(delta.abs().max());maximum_delta=max(maximum_delta,bound)
                assert bound<=16/255+1e-6 and torch.isfinite(delta).all()
                adv=(clean+delta).clamp(0,1);budget_checks+=len(names)
                for arm,im in [('clean',clean),('attacked',adv)]:
                    preds=old.find_preds(net(im))
                    for variant,maxdet,cap in manifest['config']['variants']:
                        # Evaluate one image at a time as in original counting.
                        for j,image in enumerate(names):
                            k=len(non_max_suppression(preds[j:j+1].clone(),.25,.45,nc=80,max_det=maxdet,max_nms=cap)[0])
                            wanted=expected[model,image,arm,variant];replayed+=1
                            if k!=wanted:mismatches.append(dict(model=model,image=image,arm=arm,variant=variant,recorded=wanted,replayed=k))
            print('REPLAY_MODEL_DONE',model,flush=True)
            del net
    receipt=dict(artifact_hashes_checked=hashes,native_counts_verified=native_checks,images_budget_verified=budget_checks,
                 max_saved_delta=maximum_delta,source_count_replays=replayed,mismatch_count=len(mismatches),mismatches=mismatches,
                 note='Float32 delta reconstruction and fresh GPU kernels can differ from original inputs/inference by rounding. Any mismatch retained; no raw record changed.')
    with (a.run/'replay_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='mismatches'}),flush=True)
