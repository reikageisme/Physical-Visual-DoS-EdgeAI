"""Labeled local diagnostics with a predeclared seed-0 patch, not official COCO evaluation.

The same class-agnostic custom NMS is used as in the two-image experiment.
Labels are YOLO-format boxes; crowd/ignore metadata and COCO maxDets evaluation
are not available. AP is 101-point class-mean interpolation at IoU .50:.95,
conditional on the stated confidence filter. No pretrained generalization claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from local_q1_validation import ROOT, VictimModel, load_image, save_json, sha256, summary


IOUS=np.linspace(.5,.95,10)


def labels_for(path):
    if not path.exists() or not path.read_text(encoding='utf-8').strip():
        return np.empty((0,5),dtype=float)
    data=np.loadtxt(path,ndmin=2)
    cls=data[:,0:1]; xy=data[:,1:3]*320; wh=data[:,3:5]*320
    return np.concatenate([xy-wh/2,xy+wh/2,cls],axis=1)


def match_predictions(predictions, targets):
    order=np.argsort(-predictions[:,4],kind='stable') if len(predictions) else np.array([],dtype=int)
    predictions=predictions[order]
    correct=np.zeros((len(predictions),len(IOUS)),dtype=bool)
    if not len(predictions) or not len(targets):
        return predictions,correct
    lower=np.maximum(predictions[:,None,:2],targets[None,:,:2])
    upper=np.minimum(predictions[:,None,2:4],targets[None,:,2:4])
    intersection=np.maximum(upper-lower,0).prod(-1)
    parea=np.maximum(predictions[:,2:4]-predictions[:,:2],0).prod(-1)
    tarea=np.maximum(targets[:,2:4]-targets[:,:2],0).prod(-1)
    iou=intersection/(parea[:,None]+tarea[None,:]-intersection+1e-9)
    for col,threshold in enumerate(IOUS):
        used=set()
        for index,prediction in enumerate(predictions):
            eligible=[j for j,target in enumerate(targets) if int(target[4])==int(prediction[5]) and j not in used and iou[index,j]>=threshold]
            if eligible:
                matched=max(eligible,key=lambda j:iou[index,j]); used.add(matched); correct[index,col]=True
    return predictions,correct


def detection_metrics(records):
    preds=[]; tps=[]; target_classes=[]
    for record in records:
        pred=np.asarray(record['predictions'],dtype=float).reshape(-1,6)
        targets=np.asarray(record['targets'],dtype=float).reshape(-1,5)
        pred,tp=match_predictions(pred,targets)
        preds.append(pred); tps.append(tp); target_classes.extend(targets[:,4].astype(int).tolist())
    pred=np.concatenate(preds) if preds else np.empty((0,6))
    tp=np.concatenate(tps) if tps else np.empty((0,10),dtype=bool)
    target_classes=np.asarray(target_classes,dtype=int)
    ap=[]
    for cls in np.unique(target_classes):
        n_gt=int((target_classes==cls).sum())
        indices=np.where(pred[:,5].astype(int)==cls)[0]
        indices=indices[np.argsort(-pred[indices,4],kind='stable')]
        true=tp[indices].astype(float)
        cumtp=true.cumsum(axis=0); cumfp=(1-true).cumsum(axis=0)
        recall=cumtp/n_gt; precision=cumtp/np.maximum(cumtp+cumfp,1)
        class_ap=[]
        for j in range(10):
            class_ap.append(float(np.mean([precision[recall[:,j]>=r,j].max() if np.any(recall[:,j]>=r) else 0. for r in np.linspace(0,1,101)])))
        ap.append(class_ap)
    ap=np.asarray(ap)
    true_count=int(tp[:,0].sum()) if len(tp) else 0
    return {"images":len(records),"ground_truth_boxes":len(target_classes),"ground_truth_classes":len(ap),
            "predicted_boxes":len(pred),"true_positives_iou50":true_count,
            "precision_iou50":true_count/max(len(pred),1),"recall_iou50":true_count/max(len(target_classes),1),
            "AP50_101point_diagnostic":float(ap[:,0].mean()) if len(ap) else 0.,
            "mAP50_95_101point_diagnostic":float(ap.mean()) if len(ap) else 0.}


def verify_metrics():
    gt=[[0,0,10,10,0]]
    exact=[{'targets':gt,'predictions':[[0,0,10,10,.9,0]]}]
    assert detection_metrics(exact)['mAP50_95_101point_diagnostic']==1
    assert detection_metrics([{'targets':gt,'predictions':[[0,0,10,10,.9,1]]}])['AP50_101point_diagnostic']==0
    fp_first=[{'targets':gt,'predictions':[[20,20,30,30,.95,0],[0,0,10,10,.9,0]]}]
    assert abs(detection_metrics(fp_first)['AP50_101point_diagnostic']-.5)<1e-12
    duplicate=[{'targets':gt,'predictions':[[0,0,10,10,.95,0],[0,0,10,10,.9,0]]}]
    assert detection_metrics(duplicate)['true_positives_iou50']==1
    print('AP matching and interpolation fixtures: PASS',flush=True)


def postprocess(victim, pred, threshold, cap):
    start=time.perf_counter_ns()
    scores,classes=pred[4:].max(dim=0)
    mask=scores>threshold
    boxes=victim._xywh_to_xyxy(pred[:4,mask].T.float())
    scores=scores[mask].float(); classes=classes[mask]
    active=len(scores)
    if cap is not None and active>cap:
        index=torch.topk(scores,cap).indices
        boxes,scores,classes=boxes[index],scores[index],classes[index]
    nms_start=time.perf_counter_ns()
    index=victim._nms(boxes,scores,.45)[:300]
    end=time.perf_counter_ns()
    predictions=torch.cat([boxes[index],scores[index,None],classes[index,None]],dim=1).cpu().tolist()
    return {'active_candidates':active,'nms_input_candidates':len(scores),'final_detections':len(index),
            'filter_ms':(nms_start-start)/1e6,'nms_ms':(end-nms_start)/1e6,'postprocessing_ms':(end-start)/1e6,'predictions':predictions}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--patch-run',default='outputs/local_q1_20261002_v2')
    parser.add_argument('--out-dir',default='outputs/local_coco128_20261002')
    args=parser.parse_args()
    verify_metrics()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    run=ROOT/args.patch_run; dataset=ROOT/'outputs/datasets/coco128'
    source_manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if 'finished_utc' not in source_manifest:
        raise RuntimeError('Wait for the optimizer experiment to finish before starting timing on the same CPU.')
    paths=sorted((dataset/'images/train2017').glob('*.jpg'))
    assert len(paths)==128, f'Expected 128 dataset images, found {len(paths)}'
    label_dir=dataset/'labels/train2017'
    missing_labels=[p.with_suffix('.txt').name for p in paths if not (label_dir/p.with_suffix('.txt').name).exists()]
    image_stems={p.stem for p in paths}
    extra_labels=sorted(p.name for p in label_dir.glob('*.txt') if p.stem not in image_stems)
    # Distinguish source-dataset missing labels from an incomplete extraction.
    with zipfile.ZipFile(ROOT/'outputs/datasets/coco128.zip') as archive:
        names=set(archive.namelist())
        for filename in missing_labels:
            assert f'coco128/labels/train2017/{filename}' not in names, 'Restore a label present in the source archive.'
    print(f'Dataset: {len(paths)} images; {len(missing_labels)} missing label files in source archive, treated as empty GT; {len(extra_labels)} unused labels.',flush=True)
    patches={}
    for method in ['sg_ga','ga_center','random_search']:
        data=json.loads((run/f'seed_0_{method}.json').read_text(encoding='utf-8'))
        patches[method]=(torch.from_numpy(np.load(run/f'patch_0_{method}.npy')),data['location'])
    rng=np.random.default_rng(12345)
    patches['noise']=(torch.from_numpy(rng.random((3,64,64),dtype=np.float32)),patches['sg_ga'][1])
    grid=np.indices((64,64))
    # Preserve the primary run's diagonal pattern and add a true square checkerboard.
    patches['diagonal_stripes']=(torch.from_numpy(((grid.sum(0)//4)%2).astype(np.float32)).repeat(3,1,1),patches['sg_ga'][1])
    patches['checkerboard']=(torch.from_numpy(((grid[0]//4+grid[1]//4)%2).astype(np.float32)).repeat(3,1,1),patches['sg_ga'][1])
    patches['solid']=(torch.full((3,64,64),.5),patches['sg_ga'][1])
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),'dataset_source':'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip',
              'dataset_archive_sha256':sha256(ROOT/'outputs/datasets/coco128.zip'),'script_sha256':sha256(__file__),
              'optimizer_manifest_sha256':sha256(run/'manifest.json'),'selected_patch_seed':0,
              'images':{p.name:{'sha256':sha256(p),'label_file_present':(label_dir/p.with_suffix('.txt').name).exists(),
                               'label_sha256':sha256(label_dir/p.with_suffix('.txt').name) if (label_dir/p.with_suffix('.txt').name).exists() else None} for p in paths},
              'annotation_ledger':{'missing_label_files':missing_labels,'unused_label_files':extra_labels,
                                   'missing_policy':'empty ground truth, matching installed Ultralytics verify_image_label; source archive absence verified'},
              'protocol':{'input_hw':[320,320],'patch_hw':[64,64],'threads':2,'forward_repeats_per_condition_image':3,
                          'forward_warmup_per_condition_image':1,
                          'postprocess_repeats_per_configuration':5,'postprocess_warmup':2,'iou_nms':.45,'max_det':300,
                          'threshold_cap_pairs':[[.01,None],[.25,None],[.01,100]],'class_agnostic':True,'patch_location':'fixed from bus optimization; not reoptimized on COCO128',
                          'conditions':['clean','sg_ga','ga_center','random_search','noise','diagonal_stripes','checkerboard','solid'],
                          'diagonal_stripes':'(row+column)//4 modulo 2; called checkerboard in primary raw records',
                          'checkerboard':'(row//4+column//4) modulo 2; true 4x4-pixel square tiles'},
              'limitations':['COCO128 is a subset of train2017, potentially in pretrained YOLO training data',
                             'seed 0 predeclared representative patches; no patch-seed population inference',
                             'diagnostic 101-point AP, not official pycocotools COCO evaluator',
                             'confidence-filter-conditional AP; .25 results are not standard low-threshold COCO mAP',
                             'source archive has two images without label files; these are treated as empty GT, not independently reannotated',
                             'forward and postprocessing measured separately on cached predictions; no end-to-end FPS inferred']}
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ['protocol','script_sha256','optimizer_manifest_sha256','dataset_archive_sha256','images']:
            if previous[key]!=manifest[key]:
                raise RuntimeError(f'Existing output has different {key}; use a new output directory.')
        manifest=previous
    else:
        save_json(manifest_path,manifest)
    victim=VictimModel(device='cpu')
    first=load_image(paths[0])
    for _ in range(10):
        victim.get_raw_predictions(first)
    all_rows=[]
    for idx,path in enumerate(paths):
        checkpoint=out/f'image_{path.stem}.json'
        if checkpoint.exists():
            all_rows.extend(json.loads(checkpoint.read_text(encoding='utf-8')))
            continue
        base=load_image(path); controls={'clean':base}
        for method,(patch,location) in patches.items():
            x=base.clone(); y,z=location; x[:,:,y:y+64,z:z+64]=patch; controls[method]=x
        target=labels_for(dataset/'labels/train2017'/path.with_suffix('.txt').name).tolist()
        names=list(controls); random.Random(idx+40000).shuffle(names)
        rows=[]
        for name in names:
            image=controls[name]; forward=[]
            with torch.no_grad():
                # One warm-up for each input, separate from the 3 recorded forward queries.
                victim.model(image)
                for _ in range(3):
                    start=time.perf_counter_ns(); raw=victim.model(image)
                    forward.append((time.perf_counter_ns()-start)/1e6)
                raw=raw[0] if isinstance(raw,(tuple,list)) else raw
                pred=raw[0]
            configs=[(.01,None),(.25,None),(.01,100)]
            random.Random(idx+50000).shuffle(configs)
            for tau,cap in configs:
                for _ in range(2): postprocess(victim,pred,tau,cap)
                samples=[postprocess(victim,pred,tau,cap) for _ in range(5)]
                row={k:v for k,v in samples[-1].items() if k not in ['nms_ms','filter_ms','postprocessing_ms']}
                for metric in ['nms_ms','filter_ms','postprocessing_ms']:
                    row[metric]=float(np.median([s[metric] for s in samples]))
                row.update({'image':path.name,'condition':name,'threshold':tau,'cap':cap,
                            'raw_locations':pred.shape[1],'targets':target,'forward_ms':float(np.median(forward)),
                            'forward_ms_samples':forward,'nms_ms_samples':[s['nms_ms'] for s in samples]})
                rows.append(row)
        save_json(checkpoint,rows); all_rows.extend(rows)
        if (idx+1)%8==0: print(f'COCO128 {idx+1}/128 images saved',flush=True)
    aggregate=[]
    for condition in controls:
        for tau,cap in [(.01,None),(.25,None),(.01,100)]:
            rows=[r for r in all_rows if r['condition']==condition and r['threshold']==tau and r['cap']==cap]
            aggregate.append({'condition':condition,'threshold':tau,'cap':cap,'metrics':detection_metrics(rows),
                              **{k:summary([r[k] for r in rows]) for k in ['active_candidates','nms_input_candidates','final_detections','nms_ms','postprocessing_ms','forward_ms']}})
    save_json(out/'aggregate.json',aggregate)
    flat=[]
    for record in aggregate:
        row={k:record[k] for k in ['condition','threshold','cap']}
        row.update(record['metrics'])
        for metric in ['active_candidates','nms_input_candidates','final_detections','nms_ms','postprocessing_ms','forward_ms']:
            row.update({f'{metric}_{stat}':record[metric][stat] for stat in ['mean','std','median','q1','q3']})
        flat.append(row)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=['clean','noise','diagonal_stripes','checkerboard','solid','sg_ga','ga_center','random_search']
    labels=['Sạch','Nhiễu','Sọc chéo','Ô bàn cờ','Màu đặc','GA nổi bật','GA trung tâm','Ngẫu nhiên']
    for ax,key,ylabel in [(axes[0],'recall_iou50','Độ bao phủ tại IoU 0,5'),(axes[1],'AP50_101point_diagnostic','AP50 chẩn đoán')]:
        for tau,cap,offset,label in [(.01,None,-.25,'Ngưỡng 0,01'),(.25,None,0,'Ngưỡng 0,25'),(.01,100,.25,'0,01 + top-100')]:
            vals=[next(r['metrics'][key] for r in aggregate if r['condition']==name and r['threshold']==tau and r['cap']==cap) for name in names]
            ax.bar(np.arange(len(vals))+offset,vals,.25,label=label)
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right'); ax.set_ylabel(ylabel); ax.set_ylim(0,1); ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'accuracy_defense.png',dpi=220); plt.close(fig)
    manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
    manifest['completion']={'images':128,'conditions':len(controls),'configurations':3,'records':len(all_rows)}
    save_json(out/'manifest.json',manifest)
    print(f'COCO128 COMPLETE: {out}',flush=True)


if __name__=='__main__':
    main()
