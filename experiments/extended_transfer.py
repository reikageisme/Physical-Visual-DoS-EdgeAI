"""Actual paired cross-model evaluation; never uses random image/weight fallback.

All models receive the same prepared 320x320 RGB scene and local patch.
Native normalization/postprocessing is included inside the measured call;
camera, file decoding and patch placement are excluded. Compare conditions
WITHIN each model, not absolute latency ratios between different pipelines.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch as mock_patch

from extended_validation import ROOT, find_locations, insert
sys.path.insert(0, str(ROOT/'outputs/runtime/transfer'))
import numpy as np
import torch
import torchvision
from extended_evaluation import COCO, coco_metrics, decode, postprocess
from local_q1_validation import VictimModel, load_image, save_json, sha256
from prepare_extended_validation import download

CACHE = ROOT/'outputs/weights/extended_transfer'
MODELS = ['yolov8n', 'yolov8s', 'ssdlite', 'fasterrcnn', 'retinanet', 'detr']
TV = {
    'ssdlite': (torchvision.models.detection.ssdlite320_mobilenet_v3_large,
                torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT),
    'fasterrcnn': (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
                  torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    'retinanet': (torchvision.models.detection.retinanet_resnet50_fpn,
                 torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT),
}


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, (_, weights) in TV.items():
        target = CACHE/Path(weights.url).name
        download(weights.url, target)
        files[name] = dict(path=str(target), url=weights.url, sha256=sha256(target))
        print('WEIGHTS READY: '+name, flush=True)
    url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt'
    target = CACHE/'yolov8s.pt'
    download(url, target)
    files['yolov8s'] = dict(path=str(target), url=url, sha256=sha256(target))
    files['yolov8n'] = dict(path=str(ROOT/'yolov8n.pt'), url=None, sha256=sha256(ROOT/'yolov8n.pt'))
    from huggingface_hub import HfApi, snapshot_download
    previous = CACHE/'weights_manifest.json'
    revision = None
    if previous.exists():
        revision = json.loads(previous.read_text())['detr']['revision']
    revision = revision or HfApi().model_info('facebook/detr-resnet-50').sha
    folder = CACHE/'detr'
    snapshot_download('facebook/detr-resnet-50', revision=revision, local_dir=folder,
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors'])
    files['detr'] = dict(path=str(folder), repository='facebook/detr-resnet-50', revision=revision,
                         files={n:sha256(folder/n) for n in ['config.json','preprocessor_config.json','model.safetensors']})
    save_json(previous, files)
    return files


def build(name, weights):
    if name.startswith('yolov8'):
        from ultralytics import YOLO
        return YOLO(weights[name]['path']).model.float().cpu().eval()
    if name in TV:
        builder, _ = TV[name]
        kwargs = dict(weights=None, weights_backbone=None)
        if name == 'fasterrcnn':
            kwargs.update(min_size=320, max_size=320, box_score_thresh=.01,
                          box_nms_thresh=.45, box_detections_per_img=300)
        elif name == 'retinanet':
            kwargs.update(min_size=320, max_size=320, score_thresh=.01,
                          nms_thresh=.45, detections_per_img=300)
        else:
            kwargs.update(score_thresh=.01, nms_thresh=.45, detections_per_img=300)
        model = builder(**kwargs)
        state = torch.load(weights[name]['path'], map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.float().cpu().eval()
    from transformers import DetrForObjectDetection
    model, info = DetrForObjectDetection.from_pretrained(weights[name]['path'], local_files_only=True,output_loading_info=True)
    unexpected = [key for key in info.get('unexpected_keys',[]) if not key.endswith('num_batches_tracked')]
    if info.get('missing_keys') or info.get('mismatched_keys') or unexpected:
        raise RuntimeError('DETR checkpoint does not fully initialize model: '+str(info))
    model._checkpoint_loading_info = {key:sorted(value) if isinstance(value,set) else value for key,value in info.items()}
    return model.float().cpu().eval()


def predict(name, model, image):
    events = []
    original_nms = torchvision.ops.boxes.batched_nms

    def traced_nms(boxes, scores, classes, threshold):
        started = time.perf_counter_ns()
        kept = original_nms(boxes, scores, classes, threshold)
        events.append(dict(input_candidates=len(boxes), output_boxes=len(kept),
                           iou_threshold=float(threshold), duration_ms=(time.perf_counter_ns()-started)/1e6))
        return kept

    started = time.perf_counter_ns()
    with torch.no_grad():
        if name.startswith('yolov8'):
            raw = model(image)
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            stats, found = postprocess(VictimModel, decode(VictimModel, raw[0]), .01)
            # YOLO has 80 contiguous classes; map later using official categories.
            boxes, scores, labels = found[:,:4], found[:,4], found[:,5].long()
            events.append(dict(input_candidates=stats['nms_input_candidates'],
                               output_boxes=len(found), iou_threshold=.45,
                               duration_ms=stats['nms_ms']))
        elif name in TV:
            # This also captures Faster R-CNN proposal NMS; keep each call
            # separate instead of calling every intermediate count 'detections'.
            with mock_patch.object(torchvision.ops.boxes, 'batched_nms', traced_nms):
                result = model([image[0]])[0]
            boxes, scores, labels = result['boxes'], result['scores'], result['labels']
        else:
            mean = image.new_tensor([.485,.456,.406])[None,:,None,None]
            std = image.new_tensor([.229,.224,.225])[None,:,None,None]
            result = model(pixel_values=(image-mean)/std)
            scores, labels = result.logits.softmax(-1)[0,:,:-1].max(-1)
            keep = scores > .01
            xywh = result.pred_boxes[0,keep]*320
            boxes = VictimModel._xywh_to_xyxy(xywh)
            scores, labels = scores[keep], labels[keep]
            # DETR query slots are not pre-NMS candidates; no NMS is called.
    elapsed = (time.perf_counter_ns()-started)/1e6
    return dict(total_prepared_tensor_ms=elapsed, nms_calls=events,
                nms_ms=sum(e['duration_ms'] for e in events) if events else None,
                final_detections=len(boxes), has_nms=name != 'detr'), (boxes,scores,labels)


def predictions(found, info, name, categories):
    boxes, scores, labels = [x.cpu().tolist() for x in found]
    rows = []
    for box, score, label in zip(boxes, scores, labels):
        category = categories[label] if name.startswith('yolov8') else label
        if category not in categories:
            continue
        x1,y1,x2,y2 = np.clip(box, 0, 320)
        rows.append(dict(image_id=info['id'], category_id=category, score=score,
                         bbox=[float(x1*info['width']/320), float(y1*info['height']/320),
                               float(max(0,x2-x1)*info['width']/320), float(max(0,y2-y1)*info['height']/320)]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke', action='store_true',help='real-image program check; never scientific latency results')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    if args.smoke:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        weights = json.loads((CACHE/'weights_manifest.json').read_text())
        files = sorted((ROOT/'outputs/datasets/coco_val2017_extended/images').glob('*.jpg'))
        if not files:
            raise RuntimeError('A real prepared COCO image is required')
        image = load_image(files[0])
        rows = []
        for name in MODELS:
            model = build(name,weights)
            stats, found = predict(name,model,image)
            assert torch.isfinite(found[0]).all() and found[0].shape[-1]==4
            rows.append(dict(model=name,returned_detections=len(found[0]),has_nms=stats['has_nms'],
                             checkpoint_loading_info=getattr(model,'_checkpoint_loading_info',None)))
            print('REAL-IMAGE PROGRAM CHECK: '+name,flush=True)
            del model
            gc.collect()
        save_json(CACHE/'smoke.json',dict(image_sha256=sha256(files[0]),checks=rows,
                                         script_sha256=sha256(__file__),not_scientific_latency_results=True,
                                         optimization_may_be_running_concurrently=True))
        return
    run = ROOT/args.run_dir
    optimizer = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if 'optimization_finished_utc' not in optimizer:
        raise RuntimeError('Finish optimization before latency measurement')
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'], key=lambda i:i['id'])[:16]
    categories = sorted(coco.getCatIds())
    weights = json.loads((CACHE/'weights_manifest.json').read_text())
    for name,record in weights.items():
        if name == 'detr':
            for file,expected in record['files'].items():
                if sha256(Path(record['path'])/file) != expected:
                    raise RuntimeError('Changed DETR weight/config file')
        elif sha256(record['path']) != record['sha256']:
            raise RuntimeError('Changed pretrained weight: '+name)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output = run/'transfer'
    output.mkdir(exist_ok=True)
    config = dict(script_sha256=sha256(__file__), optimizer_manifest_sha256=sha256(run/'manifest.json'),
                  dataset_manifest_sha256=sha256(data/'manifest.json'),
                  weights_manifest_sha256=sha256(CACHE/'weights_manifest.json'),
                  models=MODELS, test_ids=[i['id'] for i in infos], source='sg_ga_q2500',
                  source_seeds=optimizer['protocol']['seeds'], warmup=1, repeats=3, cpu_threads=2,
                  image_size=320, score_threshold=.01, nms_iou=.45, model_order_seed=46710,
                  inference_boundary='prepared RGB tensor to returned detections incl native normalization/resize/postprocess',
                  nms_instrumentation='batched_nms call timing; includes proposal calls separately',
                  interpretation='within-model paired latency; subset AP diagnostic, not architecture-wide proof')
    destination = output/'manifest.json'
    if destination.exists() and json.loads(destination.read_text())['protocol'] != config:
        raise RuntimeError('Transfer protocol/source changed')
    if not destination.exists():
        save_json(destination, dict(started_utc=datetime.now(timezone.utc).isoformat(), protocol=config,
                                     weights=weights, torch=torch.__version__, torchvision=torchvision.__version__))
    names = MODELS.copy()
    random.Random(config['model_order_seed']).shuffle(names)
    if args.model:
        names = [args.model]
    for name in names:
        model = build(name, weights)
        save_json(output/f'{name}_loading.json',dict(model=name,
                   checkpoint_loading_info=getattr(model,'_checkpoint_loading_info',None),
                   frozen_bn_tracking_counters_may_be_unused=True))
        for seed in range(config['source_seeds']):
            target = output/f'{name}_seed{seed}.json'
            if target.exists():
                continue
            patch_path = run/'optimization'/f'sg_ga_seed{seed}'/'patch_q2500.npy'
            patch = torch.from_numpy(np.load(patch_path))
            records = []
            predicted = dict(clean=[], patched=[])
            rng = random.Random(71071+seed)
            order = infos.copy()
            rng.shuffle(order)
            for info in order:
                clean = load_image(data/'images'/info['file_name'])
                location = find_locations([clean], 64, 'saliency', seed+info['id'])[0]
                adv = insert(clean, patch[None], [location])
                conditions = [('clean',clean), ('patched',adv)]
                rng.shuffle(conditions)
                for condition, image in conditions:
                    predict(name, model, image)
                    samples = []
                    for repeat in range(3):
                        stats, found = predict(name, model, image)
                        samples.append(dict(repeat=repeat, **stats))
                    predicted[condition].extend(predictions(found, info, name, categories))
                    records.append(dict(image_id=info['id'], condition=condition, location=location,
                                        total_prepared_tensor_ms=statistics.median([s['total_prepared_tensor_ms'] for s in samples]),
                                        final_detections=stats['final_detections'], samples=samples))
            metrics = {c:coco_metrics(coco,predicted[c],[i['id'] for i in infos]) for c in predicted}
            save_json(target, dict(model=name, source_seed=seed, patch_sha256=sha256(patch_path),
                                   records=records, coco_subset_bbox=metrics, predictions=predicted))
            print(f'TRANSFER {name} seed {seed} complete', flush=True)
        del model
        gc.collect()
    final = json.loads(destination.read_text())
    count = len(list(output.glob('*_seed*.json')))
    final['completed_model_seed_cases'] = count
    if count == len(MODELS)*config['source_seeds']:
        final['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(destination, final)


if __name__ == '__main__':
    main()
