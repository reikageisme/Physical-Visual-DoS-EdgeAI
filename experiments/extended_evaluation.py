"""Sequential, checkpointed evaluation of the predeclared extension.

COCOeval is used with original category IDs, boxes and crowd/ignore metadata.
AP is official evaluator output on a fixed SUBSET, not full COCO val2017.
No optimization or patch selection uses these evaluation results.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from extended_validation import ROOT, find_locations, insert, make_eot, objective
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from local_q1_validation import VictimModel, load_image, save_json, sha256, summary


def decode(victim, raw):
    scores, classes = raw[4:].max(dim=0)
    boxes = victim._xywh_to_xyxy(raw[:4].T.float())
    return torch.cat([boxes, scores[:, None], classes[:, None].float()], 1)


def postprocess(victim, decoded, threshold=.01, cap=None, backend='custom'):
    started = time.perf_counter_ns()
    active = decoded[decoded[:, 4] > threshold]
    count = len(active)
    if cap is not None and count > cap:
        active = active[active[:, 4].topk(cap).indices]
    nms_start = time.perf_counter_ns()
    boxes, scores = active[:, :4], active[:, 4]
    indices = victim._nms(boxes, scores, .45) if backend == 'custom' else torchvision.ops.nms(boxes, scores, .45)
    result = active[indices[:300]]
    finished = time.perf_counter_ns()
    return dict(active_candidates=count, nms_input_candidates=len(active), final_detections=len(result),
                filter_ms=(nms_start-started)/1e6, nms_ms=(finished-nms_start)/1e6,
                postprocess_ms=(finished-started)/1e6), result


def coco_metrics(coco, predictions, ids):
    if not predictions:
        dt = COCO()
        dt.dataset = {'images':coco.dataset['images'], 'categories':coco.dataset['categories'], 'annotations':[]}
        with contextlib.redirect_stdout(io.StringIO()):
            dt.createIndex()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = coco.loadRes(predictions)
    evaluator = COCOeval(coco, dt, 'bbox')
    evaluator.params.imgIds = list(ids)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    keys = ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    return dict(zip(keys, evaluator.stats.tolist()))


def official_predictions(found, info, categories):
    rows = []
    sx, sy = info['width']/320, info['height']/320
    for x1, y1, x2, y2, score, cls in found.cpu().tolist():
        # Clip to source image extent before converting xyxy to COCO xywh.
        x1, x2 = np.clip([x1, x2], 0, 320)
        y1, y2 = np.clip([y1, y2], 0, 320)
        rows.append(dict(image_id=info['id'], category_id=categories[int(cls)],
                         bbox=[float(x1*sx), float(y1*sy), float(max(0,x2-x1)*sx), float(max(0,y2-y1)*sy)],
                         score=float(score)))
    return rows


def cases(run, seeds):
    rows = [dict(name='clean', seed=seed, budget=0, spec=None, patch=None) for seed in range(seeds)]
    for path in sorted((run/'optimization').glob('*/complete.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for checkpoint in data['checkpoints']:
            budget = checkpoint['queries']
            rows.append(dict(name=data['spec']['name'], seed=data['seed'], budget=budget,
                             spec=data['spec'], patch=path.parent/f'patch_q{budget}.npy'))
    return rows


def evaluate_case(victim, case, infos, coco, data_folder, destination):
    if destination.exists():
        return json.loads(destination.read_text(encoding='utf-8'))
    ids = [i['id'] for i in infos]
    categories = sorted(coco.getCatIds())
    patch = None if case['patch'] is None else torch.from_numpy(np.load(case['patch']))
    configs = [(.01, None), (.25, None), (.01, 100)]
    if case['name'] in ['clean', 'sg_ga', 'adaptive_cap100'] and (case['budget'] == 2500 or case['name'] != 'sg_ga'):
        configs = [(tau, cap) for tau in [.01, .05, .10, .25, .50] for cap in [None, 50, 100, 300]]
    predictions = {str(config):[] for config in configs}
    records = []
    cache_folder = destination.parent/'raw_cache'/destination.stem
    cache_folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(67129+case['seed'])
    ordered = infos.copy()
    rng.shuffle(ordered)
    for info in ordered:
        image = load_image(data_folder/'images'/info['file_name'])
        location = None
        if patch is not None:
            location = find_locations([image], patch.shape[-1], case['spec']['location'], case['seed']+info['id'])[0]
            image = insert(image, patch[None], [location])
        cache = cache_folder/f'{info["id"]}.npz'
        if cache.exists():
            saved = np.load(cache)
            decoded = torch.from_numpy(saved['decoded'])
            forward_ms = saved['forward_ms'].tolist()
        else:
            for _ in range(1):
                with torch.no_grad():
                    victim.model(image)
            forward_ms = []
            with torch.no_grad():
                for _ in range(3):
                    started = time.perf_counter_ns()
                    raw = victim.model(image)
                    raw = raw[0] if isinstance(raw, (tuple, list)) else raw
                    forward_ms.append((time.perf_counter_ns()-started)/1e6)
                decoded = decode(victim, raw[0]).cpu()
            np.savez_compressed(cache, decoded=decoded.numpy(), forward_ms=forward_ms)
        order = configs.copy()
        rng.shuffle(order)
        for tau, cap in order:
            postprocess(victim, decoded, tau, cap)
            samples = [postprocess(victim, decoded, tau, cap)[0] for _ in range(3)]
            _, detections = postprocess(victim, decoded, tau, cap)
            key = str((tau, cap))
            predictions[key].extend(official_predictions(detections, info, categories))
            med = {k:statistics.median([s[k] for s in samples]) for k in samples[0]}
            med.update(image_id=info['id'], threshold=tau, pre_nms_cap=cap, location=location,
                       forward_ms=statistics.median(forward_ms), forward_ms_samples=forward_ms,
                       postprocess_samples=samples)
            records.append(med)
    metrics = []
    for tau, cap in configs:
        selected = [r for r in records if r['threshold'] == tau and r['pre_nms_cap'] == cap]
        ap = coco_metrics(coco, predictions[str((tau, cap))], ids)
        metrics.append(dict(threshold=tau, pre_nms_cap=cap, coco_subset_bbox=ap,
                            **{k:summary([r[k] for r in selected]) for k in ['active_candidates','nms_input_candidates','final_detections','forward_ms','nms_ms','postprocess_ms']}))
    result = dict(case={k:str(v) if isinstance(v, Path) else v for k,v in case.items()},
                  patch_sha256=None if case['patch'] is None else sha256(case['patch']),
                  images=len(ids), image_ids=ids, records=records, metrics=metrics,
                  predictions=predictions, evaluator='pycocotools.COCOeval bbox, maxDets 1/10/100, original crowd/ignore',
                  timing_boundary='model forward and postprocess measured separately; no camera, decode or patch placement',
                  total_latency_not_sum_of_medians=True)
    save_json(destination, result)
    return result


def independent_eot(victim, case, infos, data_folder, destination):
    if destination.exists() or case['patch'] is None:
        return
    seed = 980123+case['seed']
    torch.manual_seed(seed)
    transform = make_eot('full')
    patch = torch.from_numpy(np.load(case['patch']))
    records = []
    for info in infos[:8]:
        clean = load_image(data_folder/'images'/info['file_name'])
        location = find_locations([clean], patch.shape[-1], case['spec']['location'], case['seed']+info['id'])[0]
        adv = insert(clean, patch[None], [location])
        values = []
        for draw in range(16):
            # The same deterministic draw seed is used for clean/patched image;
            # resetting it matches transform parameters and sensor-noise sample.
            draw_seed = seed+info['id']*100+draw
            torch.manual_seed(draw_seed)
            with torch.no_grad():
                clean_score = objective(victim.get_raw_predictions(transform(clean))).item()
            torch.manual_seed(draw_seed)
            with torch.no_grad():
                adv_score = objective(victim.get_raw_predictions(transform(adv))).item()
            values.append(dict(draw=draw, clean_fitness=clean_score, patch_fitness=adv_score, difference=adv_score-clean_score))
        records.append(dict(image_id=info['id'], samples=values))
    save_json(destination, dict(seed=case['seed'], case=case['name'], queries=case['budget'],
                               independent_draws=16, images=len(records), paired_transform_parameters=True, records=records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    run = ROOT/args.run_dir
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if 'optimization_finished_utc' not in manifest:
        raise RuntimeError('Run evaluation sequentially AFTER all optimization finishes on this CPU')
    data_folder = ROOT/'outputs/datasets/coco_val2017_extended'
    dataset = json.loads((data_folder/'manifest.json').read_text(encoding='utf-8'))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data_folder/'instances_subset.json'))
    infos = sorted(coco.dataset['images'], key=lambda x:x['id'])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    victim = VictimModel(device='cpu')
    output = run/('evaluation_pilot' if args.pilot else 'evaluation')
    output.mkdir(exist_ok=True)
    config = dict(script_sha256=sha256(__file__), dataset_sha256=sha256(data_folder/'manifest.json'),
                  optimizer_manifest_sha256=sha256(run/'manifest.json'), pilot=args.pilot,
                  case_order_seed=981561, test_images=len(infos), eot_test_images=8, independent_eot_draws=16,
                  timing_repeats=3, timing_warmup=1, cpu_threads=2,
                  budget_1000_images=32, other_budgets_images=len(infos),
                  latency_excludes_dataset_loading_and_patch_placement=True,
                  upstream_model_selection_overlap=dataset['upstream_model_selection_overlap'])
    check = output/'manifest.json'
    if check.exists() and json.loads(check.read_text(encoding='utf-8'))['protocol'] != config:
        raise RuntimeError('Evaluation provenance changed; do not mix records')
    if not check.exists():
        save_json(check, dict(started_utc=datetime.now(timezone.utc).isoformat(), protocol=config))
    all_cases = cases(run, manifest['protocol']['seeds'])
    random.Random(config['case_order_seed']).shuffle(all_cases)
    if args.pilot:
        all_cases = all_cases[:2]
        infos = infos[:2]
    for index, case in enumerate(all_cases):
        selection = infos if case['budget'] in [0, 400] or case['budget'] == max(case['spec']['budgets']) else infos[:32]
        name = f'{case["name"]}_seed{case["seed"]}_q{case["budget"]}'
        evaluate_case(victim, case, selection, coco, data_folder, output/f'{name}.json')
        if case['budget'] and case['budget'] == max(case['spec']['budgets']):
            independent_eot(victim, case, infos, data_folder, output/f'eot_{name}.json')
        print(f'EVALUATED {index+1}/{len(all_cases)}: {name}', flush=True)
    final = json.loads(check.read_text(encoding='utf-8'))
    final['finished_utc'] = datetime.now(timezone.utc).isoformat()
    final['evaluated_cases'] = len(all_cases)
    save_json(check, final)
    print('EVALUATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
