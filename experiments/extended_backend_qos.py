"""Same-data NMS backend comparison and real prepared-tensor processing budgets.

This is one physical laptop with several CPU thread settings, NOT several
devices. FIFO/latest-frame quantities are explicit offline queue simulations
from measured service times, NOT camera frame-age measurements.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import statistics
import time
from datetime import datetime, timezone

from extended_validation import ROOT, find_locations, insert
from extended_evaluation import COCO, decode, postprocess
from local_q1_validation import VictimModel, load_image, save_json, sha256
import numpy as np
import psutil
import torch


def queue_model(service_ms, period_ms, policy='fifo'):
    finish = 0.
    rows = []
    dropped = 0
    next_available = 0
    for index,duration in enumerate(service_ms):
        arrival = index*period_ms
        if policy == 'drop_if_busy' and arrival < finish:
            dropped += 1
            continue
        start = max(arrival,finish)
        finish = start+duration
        age = finish-arrival
        rows.append(dict(frame=index,arrival_ms=arrival,start_ms=start,finish_ms=finish,
                         modeled_age_ms=age,modeled_wait_ms=start-arrival,late=age>period_ms))
        next_available += 1
    return dict(policy=policy,arrival_period_ms=period_ms,generated=len(service_ms),processed=next_available,
                dropped=dropped,modeled_late_fraction=sum(r['late'] for r in rows)/max(1,len(rows)),
                max_modeled_age_ms=max([r['modeled_age_ms'] for r in rows],default=0),
                max_modeled_wait_ms=max([r['modeled_wait_ms'] for r in rows],default=0),records=rows,
                actual_camera_measurement=False,queue_model_assumes_deterministic_arrivals=True)


def backend(victim,run,output):
    source = run/'evaluation'
    completed = json.loads((source/'manifest.json').read_text())
    if 'finished_utc' not in completed:
        raise RuntimeError('Finish evaluation before using saved predictions')
    rows = []
    paths = []
    rng = random.Random(771105)
    for seed in range(10):
        for name in ['clean','sg_ga']:
            case = f'{name}_seed{seed}_q{0 if name=="clean" else 2500}'
            files = sorted((source/'raw_cache'/case).glob('*.npz'),key=lambda p:int(p.stem))[:16]
            if len(files) != 16:
                raise RuntimeError('Missing actual saved predictions: '+case)
            for path in files:
                paths.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path)))
                decoded = torch.from_numpy(np.load(path)['decoded'])
                configs = [(tau,cap,engine) for tau,cap in [(.01,None),(.25,None),(.01,100)] for engine in ['custom','torchvision']]
                rng.shuffle(configs)
                for tau,cap,engine in configs:
                    postprocess(victim,decoded,tau,cap,engine)
                    samples = [postprocess(victim,decoded,tau,cap,engine)[0] for _ in range(5)]
                    _,found = postprocess(victim,decoded,tau,cap,engine)
                    _,other = postprocess(victim,decoded,tau,cap,'torchvision' if engine=='custom' else 'custom')
                    # Distinct backend numerical behavior/ties need not produce
                    # identical indices. Record it rather than silently assume.
                    equal = found.shape == other.shape and bool(torch.equal(found,other))
                    rows.append(dict(seed=seed,condition=name,image_id=int(path.stem),threshold=tau,pre_nms_cap=cap,
                                     backend=engine,outputs_exactly_equal=equal,samples=samples,
                                     nms_ms=statistics.median([s['nms_ms'] for s in samples]),
                                     postprocess_ms=statistics.median([s['postprocess_ms'] for s in samples]),
                                     nms_input_candidates=samples[0]['nms_input_candidates']))
    save_json(output/'backend.json',dict(records=rows,input_files=paths,
                                        source_evaluator_manifest_sha256=sha256(source/'manifest.json'),
                                        equal_scores_or_iou_rounding_may_change_ties=True,
                                        interpretation='paired backend comparison on identical actual decoded model predictions'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    run = ROOT/args.run_dir
    manifest = json.loads((run/'manifest.json').read_text())
    if 'optimization_finished_utc' not in manifest:
        raise RuntimeError('Finish optimization before CPU latency tests')
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'],key=lambda i:i['id'])[:16]
    output = run/'backend_qos'
    output.mkdir(exist_ok=True)
    config = dict(script_sha256=sha256(__file__),optimizer_manifest_sha256=sha256(run/'manifest.json'),
                  evaluator_sha256=sha256(ROOT/'experiments/extended_evaluation.py'),
                  dataset_sha256=sha256(data/'manifest.json'),seeds=10,test_ids=[i['id'] for i in infos],
                  cpu_thread_settings=[1,2,4],physical_devices=1,
                  conditions=['clean','sg_ga','ga_center','ga_random_location','random_search'],
                  budget=2500,threshold=.01,nms_iou=.45,pre_nms_cap=None,processing_deadline_ms=300,
                  repeats=3,warmup=1,order_seed=100317,backend_repeats=5,backend_cpu_threads=2,
                  boundary='single timed model+decode+filter+NMS on prepared tensor; no camera, IO or patch placement',
                  qos_scope='offline processing-budget violations, not sensor-to-output deadlines',
                  queue_simulation_periods_ms=[100,200,300],queue_models_not_camera_evidence=True)
    check = output/'manifest.json'
    if check.exists() and json.loads(check.read_text())['protocol'] != config:
        raise RuntimeError('Changed timing protocol/source')
    if not check.exists():
        save_json(check,dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol=config))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    victim = VictimModel(device='cpu')
    process = psutil.Process()
    if not (output/'backend.json').exists():
        backend(victim,run,output)
    images = {i['id']:load_image(data/'images'/i['file_name']) for i in infos}
    blocks = [(seed,threads) for seed in range(10) for threads in [1,2,4]]
    random.Random(config['order_seed']).shuffle(blocks)
    for seed,threads in blocks:
        target = output/f'timing_seed{seed}_threads{threads}.json'
        if target.exists():
            continue
        torch.set_num_threads(threads)
        patches = {name:torch.from_numpy(np.load(run/'optimization'/f'{name}_seed{seed}'/'patch_q2500.npy'))
                   for name in config['conditions'] if name!='clean'}
        conditions = config['conditions'].copy()
        rng = random.Random(473187+seed*10+threads)
        rng.shuffle(conditions)
        records = []
        for name in conditions:
            order = infos.copy()
            rng.shuffle(order)
            cpu_before = process.cpu_times()
            wall_started = time.perf_counter()
            for info in order:
                image = images[info['id']]
                location = None
                if name != 'clean':
                    mode = 'center' if name=='ga_center' else 'random' if name=='ga_random_location' else 'saliency'
                    location = find_locations([image],64,mode,seed+info['id'])[0]
                    image = insert(image,patches[name][None],[location])
                with torch.no_grad():
                    raw = victim.model(image)
                    raw = raw[0] if isinstance(raw,(tuple,list)) else raw
                    postprocess(victim,decode(victim,raw[0]))
                    for repeat in range(3):
                        started = time.perf_counter_ns()
                        raw = victim.model(image)
                        raw = raw[0] if isinstance(raw,(tuple,list)) else raw
                        forwarded = time.perf_counter_ns()
                        decoded = decode(victim,raw[0])
                        decoded_at = time.perf_counter_ns()
                        stats,_ = postprocess(victim,decoded)
                        ended = time.perf_counter_ns()
                        total = (ended-started)/1e6
                        records.append(dict(condition=name,image_id=info['id'],repeat=repeat,location=location,
                                            forward_ms=(forwarded-started)/1e6,decode_ms=(decoded_at-forwarded)/1e6,
                                            total_prepared_tensor_ms=total,late_processing_budget=total>300,**stats))
            wall = time.perf_counter()-wall_started
            cpu_after = process.cpu_times()
            cpu_pct = ((cpu_after.user-cpu_before.user)+(cpu_after.system-cpu_before.system))/wall*100
            selected = [r for r in records if r['condition']==name]
            values = [r['total_prepared_tensor_ms'] for r in selected]
            for r in selected:
                r['condition_block_cpu_percent_one_core_scale'] = cpu_pct
                r['condition_block_cpu_percent_machine_normalized'] = cpu_pct/psutil.cpu_count()
                r['rss_mib_at_condition_end'] = process.memory_info().rss/1024**2
            modeled = [queue_model(values,period,policy) for period in [100,200,300] for policy in ['fifo','drop_if_busy']]
            save_json(output/f'queue_seed{seed}_threads{threads}_{name}.json',dict(modeled=modeled,
                      measured_service_order='actual shuffled image order, three repeats consecutively',
                      actual_streaming_experiment=False))
        save_json(target,dict(seed=seed,cpu_threads=threads,records=records,
                              patch_sha256={n:sha256(run/'optimization'/f'{n}_seed{seed}'/'patch_q2500.npy') for n in patches},
                              temperature_available=False,temperature_not_substituted_with_zero=True))
        print(f'QOS seed {seed} threads {threads} complete',flush=True)
    final = json.loads(check.read_text())
    final['finished_utc'] = datetime.now(timezone.utc).isoformat()
    final['completed_blocks'] = len(blocks)
    save_json(check,final)


if __name__ == '__main__':
    main()
