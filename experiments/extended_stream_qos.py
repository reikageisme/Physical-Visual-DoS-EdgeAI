"""Actual scheduled software input, inference worker and adaptive backpressure.

Unlike extended_backend_qos.queue_model, these are real concurrent executions
and monotonic timestamps, not a queue simulation. Prepared images replace a
camera: neither exposure nor camera latency is measured. All offered frames,
including discarded and late frames, remain in the accuracy denominator.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import random
import threading
import time

POLICIES = ('fifo', 'drop_if_busy', 'adaptive_throttle')
CONDITIONS = ('clean', 'sg_ga_q400', 'sg_ga_q2500', 'tanh_whitebox_patch_q400')
ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANIFEST = None


def checkpoint_path(run, condition, seed):
    if condition not in CONDITIONS or not 0 <= seed < 10:
        raise ValueError('Unknown predeclared patch condition or seed')
    if condition == 'clean':
        return None
    method, budget = condition.rsplit('_q',1)
    return Path(run)/'optimization'/f'{method}_seed{seed}'/f'patch_q{budget}.npy'


class Admission:
    """Causal admission control; no future service duration is available."""
    def __init__(self, base_ms, alpha=.25, safety=1.2, maximum_ms=600):
        if not 0 < base_ms <= maximum_ms or not 0 < alpha <= 1 or safety < 1:
            raise ValueError('Invalid predeclared controller settings')
        self.base_ms, self.alpha = base_ms, alpha
        self.safety, self.maximum_ms = safety, maximum_ms
        self.ewma_ms = None
        self.last_admission_ms = None

    @property
    def interval_ms(self):
        value = self.base_ms if self.ewma_ms is None else self.safety*self.ewma_ms
        return min(self.maximum_ms, max(self.base_ms, value))

    def observe(self, service_ms):
        if not math.isfinite(service_ms) or service_ms < 0:
            raise ValueError('Negative service duration')
        self.ewma_ms = service_ms if self.ewma_ms is None else (
            self.alpha*service_ms+(1-self.alpha)*self.ewma_ms)

    def admit(self, arrival_ms, busy):
        if busy:
            return False, 'busy'
        if self.last_admission_ms is not None and arrival_ms-self.last_admission_ms < self.interval_ms:
            return False, 'adaptive_interval'
        self.last_admission_ms = arrival_ms
        return True, None


def execute_stream(frames, callback, period_ms, policy, deadline_ms=300):
    """Fixed-rate offered stream, one actual worker; callback performs inference.

    FIFO is unbounded for this finite stream. Other policies admit only when
    neither a queued item nor a running callback exists. Adaptive feedback is
    published only after a callback actually completes. Slow callbacks are
    never skipped or assigned invented elapsed times.
    """
    if policy not in POLICIES or not frames or period_ms <= 0 or deadline_ms <= 0:
        raise ValueError('Invalid finite stream protocol')
    lock = threading.Condition()
    pending, records, failures = deque(), [], []
    state = {'closed': False, 'busy': False}
    controller = Admission(period_ms)
    epoch = time.perf_counter_ns()

    def worker():
        while True:
            with lock:
                lock.wait_for(lambda: bool(pending) or state['closed'])
                if not pending:
                    return
                frame, record = pending.popleft()
                state['busy'] = True
            started = time.perf_counter_ns()
            try:
                result = callback(frame)
                ended = time.perf_counter_ns()
            except BaseException as error:
                with lock:
                    failures.append(error)
                    state.update(closed=True, busy=False)
                    lock.notify_all()
                return
            service = (ended-started)/1e6
            record.update(status='processed', service_ms=service,
                          start_ms=(started-epoch)/1e6, finish_ms=(ended-epoch)/1e6,
                          actual_arrival_age_ms=(ended-record['arrival_ns'])/1e6,
                          scheduled_release_age_ms=(ended-epoch)/1e6-record['scheduled_release_ms'],
                          result=result)
            record['late'] = record['scheduled_release_age_ms'] > deadline_ms
            with lock:
                controller.observe(service)
                record['feedback_ewma_service_ms'] = controller.ewma_ms
                state['busy'] = False
                lock.notify_all()

    thread = threading.Thread(target=worker, name='measured-inference', daemon=True)
    thread.start()
    try:
        for index, frame in enumerate(frames):
            release_ns = epoch+round(index*period_ms*1e6)
            while True:
                remaining = (release_ns-time.perf_counter_ns())/1e9
                if remaining <= 0:
                    break
                with lock:
                    if failures:
                        raise RuntimeError('Inference worker failed') from failures[0]
                    lock.wait(timeout=min(remaining, .05))
            now = time.perf_counter_ns()
            arrival_ms = (now-epoch)/1e6
            with lock:
                if failures:
                    raise RuntimeError('Inference worker failed') from failures[0]
                occupied = state['busy'] or bool(pending)
                admitted, reason = True, None
                if policy == 'drop_if_busy' and occupied:
                    admitted, reason = False, 'busy'
                elif policy == 'adaptive_throttle':
                    admitted, reason = controller.admit(arrival_ms, occupied)
                row = dict(frame_index=index, arrival_ns=now,
                           scheduled_release_ms=index*period_ms, actual_arrival_ms=arrival_ms,
                           publisher_jitter_ms=arrival_ms-index*period_ms,
                           admission_interval_ms=controller.interval_ms,
                           queue_length_before_admission=len(pending),
                           worker_busy_at_arrival=state['busy'],
                           status='pending' if admitted else 'discarded', discard_reason=reason)
                records.append(row)
                if admitted:
                    pending.append((frame, row))
                    lock.notify_all()
    finally:
        with lock:
            state['closed'] = True
            lock.notify_all()
        thread.join()
    if failures:
        raise RuntimeError('Inference worker failed') from failures[0]
    elapsed_ms = (time.perf_counter_ns()-epoch)/1e6
    processed = [r for r in records if r['status'] == 'processed']
    on_time = [r for r in processed if not r['late']]
    assert len(records) == len(frames) and all(r['status'] != 'pending' for r in records)
    for row in records:
        row.pop('arrival_ns')
    # Include a full final arrival interval: a one-frame stream cannot attain
    # an arbitrarily large reported throughput just by finishing instantly.
    throughput_wall_ms = max(elapsed_ms, len(frames)*period_ms)
    return dict(policy=policy, period_ms=period_ms, deadline_ms=deadline_ms,
                offered_frames=len(frames), processed_frames=len(processed),
                discarded_frames=len(records)-len(processed), late_frames=len(processed)-len(on_time),
                late_fraction_among_processed=(len(processed)-len(on_time))/max(1,len(processed)),
                missing_or_late_fraction=1-len(on_time)/len(records),
                processed_fraction=len(processed)/len(records),
                elapsed_ms=elapsed_ms, throughput_denominator_ms=throughput_wall_ms,
                processed_fps=len(processed)*1000/throughput_wall_ms,
                on_time_fps=len(on_time)*1000/throughput_wall_ms,
                on_time_throughput_relative_to_offered=(len(on_time)*1000/throughput_wall_ms)/(1000/period_ms),
                actual_scheduled_execution=True, actual_camera_measurement=False, records=records)


def wait_for_pipeline(run, expected_pid):
    import psutil
    state = json.loads((run/'pipeline.json').read_text(encoding='utf-8'))
    if state['status'] == 'complete' and state.get('finished_utc'):
        return
    process = psutil.Process(expected_pid) if expected_pid is not None else None
    creation = process.create_time() if process is not None else None
    print('WAITING FOR EXISTING SOFTWARE PIPELINE; no inference started', flush=True)
    while True:
        state = json.loads((run/'pipeline.json').read_text(encoding='utf-8'))
        if state['status'] == 'complete' and state.get('finished_utc'):
            return
        if state['status'] in ('failed', 'source_changed'):
            raise RuntimeError('Preceding pipeline stopped: '+state['status'])
        if process is None:
            raise RuntimeError('Pipeline unfinished; supply its verified live PID to wait')
        try:
            if not process.is_running() or process.create_time() != creation:
                raise RuntimeError('Preceding pipeline handle ended without a completion record')
        except psutil.NoSuchProcess as error:
            raise RuntimeError('Preceding pipeline handle is missing') from error
        time.sleep(30)


def main():
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--wait-for-pipeline-pid', type=int)
    args = parser.parse_args()
    run = (ROOT/args.run_dir).resolve()
    if not run.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Output must remain within task outputs')
    # Heavy dependencies do not run model inference while waiting.
    from local_q1_validation import sha256, save_json
    output = run/'stream_qos'
    output.mkdir(exist_ok=True)
    config = dict(script_sha256=sha256(__file__),
                  sources_sha256={f:sha256(ROOT/'experiments'/f) for f in
                      ['extended_evaluation.py','extended_validation.py','local_q1_validation.py','extended_analysis.py']},
                  victim_source_sha256=sha256(ROOT/'core/victim_model.py'),
                  weights_sha256=sha256(ROOT/'yolov8n.pt'),
                  dataset_sha256=sha256(ROOT/'outputs/datasets/coco_val2017_extended/manifest.json'),
                  annotations_sha256=sha256(ROOT/'outputs/datasets/coco_val2017_extended/instances_subset.json'),
                  seeds=10, heldout_images=32, policies=list(POLICIES), arrival_periods_ms=[100,300],
                  deadline_ms=300, conditions=list(CONDITIONS),
                  cpu_threads=2, interop_threads=1, input_size=320, threshold=.01, nms_iou=.45,
                  pre_nms_cap=None, backend='custom', warmup_images=3, block_order_seed=630107,
                  controller=dict(alpha=.25,safety=1.2,maximum_ms=600),
                  boundary='prepared tensor release to model/decode/filter/NMS/COCO-conversion completion',
                  same_offered_schedule_across_policies=True,
                  dropped_frames_are_empty_predictions_for_full_stream_accuracy=True,
                  late_frames_are_empty_predictions_for_deadline_accuracy=True,
                  actual_camera_measurement=False,
                  hypothesis_role='secondary exploratory, added before its execution',
                  deterioration_gate=dict(min_on_time_throughput_ratio=.8,max_missing_or_late_fraction=.1),
                  recovery_gate=dict(max_AP_drop_from_paired_clean_fifo=.02),
                  gradient_is_local_adaptation_not_native_overload=True,
                  equal_area=True, equal_budget_comparison='SG-GA 400 vs gradient 400; 2500 is separate',
                  does_not_edit_manuscripts_or_publish=True)
    protocol_digest = hashlib.sha256(json.dumps(config,sort_keys=True).encode('utf-8')).hexdigest()
    manifest = output/'manifest.json'
    ACTIVE_MANIFEST = manifest
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['protocol'] != config:
            raise RuntimeError('Frozen stream protocol/source changed')
        if previous.get('finished_utc'):
            print('ACTUAL SOFTWARE STREAM STUDY ALREADY COMPLETE', flush=True)
            return
    else:
        save_json(manifest, dict(declared_utc=datetime.now(timezone.utc).isoformat(),protocol=config,status='waiting'))
    wait_for_pipeline(run,args.wait_for_pipeline_pid)
    for filename, digest in config['sources_sha256'].items():
        if sha256(ROOT/'experiments'/filename) != digest:
            raise RuntimeError('A declared dependency changed before stream execution')
    if sha256(__file__) != config['script_sha256']:
        raise RuntimeError('Declared stream source changed')
    if sha256(ROOT/'core/victim_model.py') != config['victim_source_sha256'] or sha256(ROOT/'yolov8n.pt') != config['weights_sha256']:
        raise RuntimeError('Declared victim implementation or weights changed')
    import contextlib
    import io
    import numpy as np
    import psutil
    import torch
    from extended_evaluation import COCO, decode, postprocess, official_predictions, coco_metrics
    from extended_validation import find_locations, insert
    from local_q1_validation import VictimModel, load_image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    victim = VictimModel(device='cpu')
    process = psutil.Process()
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    if sha256(data/'manifest.json') != config['dataset_sha256'] or sha256(data/'instances_subset.json') != config['annotations_sha256']:
        raise RuntimeError('Declared test subset or labels changed')
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'],key=lambda item:item['id'])[:32]
    ids, categories = [info['id'] for info in infos], sorted(coco.getCatIds())
    dataset_manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    recorded_hashes = {item['file_name']:item['sha256'] for item in dataset_manifest['images']}
    for info in infos:
        if sha256(data/'images'/info['file_name']) != recorded_hashes[info['file_name']]:
            raise RuntimeError('Held-out image changed: '+info['file_name'])
    status = json.loads(manifest.read_text(encoding='utf-8'))
    status.update(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  image_ids=ids,protocol_sha256=protocol_digest,
                  actual_runtime=dict(torch_version=torch.__version__,numpy_version=np.__version__,
                                      cpu_logical_processors=psutil.cpu_count()))
    save_json(manifest,status)
    clean_images = {info['id']:load_image(data/'images'/info['file_name']) for info in infos}
    blocks = [(seed,period,name,policy) for seed in range(10) for period in [100,300]
              for name in config['conditions'] for policy in POLICIES]
    random.Random(config['block_order_seed']).shuffle(blocks)
    results = []
    for seed,period,name,policy in blocks:
        destination = output/f'{name}_seed{seed}_period{period}_{policy}.json'
        if destination.exists():
            result = json.loads(destination.read_text(encoding='utf-8'))
            if result['protocol_sha256'] != protocol_digest:
                raise RuntimeError('Saved stream belongs to another protocol')
            results.append(result)
            continue
        patch_path = checkpoint_path(run,name,seed)
        images = clean_images
        if name != 'clean':
            patch = torch.from_numpy(np.load(patch_path))
            images = {info['id']:insert(clean_images[info['id']],patch[None],
                find_locations([clean_images[info['id']]],64,'saliency',seed+info['id'])) for info in infos}
        ordered = infos.copy()
        random.Random(53017+seed).shuffle(ordered)
        def callback(info):
            with torch.no_grad():
                started = time.perf_counter_ns()
                raw = victim.model(images[info['id']])
                forwarded = time.perf_counter_ns()
                raw = raw[0] if isinstance(raw,(tuple,list)) else raw
                decoded = decode(victim,raw[0])
                decoded_at = time.perf_counter_ns()
                stats,found = postprocess(victim,decoded)
                return dict(image_id=info['id'],forward_ms=(forwarded-started)/1e6,
                            decode_ms=(decoded_at-forwarded)/1e6,**stats,
                            predictions=official_predictions(found,info,categories))
        for info in ordered[:3]:
            callback(info)
        cpu_before, wall_start = process.cpu_times(), time.perf_counter()
        result = execute_stream(ordered,callback,period,policy)
        wall_elapsed = time.perf_counter()-wall_start
        cpu_after = process.cpu_times()
        predictions, timely_predictions, processed_ids = [], [], []
        for row in result['records']:
            row['image_id'] = ordered[row['frame_index']]['id']
            if row['status'] == 'processed':
                prediction = row['result'].pop('predictions')
                predictions.extend(prediction)
                processed_ids.append(row['image_id'])
                if not row['late']:
                    timely_predictions.extend(prediction)
        result.update(seed=seed,condition=name,image_ids=ids,processed_image_ids=processed_ids,
                      protocol_sha256=protocol_digest,weights_sha256=sha256(ROOT/'yolov8n.pt'),
                      patch_sha256=None if patch_path is None else sha256(patch_path),
                      cpu_percent_one_core_scale=100*((cpu_after.user-cpu_before.user)+(cpu_after.system-cpu_before.system))/wall_elapsed,
                      rss_mib_at_block_end=process.memory_info().rss/1024**2,
                      full_offered_stream_coco_bbox=coco_metrics(coco,predictions,ids),
                      deadline_delivered_coco_bbox=coco_metrics(coco,timely_predictions,ids),
                      predictions=predictions,on_time_predictions=timely_predictions,
                      temperature_available=False)
        save_json(destination,result)
        results.append(result)
        print(f'ACTUAL STREAM {len(results)}/{len(blocks)}: seed={seed} {name} {period}ms {policy}',flush=True)
    from extended_analysis import paired
    comparisons = []
    indexed = {(r['seed'],r['period_ms'],r['condition'],r['policy']):r for r in results}
    for period in [100,300]:
        for name in config['conditions']:
            for policy in ['drop_if_busy','adaptive_throttle']:
                pairs = [(indexed[(seed,period,name,policy)],indexed[(seed,period,name,'fifo')]) for seed in range(10)]
                for field in ['missing_or_late_fraction','on_time_fps','processed_fps']:
                    comparisons.append(dict(period_ms=period,condition=name,policy=policy,metric=field,
                        difference_policy_minus_fifo=paired([a[field]-b[field] for a,b in pairs]),
                        exploratory_p_not_multiplicity_adjusted=True))
    # Recovery cannot be claimed by discarding most of the offered input.
    for result in results:
        baseline = indexed[(result['seed'],result['period_ms'],'clean','fifo')]
        result['predeclared_service_deterioration'] = (
            result['on_time_throughput_relative_to_offered'] < .8 or result['missing_or_late_fraction'] > .1)
        result['predeclared_recovery_gate_met'] = (not result['predeclared_service_deterioration'] and
            result['deadline_delivered_coco_bbox']['AP'] >= baseline['full_offered_stream_coco_bbox']['AP']-.02)
    save_json(output/'summary.json',dict(streams=[{k:v for k,v in r.items() if k not in
              ('records','predictions','on_time_predictions')} for r in results],paired_comparisons=comparisons,
              actual_camera_measurement=False,not_a_queue_simulation=True,
              complete_streams=len(results),protocol_sha256=protocol_digest))
    final = json.loads(manifest.read_text(encoding='utf-8'))
    final.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),completed_streams=len(results))
    save_json(manifest,final)
    print('ACTUAL SOFTWARE STREAM STUDY COMPLETE; manuscript integration still required',flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if ACTIVE_MANIFEST is not None and ACTIVE_MANIFEST.exists():
            from local_q1_validation import save_json
            state = json.loads(ACTIVE_MANIFEST.read_text(encoding='utf-8'))
            state.update(status='failed',failed_utc=datetime.now(timezone.utc).isoformat(),
                         error_type=type(error).__name__,error_message=str(error))
            save_json(ACTIVE_MANIFEST,state)
        raise
