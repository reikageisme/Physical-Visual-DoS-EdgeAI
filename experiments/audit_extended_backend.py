"""Read-only audit of saved NMS-backend and prepared-tensor timing measurements.

Checks exact block/sample coverage, checkpoint/cached-input provenance, timer
boundaries and budget flags. Independently recalculates OFFLINE queue arithmetic,
not model inference. Passing does not establish DoS or camera performance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_extended_evaluation import ROOT, read, close

CONDITIONS = ('clean', 'sg_ga', 'ga_center', 'ga_random_location', 'random_search')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_timing(block, seed, threads, ids, check):
    check(block['seed'] == seed and block['cpu_threads'] == threads, 'seed/thread block mismatch')
    check(block['temperature_available'] is False and block['temperature_not_substituted_with_zero'] is True,
          'unsupported temperature interpretation')
    expected = {(name, image_id, repeat) for name in CONDITIONS for image_id in ids for repeat in range(3)}
    actual = Counter((r['condition'], r['image_id'], r['repeat']) for r in block['records'])
    check(set(actual) == expected and all(n == 1 for n in actual.values()), 'timing coverage/uniqueness mismatch')
    for row in block['records']:
        fields = ('forward_ms', 'decode_ms', 'filter_ms', 'nms_ms', 'postprocess_ms', 'total_prepared_tensor_ms',
                  'condition_block_cpu_percent_one_core_scale', 'condition_block_cpu_percent_machine_normalized',
                  'rss_mib_at_condition_end')
        check(all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0 for k in fields),
              'nonfinite/negative timing/resource value')
        check(close(row['filter_ms']+row['nms_ms'], row['postprocess_ms']), 'postprocess clock boundary mismatch')
        parts = row['forward_ms']+row['decode_ms']+row['postprocess_ms']
        check(row['total_prepared_tensor_ms']+1e-6 >= parts, 'whole-tensor duration smaller than timed components')
        check(row['late_processing_budget'] == (row['total_prepared_tensor_ms'] > 300), '300ms budget flag mismatch')
        check(close(row['condition_block_cpu_percent_one_core_scale']/12,
                    row['condition_block_cpu_percent_machine_normalized']), 'CPU normalization mismatch')
        if 'threshold' in row:
            check(close(row['threshold'], .01), 'timing threshold mismatch')
        if 'pre_nms_cap' in row:
            check(row['pre_nms_cap'] is None, 'unexpected pre-NMS limit')
        count, entered, final = [row[k] for k in ('active_candidates', 'nms_input_candidates', 'final_detections')]
        check(all(isinstance(v, (int, float)) and math.isfinite(v) and v == int(v) for v in (count, entered, final)), 'noninteger candidate count')
        check(0 <= count <= 2100 and entered == count and 0 <= final <= min(300, count), 'candidate/final count limit mismatch')
        location = row['location']
        check(location is None if row['condition']=='clean' else isinstance(location, list) and len(location)==2
              and all(isinstance(v, int) and 0 <= v <= 256 for v in location), 'timing patch placement mismatch')
    return len(block['records'])


def audit_queue(record, service, period, policy, check):
    """Independent scalar recurrence, including the per-period lateness rule."""
    check(record['policy'] == policy and record['arrival_period_ms'] == period, 'queue configuration mismatch')
    check(record['actual_camera_measurement'] is False and record['queue_model_assumes_deterministic_arrivals'] is True,
          'offline queue scope changed')
    clock, dropped, expected = 0., 0, []
    for index, duration in enumerate(service):
        arrival = period*index
        if policy == 'drop_if_busy' and arrival < clock:
            dropped += 1
            continue
        start = max(arrival, clock)
        clock = start+duration
        expected.append(dict(frame=index, arrival_ms=arrival, start_ms=start, finish_ms=clock,
                             modeled_age_ms=clock-arrival, modeled_wait_ms=start-arrival,
                             late=clock-arrival > period))
    check(record['generated'] == len(service) and record['processed'] == len(expected)
          and record['dropped'] == dropped, 'queue frame conservation mismatch')
    check(len(record['records']) == len(expected), 'queue row count mismatch')
    for actual, reference in zip(record['records'], expected):
        for key, value in reference.items():
            check(actual[key] == value if isinstance(value, (bool, int)) else close(actual[key], value),
                  'queue recurrence mismatch: '+key)
    check(close(record['modeled_late_fraction'], sum(r['late'] for r in expected)/max(1, len(expected))), 'modeled late fraction mismatch')
    check(close(record['max_modeled_age_ms'], max((r['modeled_age_ms'] for r in expected), default=0)), 'modeled age maximum mismatch')
    check(close(record['max_modeled_wait_ms'], max((r['modeled_wait_ms'] for r in expected), default=0)), 'modeled wait maximum mismatch')


def audit(run):
    import numpy as np
    output = run/'backend_qos'
    manifest, _ = read(output/'manifest.json')
    optimizer, optimizer_sha = read(run/'manifest.json')
    config = manifest['protocol']
    errors, total_errors = [], 0
    def check(condition, message):
        nonlocal total_errors
        if not condition:
            total_errors += 1
            if len(errors) < 50:
                errors.append(message)
    check(config['script_sha256'] == digest(ROOT/'experiments/extended_backend_qos.py'), 'backend measurement source changed')
    check(config['optimizer_manifest_sha256'] == optimizer_sha, 'optimizer manifest changed')
    check(config['evaluator_sha256'] == digest(ROOT/'experiments/extended_evaluation.py'), 'evaluator source changed')
    check(config['dataset_sha256'] == digest(ROOT/'outputs/datasets/coco_val2017_extended/manifest.json'), 'dataset manifest changed')
    check(optimizer['weights_sha256'] == digest(ROOT/'yolov8n.pt'), 'model weights changed')
    subset, _ = read(ROOT/'outputs/datasets/coco_val2017_extended/instances_subset.json')
    ids = [i['id'] for i in sorted(subset['images'], key=lambda i:i['id'])[:16]]
    check(config['test_ids'] == ids and config['seeds'] == 10 and config['physical_devices'] == 1, 'timing cohort mismatch')
    check(config['conditions'] == list(CONDITIONS) and config['cpu_thread_settings'] == [1,2,4], 'condition/thread settings changed')
    check(config['budget'] == 2500 and config['threshold'] == .01 and config['nms_iou'] == .45
          and config['pre_nms_cap'] is None and config['processing_deadline_ms'] == 300, 'timing configuration changed')
    check(config['repeats'] == 3 and config['warmup'] == 1 and config['backend_repeats'] == 5
          and config['backend_cpu_threads'] == 2, 'measurement repeats/warmup changed')
    expected = {f'timing_seed{s}_threads{t}':(s,t) for s in range(10) for t in [1,2,4]}
    present = {p.stem:p for p in output.glob('timing_seed*.json')}
    check(set(present) <= set(expected), 'unexpected timing block file')
    hashes, count, queues = {}, 0, 0
    for name, path in sorted(present.items()):
        if name not in expected:
            continue
        seed, threads = expected[name]
        block, file_sha = read(path)
        hashes[str(path.relative_to(ROOT))] = file_sha
        def block_check(condition, message):
            check(condition, name+': '+message)
        count += audit_timing(block, seed, threads, ids, block_check)
        check(set(block['patch_sha256']) == set(CONDITIONS)-{'clean'}, name+': patch inventory mismatch')
        for method, patch_sha in block['patch_sha256'].items():
            block_check(patch_sha == digest(run/'optimization'/f'{method}_seed{seed}/patch_q2500.npy'), 'timed patch changed')
        for condition in CONDITIONS:
            queue_path = output/f'queue_seed{seed}_threads{threads}_{condition}.json'
            queued, queue_sha = read(queue_path)
            hashes[str(queue_path.relative_to(ROOT))] = queue_sha
            block_check(queued['actual_streaming_experiment'] is False, 'queue called actual stream')
            service = [r['total_prepared_tensor_ms'] for r in block['records'] if r['condition']==condition]
            queue_configs = {(period, policy) for period in [100,200,300] for policy in ['fifo','drop_if_busy']}
            keys = [(r['arrival_period_ms'], r['policy']) for r in queued['modeled']]
            block_check(set(keys) == queue_configs and len(keys) == 6, 'queue configuration coverage mismatch')
            for record in queued['modeled']:
                audit_queue(record, service, record['arrival_period_ms'], record['policy'], block_check)
                queues += 1
    comparison, comparison_sha = read(output/'backend.json')
    hashes[str((output/'backend.json').relative_to(ROOT))] = comparison_sha
    check(comparison['source_evaluator_manifest_sha256'] == digest(run/'evaluation/manifest.json'), 'cached evaluator manifest changed')
    expected_backend = {(s, n, i, tau, cap, engine) for s in range(10) for n in ['clean','sg_ga'] for i in ids
                        for tau, cap in [(.01,None),(.25,None),(.01,100)] for engine in ['custom','torchvision']}
    actual_backend = Counter((r['seed'],r['condition'],r['image_id'],r['threshold'],r['pre_nms_cap'],r['backend'])
                             for r in comparison['records'])
    check(set(actual_backend) == expected_backend and all(v==1 for v in actual_backend.values()), 'backend sample coverage mismatch')
    cached = {}
    for entry in comparison['input_files']:
        path = ROOT/entry['path']
        check(path.resolve().is_relative_to((run/'evaluation/raw_cache').resolve()), 'cache path outside declared evaluator')
        check(digest(path) == entry['sha256'], 'actual cached predictions changed')
        with np.load(path, allow_pickle=False) as source:
            decoded = source['decoded']
            check(decoded.shape == (2100,6) and decoded.dtype == np.float32 and np.isfinite(decoded).all(), 'invalid decoded input')
            check(((decoded[:,4]>=0)&(decoded[:,4]<=1)).all()
                  and ((decoded[:,5]>=0)&(decoded[:,5]<=79)&(decoded[:,5]==np.floor(decoded[:,5]))).all(), 'invalid cached scores/classes')
            cached[(path.parent.name, int(path.stem))] = decoded
    expected_cache = {(f'{n}_seed{s}_q{0 if n=="clean" else 2500}',i) for s in range(10) for n in ['clean','sg_ga'] for i in ids}
    check(set(cached) == expected_cache and len(comparison['input_files']) == 320, 'cached input coverage mismatch')
    for row in comparison['records']:
        name = f'{row["condition"]}_seed{row["seed"]}_q{0 if row["condition"]=="clean" else 2500}'
        decoded = cached[(name,row['image_id'])]
        candidates = int(np.count_nonzero(decoded[:,4] > row['threshold']))
        entered = candidates if row['pre_nms_cap'] is None else min(candidates, row['pre_nms_cap'])
        samples = row['samples']
        check(len(samples) == 5, 'backend repeat count mismatch')
        check(isinstance(row['outputs_exactly_equal'], bool), 'backend equality observation missing')
        check(row['nms_input_candidates'] == entered, 'cached candidate count mismatch')
        for field in ['nms_ms','postprocess_ms']:
            check(close(row[field], statistics.median(s[field] for s in samples)), 'backend timing median mismatch')
        for sample in samples:
            check(all(isinstance(sample[k], (int,float)) and math.isfinite(sample[k]) and sample[k] >= 0
                      for k in ['filter_ms','nms_ms','postprocess_ms']), 'invalid backend timing sample')
            check(sample['active_candidates'] == candidates and sample['nms_input_candidates'] == entered
                  and 0 <= sample['final_detections'] <= min(300, entered), 'backend sample candidate mismatch')
            check(close(sample['filter_ms']+sample['nms_ms'],sample['postprocess_ms']), 'backend sample clock boundary mismatch')
    missing = sorted(set(expected)-set(present))
    finished = 'finished_utc' in manifest
    check(not finished or (not missing and manifest['completed_blocks'] == 30), 'completion contradicts block coverage')
    return dict(status='failed' if total_errors else 'complete_saved_backend_verified' if finished and not missing else 'partial',
                checked_blocks=len(present), expected_blocks=30, missing_blocks=missing,
                checked_whole_tensor_samples=count, checked_cached_prediction_inputs=len(cached),
                checked_backend_configurations=len(actual_backend), checked_offline_queues=queues,
                exact_backend_output_mismatches=sum(not r['outputs_exactly_equal'] for r in comparison['records']),
                model_predictions_independently_recomputed=False, actual_camera_measurement=False,
                offline_queue_lateness_threshold='its arrival period, not the 300ms prepared-tensor processing flag',
                source_hash_inventory_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                evidence_scope='saved timing/cached-input provenance and offline queue arithmetic; not efficacy, actual streams or camera proof',
                error_count=total_errors, first_errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    import os
    if os.name == 'nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    result = audit(ROOT/args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['error_count'] else 0 if result['status']=='complete_saved_backend_verified' else 2)


if __name__ == '__main__':
    main()
