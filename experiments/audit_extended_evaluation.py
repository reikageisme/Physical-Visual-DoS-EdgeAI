"""Read-only coverage/arithmetic audit of saved extended evaluation records.

No model inference, optimization, patch selection or source changes. This
checks file provenance, exact image/configuration coverage, timing medians,
count limits, summary arithmetic, COCO prediction bounds and EOT pairing.
It does NOT independently recompute AP or certify efficacy/camera evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('active_candidates', 'nms_input_candidates', 'final_detections',
          'forward_ms', 'nms_ms', 'postprocess_ms')
AP_KEYS = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
           'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def read(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def close(a, b):
    return (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6))


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low]+(position-low)*(ordered[high]-ordered[low])


def summary(values):
    return dict(n=len(values), mean=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else None,
                median=statistics.median(values), q1=quantile(values, .25), q3=quantile(values, .75))


def audit_case(record, expected, infos, categories, check, patch_size=None):
    """All checks use explicit expected IDs/configurations, not file self-report."""
    ids = [i['id'] for i in infos]
    info_by_id = {i['id']:i for i in infos}
    case = record['case']
    name, seed, budget, spec = expected
    check(case['name'] == name and case['seed'] == seed and case['budget'] == budget
          and case['spec'] == spec, 'case metadata mismatch')
    check(record['image_ids'] == ids and record['images'] == len(ids), 'image selection mismatch')
    configs = [(tau, cap) for tau in [.01, .05, .10, .25, .50] for cap in [None, 50, 100, 300]] \
        if name == 'clean' or (name in ('sg_ga', 'adaptive_cap100') and budget in (400, 2500)) \
        else [(.01, None), (.25, None), (.01, 100)]
    expected_keys = {(i, tau, cap) for i in ids for tau, cap in configs}
    keys = Counter((r['image_id'], r['threshold'], r['pre_nms_cap']) for r in record['records'])
    check(set(keys) == expected_keys and all(n == 1 for n in keys.values()), 'record coverage/uniqueness mismatch')
    metrics = {(m['threshold'], m['pre_nms_cap']):m for m in record['metrics']}
    check(set(metrics) == set(configs) and len(record['metrics']) == len(configs), 'metric configuration mismatch')
    prediction_keys = {str(config) for config in configs}
    check(set(record['predictions']) == prediction_keys, 'prediction configuration mismatch')
    for row in record['records']:
        values = [row[f] for f in FIELDS]+[row['filter_ms']]
        check(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values), 'nonfinite/negative record')
        count, entered, final = (row[k] for k in FIELDS[:3])
        check(all(isinstance(v, (int, float)) and v == int(v) for v in (count, entered, final)), 'noninteger count')
        cap = row['pre_nms_cap']
        check(count <= 2100 and entered == (count if cap is None else min(cap, count))
              and final <= min(300, entered), 'candidate/cap/final count limit mismatch')
        forwards = row['forward_ms_samples']
        samples = row['postprocess_samples']
        check(len(forwards) == 3 and len(samples) == 3, 'timing repeat count mismatch')
        check(all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in forwards), 'invalid forward sample')
        if len(forwards) == 3:
            check(close(row['forward_ms'], statistics.median(forwards)), 'forward median mismatch')
        if len(samples) == 3:
            for field in ('active_candidates', 'nms_input_candidates', 'final_detections',
                          'filter_ms', 'nms_ms', 'postprocess_ms'):
                check(close(row[field], statistics.median(s[field] for s in samples)), 'postprocess median mismatch: '+field)
            for sample in samples:
                check(close(sample['filter_ms']+sample['nms_ms'], sample['postprocess_ms']), 'sample timing boundary mismatch')
        if name == 'clean':
            check(row['location'] is None, 'clean image has patch location')
        else:
            size = spec['size'] if patch_size is None else patch_size
            check(isinstance(row['location'], list) and len(row['location']) == 2
                  and all(isinstance(v, int) and 0 <= v <= 320-size for v in row['location']), 'invalid patch placement')
    for tau, cap in configs:
        if (tau, cap) not in metrics:
            continue
        rows = [r for r in record['records'] if r['threshold'] == tau and r['pre_nms_cap'] == cap]
        if not rows:
            continue
        metric = metrics[(tau, cap)]
        for field in FIELDS:
            calculated = summary([r[field] for r in rows])
            saved = metric[field]
            check(set(saved) == set(calculated), 'summary fields mismatch: '+field)
            for key, value in calculated.items():
                check(saved.get(key) is None if value is None else close(saved.get(key), value), 'summary arithmetic mismatch: '+field+'/'+key)
        ap = metric['coco_subset_bbox']
        check(set(ap) == set(AP_KEYS), 'COCO metric fields mismatch')
        check(all(isinstance(v, (int, float)) and math.isfinite(v) and (v == -1 or 0 <= v <= 1) for v in ap.values()), 'COCO metric out of range')
        predictions = record['predictions'].get(str((tau, cap)), [])
        prediction_counts = Counter(p['image_id'] for p in predictions)
        check(all(count <= 300 for count in prediction_counts.values()), 'more than 300 predictions per image')
        for prediction in predictions:
            image_id = prediction['image_id']
            check(image_id in info_by_id and prediction['category_id'] in categories, 'prediction image/category mismatch')
            check(isinstance(prediction['score'], (int, float)) and math.isfinite(prediction['score'])
                  and tau < prediction['score'] <= 1, 'prediction confidence outside filter threshold')
            box = prediction['bbox']
            check(len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in box), 'invalid COCO box')
            if image_id in info_by_id and len(box) == 4:
                image = info_by_id[image_id]
                check(box[0]+box[2] <= image['width']+1e-5 and box[1]+box[3] <= image['height']+1e-5, 'COCO box exceeds image extent')
        for row in rows:
            check(prediction_counts[row['image_id']] == row['final_detections'], 'final detection/prediction count mismatch')
    check(record['total_latency_not_sum_of_medians'] is True, 'separate timing interpretation changed')
    check(record['evaluator'] == 'pycocotools.COCOeval bbox, maxDets 1/10/100, original crowd/ignore', 'evaluator metadata changed')
    return len(record['records']), sum(len(p) for p in record['predictions'].values())


def audit_eot(record, expected, infos, check):
    name, seed, budget, _ = expected
    check(record['case'] == name and record['seed'] == seed and record['queries'] == budget, 'EOT case mismatch')
    check(record['paired_transform_parameters'] is True and record['images'] == 8
          and record['independent_draws'] == 16, 'EOT protocol mismatch')
    check([r['image_id'] for r in record['records']] == [i['id'] for i in infos[:8]], 'EOT image selection mismatch')
    for row in record['records']:
        check([s['draw'] for s in row['samples']] == list(range(16)), 'EOT draw coverage mismatch')
        for sample in row['samples']:
            check(close(sample['difference'], sample['patch_fitness']-sample['clean_fitness']), 'EOT difference mismatch')
            check(all(math.isfinite(sample[k]) for k in ('difference', 'patch_fitness', 'clean_fitness')), 'nonfinite EOT sample')
    return sum(len(r['samples']) for r in record['records'])


def classify_files(files, expected):
    # Method names such as eot_no_blur ALSO start with eot_. Classify by
    # canonical expected names, never by prefix alone (paired file is eot_eot_*).
    expected_eot = {'eot_'+name for name, context in expected.items()
                    if context[2] and (context[2] == max(context[3]['budgets'])
                                       or (context[0] == 'sg_ga' and context[2] == 400))}
    cases = {name:path for name, path in files.items() if name in expected}
    paired = {name:path for name, path in files.items() if name in expected_eot}
    unexpected = sorted(set(files)-set(expected)-expected_eot)
    return cases, paired, expected_eot, unexpected


def audit(run):
    optimizer, optimizer_sha = read(run/'manifest.json')
    manifest, _ = read(run/'evaluation/manifest.json')
    protocol = manifest['protocol']
    subset, subset_sha = read(ROOT/'outputs/datasets/coco_val2017_extended/instances_subset.json')
    dataset, dataset_sha = read(ROOT/'outputs/datasets/coco_val2017_extended/manifest.json')
    infos = sorted(subset['images'], key=lambda i:i['id'])
    categories = {c['id'] for c in subset['categories']}
    errors, error_count = [], 0
    def check(condition, message):
        nonlocal error_count
        if not condition:
            error_count += 1
            if len(errors) < 50:
                errors.append(message)
    check(protocol['script_sha256'] == hashlib.sha256((ROOT/'experiments/extended_evaluation.py').read_bytes()).hexdigest(), 'evaluator source changed')
    check(protocol['optimizer_manifest_sha256'] == optimizer_sha, 'optimizer manifest changed')
    check(protocol['dataset_sha256'] == dataset_sha, 'dataset manifest changed')
    check(dataset['subset_sha256'] == subset_sha, 'annotation subset changed')
    check(protocol['pilot'] is False and protocol['test_images'] == len(infos) == 128, 'not full declared subset evaluation')
    check(optimizer['protocol']['seeds'] == 10, 'not ten optimizer seeds')
    expected = {}
    for seed in range(10):
        expected[f'clean_seed{seed}_q0'] = ('clean', seed, 0, None)
        for spec in optimizer['protocol']['specifications']:
            for budget in spec['budgets']:
                expected[f'{spec["name"]}_seed{seed}_q{budget}'] = (spec['name'], seed, budget, spec)
    inventory = {p.stem:p for p in (run/'evaluation').glob('*.json') if p.stem != 'manifest'}
    actual, present_eot, expected_eot, unexpected = classify_files(inventory, expected)
    check(not unexpected, 'unexpected evaluation case file: '+','.join(unexpected))
    missing = sorted(set(expected)-set(actual))
    source_hashes, total_rows, total_predictions, eot_samples, eot_cases = {}, 0, 0, 0, 0
    for name, context in expected.items():
        if name not in actual:
            continue
        path = actual[name]
        record, file_sha = read(path)
        source_hashes[name] = file_sha
        def case_check(condition, message):
            check(condition, name+': '+message)
        if context[0] == 'clean':
            case_check(record['patch_sha256'] is None and record['case']['patch'] is None, 'unexpected clean patch')
        else:
            patch_path = run/'optimization'/f'{context[0]}_seed{context[1]}'/f'patch_q{context[2]}.npy'
            case_check(Path(record['case']['patch']).resolve() == patch_path.resolve(), 'wrong patch path')
            case_check(record['patch_sha256'] == hashlib.sha256(patch_path.read_bytes()).hexdigest(), 'patch hash changed')
        selected = infos[:32] if context[2] == 1000 else infos
        rows, predictions = audit_case(record, context, selected, categories, case_check)
        total_rows += rows
        total_predictions += predictions
        eot_path = path.with_name('eot_'+path.name)
        if 'eot_'+name in expected_eot and eot_path.exists():
            eot, eot_sha = read(eot_path)
            source_hashes['eot_'+name] = eot_sha
            eot_samples += audit_eot(eot, context, infos, case_check)
            eot_cases += 1
    missing_eot = sorted(expected_eot-set(present_eot))
    finished = 'finished_utc' in manifest
    check(not finished or (not missing and not missing_eot and manifest['evaluated_cases'] == len(expected)), 'finished declaration contradicts coverage')
    return dict(status='failed' if error_count else 'complete_saved_evaluation_verified' if finished and not missing and not missing_eot else 'partial',
                expected_cases=len(expected), completed_cases=len(actual), missing_cases=missing,
                checked_image_configuration_records=total_rows, checked_predictions=total_predictions,
                expected_eot_cases=len(expected_eot), completed_eot_cases=eot_cases,
                missing_eot_cases=missing_eot, checked_paired_eot_samples=eot_samples,
                source_hash_inventory_sha256=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                AP_independently_recomputed=False, model_predictions_independently_recomputed=False,
                evidence_scope='saved record coverage, provenance and arithmetic; not efficacy or camera validation',
                error_count=error_count, first_errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    # Keep this read-only JSON audit below the live research worker's priority.
    import os
    if os.name == 'nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    result = audit(ROOT/args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if result['error_count'] else 0 if result['status']=='complete_saved_evaluation_verified' else 2)


if __name__ == '__main__':
    main()
