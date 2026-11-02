"""Read-only audit of completed controls; partial is not experimental completion.

Checks the frozen protocol, patch/checkpoint integrity, query history and saved
evaluation arithmetic. Does not measure the model, recompute AP or infer DoS.
Deterministic patterns are not called ten independently optimized patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_extended_evaluation import ROOT, read, audit_case

CONTROLS = ('uniform_noise', 'gaussian_noise', 'checkerboard', 'binary_modules',
            'solid_gray', 'integrity_gradient')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_integrity(record, seed, patch, folder, check):
    import torch
    check(record['seed'] == seed and record['queries'] == 400, 'integrity seed/budget mismatch')
    check(record['backward_calls'] == 100 and record['backward_image_examples'] == 400, 'backward count mismatch')
    history = record['history']
    check([h['step'] for h in history] == list(range(1, 101)), 'gradient step history mismatch')
    check([h['image_forwards'] for h in history] == list(range(4, 401, 4)), 'actual image-forward history mismatch')
    check(all(math.isfinite(h[k]) for h in history for k in ('smooth_confidence_sum', 'sponge_fitness')), 'nonfinite integrity objective')
    check(record['final_iterate_not_stochastic_best'] is True and record['not_exact_published_reproduction'] is True,
          'integrity interpretation changed')
    check(len(record['locations']) == 4 and all(len(loc) == 2 and all(isinstance(v, int) and 0 <= v <= 256 for v in loc)
          for loc in record['locations']), 'integrity optimization locations mismatch')
    # Only our own local checkpoint, validated by frozen source provenance.
    state = torch.load(folder/'resume.pt', weights_only=False, map_location='cpu')
    check(state['seed'] == seed and state['step'] == 100 and state['history'] == history, 'integrity checkpoint history mismatch')
    check(np.array_equal(state['evaluated'].numpy(), patch), 'saved patch is not last measured candidate')


def audit(run):
    output = run/'controls'
    manifest, _ = read(output/'manifest.json')
    config = manifest['protocol']
    dataset, dataset_sha = read(ROOT/'outputs/datasets/coco_val2017_extended/manifest.json')
    subset, subset_sha = read(ROOT/'outputs/datasets/coco_val2017_extended/instances_subset.json')
    infos = sorted(subset['images'], key=lambda i:i['id'])
    categories = {c['id'] for c in subset['categories']}
    errors, total_errors = [], 0
    def check(condition, message):
        nonlocal total_errors
        if not condition:
            total_errors += 1
            if len(errors) < 50:
                errors.append(message)
    check(config['script_sha256'] == digest(ROOT/'experiments/extended_controls.py'), 'controls source changed')
    check(config['evaluator_sha256'] == digest(ROOT/'experiments/extended_evaluation.py'), 'evaluator source changed')
    check(config['optimizer_manifest_sha256'] == digest(run/'manifest.json'), 'optimizer manifest changed')
    check(config['data_manifest_sha256'] == dataset_sha and dataset['subset_sha256'] == subset_sha, 'test dataset/labels changed')
    check(config['test_ids'] == [i['id'] for i in infos] and len(infos) == 128, 'test image selection mismatch')
    check(config['controls'] == list(CONTROLS) and config['seeds'] == 10, 'control inventory changed')
    check(config['size'] == 64 and config['location'] == 'saliency', 'control geometry changed')
    check(config['integrity_image_forwards'] == 400 and config['integrity_backward_calls'] == 100
          and config['noise_optimization_queries'] == 0, 'declared query accounting changed')
    expected = {f'{name}_seed{seed}':(name, seed) for seed in range(10) for name in CONTROLS}
    present = {p.parent.name:p for p in output.glob('*/evaluation.json')}
    check(set(present) <= set(expected), 'unexpected completed control')
    missing = sorted(set(expected)-set(present))
    total_rows, total_predictions, gradient_cases = 0, 0, 0
    hashes, patch_hashes = {}, {name:set() for name in CONTROLS}
    for key, path in sorted(present.items()):
        if key not in expected:
            continue
        name, seed = expected[key]
        def case_check(condition, message):
            check(condition, key+': '+message)
        result, result_sha = read(path)
        hashes[str(path.relative_to(ROOT))] = result_sha
        patch_path = path.parent/'patch.npy'
        patch_sha = digest(patch_path)
        patch_hashes[name].add(patch_sha)
        patch = np.load(patch_path, allow_pickle=False)
        case_check(patch.shape == (3, 64, 64) and patch.dtype == np.float32
                   and np.isfinite(patch).all() and patch.min() >= 0 and patch.max() <= 1, 'invalid patch tensor')
        case_check(Path(result['case']['patch']).resolve() == patch_path.resolve()
                   and result['patch_sha256'] == patch_sha, 'patch path/hash mismatch')
        budget = 400 if name == 'integrity_gradient' else 0
        expected_case = (name, seed, budget, dict(location='saliency', budgets=[budget]))
        rows, predictions = audit_case(result, expected_case, infos, categories, case_check, patch_size=64)
        total_rows += rows
        total_predictions += predictions
        if name == 'integrity_gradient':
            record, record_sha = read(path.parent/'optimization.json')
            hashes[str((path.parent/'optimization.json').relative_to(ROOT))] = record_sha
            audit_integrity(record, seed, patch, path.parent, case_check)
            gradient_cases += 1
    for name in ('checkerboard', 'solid_gray'):
        check(len(patch_hashes[name]) <= 1, name+': deterministic texture differs across seeds')
    finished = 'finished_utc' in manifest
    check(not finished or (not missing and manifest['evaluated_cases'] == 60), 'completion declaration contradicts coverage')
    return dict(status='failed' if total_errors else 'complete_saved_controls_verified' if finished and not missing else 'partial_saved_controls_verified',
                expected_cases=60, checked_cases=len(present), missing_cases=missing,
                checked_image_configuration_records=total_rows, checked_predictions=total_predictions,
                checked_integrity_optimizations=gradient_cases, checked_integrity_image_forwards=gradient_cases*400,
                unique_patch_files_by_method={name:len(values) for name, values in patch_hashes.items()},
                deterministic_patterns_not_independent_optimization_replicates=['checkerboard', 'solid_gray'],
                source_hash_inventory_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                AP_independently_recomputed=False, model_predictions_independently_recomputed=False,
                evidence_scope='saved control tensors, query histories and evaluation arithmetic; not attack efficacy or camera evidence',
                error_count=total_errors, first_errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    import os
    if os.name == 'nt':
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    result = audit(ROOT/args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['error_count'] else 2 if args.require_complete and result['status'] != 'complete_saved_controls_verified' else 0)


if __name__ == '__main__':
    main()
