"""Read-only audit of real saved optimizer artifacts; never invents results.

Partial runs are reported as partial. Passing this audit validates saved
queries, shapes, provenance and checkpoint consistency, NOT attack efficacy,
camera experiments, evaluator correctness or statistical significance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    protocol = manifest['protocol']
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    check(digest(ROOT/'experiments/extended_validation.py') == manifest['script_sha256'], 'Optimizer source hash changed')
    check(digest(ROOT/'yolov8n.pt') == manifest['weights_sha256'], 'Victim weights hash changed')
    for relative, expected in manifest['sources_sha256'].items():
        check(digest(ROOT/relative) == expected, 'Dependency hash changed: '+relative)
    training_ids = set()
    for image in protocol['training_images']:
        path = ROOT/image['path']
        check(digest(path) == image['sha256'], 'Training image hash changed: '+str(path))
        training_ids.add(int(path.stem))
    data_manifest = json.loads((ROOT/'outputs/datasets/coco_val2017_extended/manifest.json').read_text(encoding='utf-8'))
    data_root = ROOT/'outputs/datasets/coco_val2017_extended'
    original_hash_matches = digest(data_root/'instances_val2017.json') == data_manifest['annotation_sha256']
    subset_hash_matches = digest(data_root/'instances_subset.json') == data_manifest['subset_sha256']
    check(original_hash_matches, 'Full annotation source hash changed')
    check(subset_hash_matches, 'Held-out annotation subset hash changed')
    check(digest(ROOT/'experiments/prepare_extended_validation.py') == data_manifest['script_sha256'], 'Dataset preparation source hash changed')
    original = json.loads((data_root/'instances_val2017.json').read_text(encoding='utf-8'))
    subset = json.loads((data_root/'instances_subset.json').read_text(encoding='utf-8'))
    selected = sorted(original['images'],key=lambda image:image['id'])[:data_manifest['images_count']]
    check(subset['images'] == selected, 'Held-out images do not match declared deterministic selection')
    expected_ids = {image['id'] for image in selected}
    check(subset['annotations'] == [entry for entry in original['annotations'] if entry['image_id'] in expected_ids],
          'Annotations or crowd/ignore metadata differ from original source')
    check(subset['categories'] == original['categories'] and len(subset['categories']) == 80, 'Original 80-class category metadata changed')
    check(len({entry['id'] for entry in subset['annotations']}) == len(subset['annotations']), 'Duplicate annotation IDs')
    check(len(data_manifest['images']) == data_manifest['images_count'] == len(subset['images']), 'Image inventory count mismatch')
    image_integrity = True
    for image in data_manifest['images']:
        path = data_root/'images'/image['file_name']
        hash_matches = digest(path) == image['sha256']
        check(hash_matches, 'Held-out image hash changed: '+image['file_name'])
        with Image.open(path) as decoded:
            dimensions_match = decoded.size == (image['width'],image['height'])
            check(dimensions_match, 'Held-out decoded image size mismatch: '+image['file_name'])
            decoded.verify()
        image_integrity = image_integrity and hash_matches and dimensions_match
    test_ids = {image['image_id'] for image in data_manifest['images']}
    check(test_ids == expected_ids, 'Held-out manifest image IDs differ from annotation selection')
    coco128_ids = {int(path.stem) for path in (ROOT/'outputs/datasets/coco128/images/train2017').glob('*.jpg')}
    check(not coco128_ids & test_ids, 'Any COCO128/test image IDs overlap')
    check(not training_ids & test_ids, 'Optimizer/test image IDs overlap')
    check(len(protocol['training_images']) == 4, 'Not exactly four optimization images')
    expected_cases = {f'{spec["name"]}_seed{seed}':(spec,seed)
                      for spec in protocol['specifications'] for seed in range(protocol['seeds'])}
    files = {path.parent.name:path for path in (run/'optimization').glob('*/complete.json')}
    check(set(files) <= set(expected_cases), 'Unexpected completed case names')
    checked_checkpoints, completed_queries = 0, 0
    for name,path in sorted(files.items()):
        if name not in expected_cases:
            continue
        spec,seed = expected_cases[name]
        record = json.loads(path.read_text(encoding='utf-8'))
        check(record['spec'] == spec and record['seed'] == seed, name+': saved spec/seed mismatch')
        cost = 4*spec['draws']
        maximum = max(spec['budgets'])
        history = record['history']
        check(record['queries'] == maximum, name+': incomplete final query budget')
        check([entry['queries'] for entry in history] == list(range(cost,maximum+1,cost)), name+': query history not exact')
        check(len(record['checkpoints']) == len(spec['budgets']), name+': wrong checkpoint count')
        check([entry['queries'] for entry in record['checkpoints']] == spec['budgets'], name+': checkpoint budgets differ')
        for entry in history:
            check(math.isfinite(entry['fitness']) and math.isfinite(entry['selected_fitness']), name+': nonfinite history')
        if spec['method'] != 'gradient_tanh':
            check(all(a['selected_fitness'] <= b['selected_fitness'] for a,b in zip(history,history[1:])), name+': selected fitness decreased')
            check(all(abs(entry['selected_fitness']-max(previous['fitness'] for previous in history[:index+1])) < 1e-6
                      for index,entry in enumerate(history)), name+': selected fitness not observed training maximum')
        for checkpoint in record['checkpoints']:
            budget = checkpoint['queries']
            folder = path.parent
            saved = json.loads((folder/f'checkpoint_q{budget}.json').read_text(encoding='utf-8'))
            check(saved == checkpoint, name+': duplicate checkpoint metadata differs')
            patch = np.load(folder/f'patch_q{budget}.npy',allow_pickle=False)
            check(patch.shape == (3,spec['size'],spec['size']), name+': patch shape mismatch')
            check(patch.dtype == np.float32, name+': patch dtype mismatch')
            check(bool(np.isfinite(patch).all() and (patch >= 0).all() and (patch <= 1).all()), name+': patch outside finite [0,1]')
            check(abs(checkpoint['area_percent']-100*spec['size']**2/320**2) < 1e-9, name+': wrong area')
            check(len(checkpoint['locations']) == 4 and all(0 <= y <= 320-spec['size'] and 0 <= x <= 320-spec['size']
                  for y,x in checkpoint['locations']), name+': invalid placement')
            position = budget//cost-1
            if position < len(history):
                check(abs(checkpoint['best_training_fitness']-history[position]['selected_fitness']) < 1e-6,
                      name+': checkpoint fitness does not match evaluated history')
            checked_checkpoints += 1
        completed_queries += record['queries']
    missing = sorted(set(expected_cases)-set(files))
    declared_finished = 'optimization_finished_utc' in manifest
    check(not declared_finished or not missing, 'Manifest says finished but cases are missing')
    if declared_finished:
        check(manifest.get('optimizer_runs') == len(expected_cases), 'Incorrect final run count')
    return dict(status='failed' if errors else 'complete_artifacts_verified' if not missing and declared_finished else 'partial_artifacts_verified',
                optimizer_declared_finished=declared_finished,expected_cases=len(expected_cases),
                completed_cases=len(files),missing_cases=missing,verified_checkpoints=checked_checkpoints,
                completed_image_forwards=completed_queries,training_test_ids_disjoint=not training_ids & test_ids,
                dataset_checks=dict(heldout_images=len(test_ids),original_annotations=len(subset['annotations']),
                    categories=len(subset['categories']),all_image_hashes_and_decoding_checked=image_integrity,
                    metadata_matches_original_annotation_source=bool(original_hash_matches and subset_hash_matches
                        and subset['images'] == selected and subset['categories'] == original['categories']
                        and subset['annotations'] == [entry for entry in original['annotations'] if entry['image_id'] in expected_ids]),
                    entire_coco128_test_ids_disjoint=not coco128_ids & test_ids,
                    upstream_checkpoint_training_or_selection_overlap=data_manifest['upstream_model_selection_overlap']),
                evidence_scope='saved optimizer artifacts and dataset provenance only; no attack efficacy or camera claims',errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    parser.add_argument('--require-complete',action='store_true')
    args = parser.parse_args()
    result = audit(ROOT/args.run_dir)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['errors']:
        raise SystemExit(1)
    if args.require_complete and result['status'] != 'complete_artifacts_verified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
