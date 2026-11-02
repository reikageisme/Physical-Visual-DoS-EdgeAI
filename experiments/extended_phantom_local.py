"""Area/query-matched, explicitly LOCAL adaptation of Phantom Sponges.

This is an independent implementation of the published objective, not copied
vendor code or an exact reproduction. The author's YOLOv5/UAP/L2 experiment
and this YOLOv8/overwrite/EOT experiment must not be conflated. No camera,
remote target, manuscript, Git operation or upload is accessed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments'))
AUTHOR_COMMIT = '65d4fafe9aa7456d0e58f9e10dcab05ef81fe9c0'
AUTHOR_SOURCE = f'https://github.com/AvishagS422/PhantomSponges/blob/{AUTHOR_COMMIT}/attack/uap_phantom_sponge.py'
AUTHOR_SOURCE_UTF8_SHA256 = 'f28e9bcc3b204711cbd1a645ffdbb79de8fe00b8e76f616b19689061630adf48'
ACTIVE_MANIFEST = None


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Atomic, recoverable experiment output; not a source-file editing utility.
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def raw_tensor(output):
    import torch
    raw = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[1] <= 6 or raw.shape[2] == 0:
        raise ValueError('Expected YOLOv8 B x (4 + classes) x candidates decoded predictions')
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite victim predictions')
    return raw


def target_margin(raw, threshold=.25, target_class=2):
    """Author's inactive-box target margin, with YOLOv8 class scores directly.

    YOLOv8 has no separate objectness factor. Its decoded class confidence
    replaces YOLOv5's objectness * conditional-class confidence; this mapping
    is an adaptation, not an assertion that the models have identical heads.
    The denominator includes ALL batch/candidate slots, not just inactive ones.
    """
    import torch
    scores = raw[:, 4:, :]
    if not 0 <= target_class < scores.shape[1]:
        raise ValueError('Target class outside decoded classes')
    inactive = scores.max(dim=1).values < threshold
    margins = torch.clamp(threshold-scores[:, target_class, :], min=0)
    return (margins*inactive).sum()/(raw.shape[0]*raw.shape[2])


def classwise_detections(raw, threshold, maximum):
    """Differentiable selected boxes; nondifferentiable NMS indices as in author.

    Confidence/.45 IoU, best-class selection, 4096 class offsets and 30000
    pre-NMS limit match the author's local YOLOv5 utility. No wall-time cutoff:
    an overloaded image must not silently become an empty prediction here.
    """
    import torch
    from torchvision.ops import nms
    outputs = []
    for prediction in raw:
        scores, labels = prediction[4:].max(dim=0)
        keep = scores > threshold
        xywh = prediction[:4, keep].T
        boxes = torch.cat([xywh[:, :2]-xywh[:, 2:]/2,
                           xywh[:, :2]+xywh[:, 2:]/2], dim=1)
        selected_scores, selected_labels = scores[keep], labels[keep]
        if len(boxes) > 30000:
            order = selected_scores.argsort(descending=True)[:30000]
            boxes, selected_scores, selected_labels = boxes[order], selected_scores[order], selected_labels[order]
        if len(boxes):
            indices = nms(boxes+4096*selected_labels[:, None], selected_scores, .45)[:maximum]
            outputs.append(torch.cat([boxes[indices], selected_scores[indices, None],
                                      selected_labels[indices, None].to(boxes.dtype)], dim=1))
        else:
            outputs.append(prediction.new_empty((0, 6)))
    return outputs


def preservation_loss(clean, patched):
    """1 - average best same-class IoU for every clean final detection.

    Empty clean detections yield constant one, as in the author's code.
    Nonpositive union is handled as zero IoU instead of introducing NaNs.
    """
    import torch
    if clean.shape != patched.shape:
        raise ValueError('Clean/patch prediction layouts differ')
    original = classwise_detections(clean.detach(), .25, 1000)
    adversarial = classwise_detections(patched, .001, 30000)
    overlaps = []
    for originals, modified in zip(original, adversarial):
        for original_box in originals:
            boxes = modified[modified[:, 5] == original_box[5], :4]
            if not len(boxes):
                overlaps.append(patched.sum()*0.)
                continue
            intersection_wh = (torch.minimum(boxes[:, 2:], original_box[2:4])-
                               torch.maximum(boxes[:, :2], original_box[:2])).clamp(min=0)
            intersection = intersection_wh.prod(dim=1)
            area = (boxes[:, 2:]-boxes[:, :2]).clamp(min=0).prod(dim=1)
            original_area = (original_box[2:4]-original_box[:2]).clamp(min=0).prod()
            union = area+original_area-intersection
            iou = torch.where(union > 0, intersection/union.clamp(min=1e-12), torch.zeros_like(union))
            overlaps.append(iou.max())
    return 1-torch.stack(overlaps).mean() if overlaps else patched.sum()*0.+1


def phantom_loss(clean, patched):
    margin, preservation = target_margin(patched), preservation_loss(clean, patched)
    # Author constructor defaults lambda_1=.75, lambda_2=0 (area term absent).
    return .75*margin+.25*preservation, margin, preservation


def optimize(victim, images, seed, folder, protocol_sha256, budget=400):
    import numpy as np
    import torch
    from PIL import Image
    from extended_validation import find_locations, insert, make_eot
    if len(images) != 4 or budget <= 0 or budget % (2*len(images)):
        raise ValueError('Four optimization images, positive budget divisible by eight required')
    folder.mkdir(parents=True, exist_ok=True)
    completed, state_file = folder/'optimization.json', folder/'resume.pt'
    if completed.exists():
        result = load(completed)
        if result['protocol_sha256'] != protocol_sha256 or result['patch_sha256'] != digest(folder/f'patch_q{budget}.npy'):
            raise RuntimeError('Completed local reference provenance mismatch')
        return result
    torch.manual_seed(seed)
    patch = torch.rand(3, 64, 64)
    train = torch.cat(images)
    locations = find_locations(images, 64, 'saliency', seed)
    transform = make_eot('full')
    history, first_step, saved_wall = [], 0, 0.
    if state_file.exists():
        state = torch.load(state_file, weights_only=False, map_location='cpu')
        if state['protocol_sha256'] != protocol_sha256 or state['seed'] != seed or state['budget'] != budget:
            raise RuntimeError('Local reference resume protocol mismatch')
        patch, history, first_step, saved_wall = state['patch'], state['history'], state['step'], state['wall_seconds']
        evaluated = state['evaluated']
    started = time.perf_counter()
    steps = budget//8
    for step in range(first_step, steps):
        candidate = patch.detach().clone().requires_grad_(True)
        draw_seed = 270013+seed*1000+step
        torch.manual_seed(draw_seed)
        with torch.no_grad():
            clean = raw_tensor(victim.model(transform(train)))
        # Match clean/patch EOT parameters AND generated sensor noise.
        torch.manual_seed(draw_seed)
        patched = raw_tensor(victim.model(transform(insert(train, candidate[None], locations))))
        loss, margin, preservation = phantom_loss(clean, patched)
        gradient, = torch.autograd.grad(loss, candidate)
        if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
            raise RuntimeError('Nonfinite Phantom-local objective/gradient')
        evaluated = candidate.detach().clone()
        history.append(dict(step=step+1, image_forwards=(step+1)*8,
                            clean_image_forwards=(step+1)*4, patched_image_forwards=(step+1)*4,
                            backward_calls=step+1, eot_draw_seed=draw_seed,
                            loss=float(loss.detach()), target_margin=float(margin.detach()),
                            preservation_loss=float(preservation.detach()),
                            gradient_l1=float(gradient.detach().abs().sum())))
        patch = (candidate-.0005*gradient.sign()).clamp(0, 1).detach()
        temporary = state_file.with_suffix('.tmp')
        torch.save(dict(protocol_sha256=protocol_sha256, seed=seed, budget=budget,
                        patch=patch, evaluated=evaluated, step=step+1, history=history,
                        wall_seconds=saved_wall+time.perf_counter()-started), temporary)
        temporary.replace(state_file)
        if (step+1) % 10 == 0 or step+1 == steps:
            print(f'PHANTOM LOCAL seed {seed}: {(step+1)*8}/{budget} image forwards', flush=True)
    # The final measured iterate is saved, not the unevaluated next update.
    patch_path = folder/f'patch_q{budget}.npy'
    np.save(patch_path, evaluated.numpy())
    Image.fromarray((evaluated.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)).save(folder/f'patch_q{budget}.png')
    result = dict(seed=seed, protocol_sha256=protocol_sha256, queries=budget,
                  clean_image_forwards=budget//2, patched_image_forwards=budget//2,
                  backward_calls=steps, backward_image_examples=steps*4,
                  locations=locations, history=history, patch_sha256=digest(patch_path),
                  final_evaluated_iterate=True, not_exact_published_reproduction=True,
                  wall_seconds=saved_wall+time.perf_counter()-started)
    write_json(completed, result)
    return result


def wait_for_stream(run, pid):
    import psutil
    preceding = run/'stream_qos/manifest.json'
    handle = psutil.Process(pid) if pid is not None else None
    if handle is not None and 'extended_stream_qos.py' not in ' '.join(handle.cmdline()):
        raise RuntimeError('Wait handle is not the declared software stream process')
    birth = None if handle is None else handle.create_time()
    while True:
        state = load(preceding)
        if state.get('status') == 'complete' and state.get('completed_streams') == 240:
            return
        if state.get('status') == 'failed':
            raise RuntimeError('Preceding software stream failed; do not overlap inference')
        if handle is None:
            raise RuntimeError('Finish software stream or supply its live PID before local reference')
        try:
            if not handle.is_running() or handle.create_time() != birth or handle.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError('Preceding software stream stopped without completion')
        except psutil.NoSuchProcess as error:
            raise RuntimeError('Preceding software stream handle missing') from error
        time.sleep(30)


def main():
    global ACTIVE_MANIFEST
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--wait-for-stream-pid', type=int)
    args = parser.parse_args()
    run = (ROOT/args.run_dir).resolve()
    if not run.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Keep local reference output within task outputs')
    output = run/'phantom_local'
    output.mkdir(exist_ok=True)
    optimizer = load(run/'manifest.json')
    dependencies = ['extended_validation.py', 'extended_evaluation.py', 'local_q1_validation.py', 'extended_analysis.py']
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    config = dict(script_sha256=digest(__file__),
                  dependencies_sha256={f:digest(ROOT/'experiments'/f) for f in dependencies},
                  victim_source_sha256=digest(ROOT/'core/victim_model.py'), weights_sha256=digest(ROOT/'yolov8n.pt'),
                  optimizer_manifest_sha256=digest(run/'manifest.json'),
                  dataset_manifest_sha256=digest(data/'manifest.json'), annotations_sha256=digest(data/'instances_subset.json'),
                  training_images=optimizer['protocol']['training_images'], seeds=10, size=64,
                  area_percent=4., input_size=320, cpu_threads=2, interop_threads=1, location='saliency',
                  optimization_image_forwards=400, clean_image_forwards=200, patched_image_forwards=200,
                  backward_calls=50, optimization_query_unit='one clean OR patched image forwarded once',
                  eot='full, paired clean/patched transform parameters per gradient step',
                  objective=dict(target_class_index=2, confidence_threshold=.25, preservation_patch_threshold=.001,
                                 preservation_nms_iou=.45, lambda_1=.75, lambda_2=0, sign_descent_step=.0005),
                  patch_initialization='uniform [0,1], torch.manual_seed(seed)',
                  optimizer='fixed-budget sign descent, final measured iterate; no test selection or early stop',
                  source=dict(author_repository='https://github.com/AvishagS422/PhantomSponges',
                              commit=AUTHOR_COMMIT, source_url=AUTHOR_SOURCE, utf8_sha256=AUTHOR_SOURCE_UTF8_SHA256,
                              implementation='independent mathematical implementation; no vendored unlicensed source'),
                  adaptations=['YOLOv8n direct decoded class confidence instead of YOLOv5 objectness times class',
                               '64x64 overwrite patch instead of additive whole-image L2 UAP',
                               'same four optimization images and EOT as SG-GA',
                               '400 actual clean+patched image forwards instead of epochs/validation selection',
                               'no wall-time truncation in objective NMS; zero IoU for nonpositive union'],
                  equal_hardware_model_area_training_set_query_budget=True,
                  access_permissions_equal=False, access='white-box gradients, unlike gray-box SG-GA',
                  heldout_images=128, evaluation='same frozen COCO evaluator/postprocessing as SG-GA q400',
                  statistical_role='secondary exploratory added after primary evaluation; not new primary hypothesis',
                  comparisons='paired seeds against SG-GA q400 only; p-values exploratory, not multiplicity adjusted',
                  order_seed=430917, actual_camera_measurement=False, not_exact_published_reproduction=True,
                  does_not_write_manuscripts_or_publish=True)
    protocol_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
    ACTIVE_MANIFEST = output/'manifest.json'
    if ACTIVE_MANIFEST.exists():
        prior = load(ACTIVE_MANIFEST)
        if prior['protocol'] != config:
            raise RuntimeError('Declared local reference protocol/source changed')
        if prior.get('status') == 'complete':
            print('PHANTOM LOCAL STUDY ALREADY COMPLETE', flush=True)
            return
    else:
        write_json(ACTIVE_MANIFEST, dict(declared_utc=datetime.now(timezone.utc).isoformat(),
                                       protocol=config, protocol_sha256=protocol_digest, status='waiting'))
    print('WAITING FOR ACTUAL SOFTWARE STREAM; no inference started', flush=True)
    wait_for_stream(run, args.wait_for_stream_pid)
    for filename, expected in config['dependencies_sha256'].items():
        if digest(ROOT/'experiments'/filename) != expected:
            raise RuntimeError('Frozen local-reference dependency changed')
    for path, expected in [(Path(__file__), config['script_sha256']),
                           (ROOT/'core/victim_model.py', config['victim_source_sha256']),
                           (ROOT/'yolov8n.pt', config['weights_sha256']),
                           (data/'manifest.json', config['dataset_manifest_sha256']),
                           (data/'instances_subset.json', config['annotations_sha256']),
                           (run/'manifest.json', config['optimizer_manifest_sha256'])]:
        if digest(path) != expected:
            raise RuntimeError('Declared reference input changed: '+str(path))
    for record in config['training_images']:
        if digest(ROOT/record['path']) != record['sha256']:
            raise RuntimeError('Optimization image changed')
    import contextlib
    import io
    import numpy as np
    import torch
    from extended_evaluation import COCO, evaluate_case, independent_eot
    from extended_analysis import paired, ratio, baseline_metric
    from local_q1_validation import VictimModel, load_image
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'], key=lambda i:i['id'])
    if len(infos) != config['heldout_images']:
        raise RuntimeError('Unexpected held-out subset size')
    train_ids = {int(Path(item['path']).stem) for item in config['training_images']}
    if train_ids & {i['id'] for i in infos}:
        raise RuntimeError('Optimization/test image IDs overlap')
    hashes = {i['file_name']:i['sha256'] for i in load(data/'manifest.json')['images']}
    for info in infos:
        if digest(data/'images'/info['file_name']) != hashes[info['file_name']]:
            raise RuntimeError('Held-out image changed')
    state = load(ACTIVE_MANIFEST)
    state.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 test_ids=[i['id'] for i in infos], executable=sys.executable, torch_version=torch.__version__)
    write_json(ACTIVE_MANIFEST, state)
    victim = VictimModel(device='cpu')
    victim.model.eval().requires_grad_(False)
    train = [load_image(ROOT/i['path']) for i in config['training_images']]
    order = list(range(10))
    random.Random(config['order_seed']).shuffle(order)
    cases = {}
    for seed in order:
        folder = output/f'phantom_local_seed{seed}'
        optimize(victim, train, seed, folder, protocol_digest)
        case = dict(name='phantom_local', seed=seed, budget=400,
                    spec=dict(location='saliency', budgets=[400]), patch=folder/'patch_q400.npy')
        cases[seed] = evaluate_case(victim, case, infos, coco, data, folder/'evaluation.json')
        independent_eot(victim, case, infos, data, folder/'independent_eot.json')
        print(f'PHANTOM LOCAL seed {seed} optimization/evaluation complete', flush=True)
    comparisons, source_hashes = [], {}
    for field in ['active_candidates', 'nms_ms', 'postprocess_ms']:
        values = []
        for seed in range(10):
            clean_path = run/'evaluation'/f'clean_seed{seed}_q0.json'
            sg_path = run/'evaluation'/f'sg_ga_seed{seed}_q400.json'
            clean, sg = load(clean_path), load(sg_path)
            if clean['image_ids'] != cases[seed]['image_ids'] or sg['image_ids'] != cases[seed]['image_ids']:
                raise RuntimeError('Comparison image IDs differ')
            if field == 'active_candidates':
                values.append(ratio(cases[seed], clean, field)-ratio(sg, clean, field))
            else:
                # Timing is a paired millisecond difference, not a candidate
                # ratio (which deliberately floors count denominators at one).
                local_rows = {r['image_id']:r for r in cases[seed]['records']
                              if r['threshold'] == .01 and r['pre_nms_cap'] is None}
                sg_rows = {r['image_id']:r for r in sg['records']
                           if r['threshold'] == .01 and r['pre_nms_cap'] is None}
                values.append(float(np.mean([local_rows[i][field]-sg_rows[i][field] for i in local_rows])))
            for path in [clean_path, sg_path, output/f'phantom_local_seed{seed}/evaluation.json']:
                source_hashes[str(path.relative_to(ROOT))] = digest(path)
        comparisons.append(dict(metric=field, local_reference_minus_sg400=paired(values),
                                effect_unit='candidate amplification ratio' if field=='active_candidates' else 'milliseconds',
                                exploratory_p_not_multiplicity_adjusted=True,
                                timing_order_not_interleaved_between_studies=True))
    aps = [baseline_metric(cases[s])['coco_subset_bbox']['AP'] for s in range(10)]
    write_json(output/'summary.json', dict(protocol_sha256=protocol_digest, comparisons=comparisons,
                AP_mean_over_patch_seeds=float(np.mean(aps)), AP_seed_values=aps,
                AP_pooled_over_images_first=True, source_sha256=source_hashes,
                n_seeds=10, images_per_seed=128, exact_published_reproduction=False,
                timing_cross_study_drift_possible=True, availability_failure_not_established_by_candidate_ratio=True))
    state = load(ACTIVE_MANIFEST)
    state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(), completed_seeds=10)
    write_json(ACTIVE_MANIFEST, state)
    print('PHANTOM LOCAL STUDY COMPLETE; integrate verified findings separately', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if ACTIVE_MANIFEST is not None and ACTIVE_MANIFEST.exists():
            state = load(ACTIVE_MANIFEST)
            state.update(status='failed', failed_utc=datetime.now(timezone.utc).isoformat(),
                         error_type=type(error).__name__, error_message=str(error))
            write_json(ACTIVE_MANIFEST, state)
        raise
