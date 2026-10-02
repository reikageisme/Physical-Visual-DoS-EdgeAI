"""Checkpointed local extension with budgets measured in image forwards.

This is a new protocol, not a rewrite of the previous two-image experiment.
The white-box tanh baseline is inspired by Overload, adapted to YOLOv8 class
scores and local overwrite patches; it is NOT an exact Overload reproduction.
No camera, network target, or private image is accessed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'outputs/runtime/coco_eval'))

import cv2
import kornia.augmentation as K
import numpy as np
import torch
from PIL import Image
from attack.genetic_algo import SpongeGA
from core.sponge_fitness import calculate_sponge_fitness
from local_q1_validation import VictimModel, load_image, save_json, sha256


def make_eot(mode):
    layers = []
    if mode != 'no_rotation':
        layers.append(K.RandomRotation(degrees=15., p=.8))
    if mode != 'no_color':
        layers.append(K.ColorJitter(brightness=.2, contrast=.2, p=.8))
    if mode != 'no_blur':
        layers.append(K.RandomGaussianBlur((3, 3), (.1, 1.), p=.5))
    pipeline = K.AugmentationSequential(*layers, data_keys=['input'], same_on_batch=False)

    def transform(x):
        if mode == 'none':
            return x
        x = pipeline(x.contiguous())
        if mode != 'no_noise':
            x = x + torch.randn_like(x) * .02
        return x.clamp(0, 1)
    return transform


def find_locations(images, size, mode, seed):
    if mode == 'center':
        return [[(320 - size) // 2] * 2 for _ in images]
    if mode == 'random':
        rng = random.Random(seed + 98171)
        return [[rng.randrange(321 - size), rng.randrange(321 - size)] for _ in images]
    locations = []
    for image in images:
        gray = cv2.cvtColor((image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        best = (-1., (320-size)//2, (320-size)//2)
        step = max(size//4, 8)
        # Include the boundary; the new location search is explicitly distinct
        # from the legacy loop that did not include the last coordinate.
        coords = sorted(set(range(0, 321-size, step)) | {320-size})
        for y in coords:
            for x in coords:
                score = cv2.Laplacian(gray[y:y+size, x:x+size], cv2.CV_64F).var()
                if score > best[0]:
                    best = score, y, x
        locations.append([best[1], best[2]])
    return locations


def insert(images, patches, locations):
    result = images.unsqueeze(0).expand(len(patches), -1, -1, -1, -1).clone()
    size = patches.shape[-1]
    for index, (y, x) in enumerate(locations):
        result[:, index, :, y:y+size, x:x+size] = patches
    return result.flatten(0, 1).contiguous()


def objective(scores, cap=None, threshold=.01):
    if cap is not None:
        scores = scores.topk(min(cap, scores.shape[-1]), dim=-1).values
    return calculate_sponge_fitness(scores, conf_thresh=threshold)[0]


def specifications():
    rows = []
    for method, location in [('sg_ga', 'saliency'), ('ga_center', 'center'),
                             ('ga_random_location', 'random'), ('random_search', 'saliency')]:
        rows.append(dict(name=method, method=method, location=location, size=64, eot='full', draws=1, budgets=[400, 1000, 2500]))
    for size, name in [(32, 'area_1'), (45, 'area_2'), (90, 'area_8'), (128, 'area_16')]:
        rows.append(dict(name=name, method='sg_ga', location='saliency', size=size, eot='full', draws=1, budgets=[400]))
    for mode in ['none', 'no_rotation', 'no_color', 'no_blur', 'no_noise']:
        rows.append(dict(name='eot_'+mode, method='sg_ga', location='saliency', size=64, eot=mode, draws=1, budgets=[400]))
    rows.append(dict(name='eot_four_draws', method='sg_ga', location='saliency', size=64, eot='full', draws=4, budgets=[400]))
    rows.append(dict(name='adaptive_cap100', method='sg_ga', location='saliency', size=64, eot='full', draws=1, budgets=[400], cap=100))
    rows.append(dict(name='tanh_whitebox_patch', method='gradient_tanh', location='saliency', size=64, eot='full', draws=1, budgets=[400]))
    return rows


def optimize(victim, train_images, spec, seed, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / 'complete.json'
    if completed.exists():
        return json.loads(completed.read_text(encoding='utf-8'))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    size, draws = spec['size'], spec['draws']
    locations = find_locations(train_images, size, spec['location'], seed)
    train = torch.cat(train_images)
    ga = SpongeGA(patch_size=size, pop_size=20, elite_k=5, generations=1, seed=seed)
    transform = make_eot(spec['eot'])
    population = ga.population.cpu()
    patch = population[0].clone()
    best, queries, generation = -float('inf'), 0, 0
    history, checkpoints = [], []
    started = time.perf_counter()
    state_file = run_dir / 'resume.pt'
    resumed_wall = 0.
    if state_file.exists():
        state = torch.load(state_file, map_location='cpu', weights_only=False)
        if state['spec'] != spec:
            raise RuntimeError('Resume specification mismatch')
        population, patch = state['population'], state['patch']
        best, queries, generation = state['best'], state['queries'], state['generation']
        history, checkpoints = state['history'], state['checkpoints']
        resumed_wall = state['wall_seconds']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        rng.setstate(state['python_rng'])

    def checkpoint(budget):
        np.save(run_dir / f'patch_q{budget}.npy', patch.numpy())
        Image.fromarray((patch.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)).save(run_dir / f'patch_q{budget}.png')
        record = dict(seed=seed, specification=spec, queries=budget, best_training_fitness=best,
                      locations=locations, wall_seconds=resumed_wall+time.perf_counter()-started,
                      area_percent=size*size/320**2*100)
        save_json(run_dir / f'checkpoint_q{budget}.json', record)
        checkpoints.append(record)

    maximum = max(spec['budgets'])
    cost = len(train_images)*draws
    if any(b % cost for b in spec['budgets']):
        raise ValueError('Budgets must be divisible by images times EOT draws')
    while queries < maximum:
        generation += 1
        fits = []
        for index in range(len(population)):
            if queries == maximum:
                break
            candidate = population[index].detach().clone()
            if spec['method'] == 'gradient_tanh':
                candidate = patch.detach().clone().requires_grad_(True)
                augmented = transform(insert(train, candidate.unsqueeze(0), locations))
                pred = victim.model(augmented)
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
                scores = pred[:, 4:].max(dim=1).values
                smooth = scores.tanh().sum(dim=1).mean()
                grad, = torch.autograd.grad(smooth, candidate)
                with torch.no_grad():
                    current = float(objective(scores).mean())
                    patch = (candidate + grad.sign()/255.).clamp(0, 1).detach()
                # Current objective describes the pre-update patch; checkpoint
                # must save that exact evaluated candidate, not the next patch.
                evaluated = candidate.detach()
            else:
                with torch.no_grad():
                    values = []
                    for _ in range(draws):
                        x = transform(insert(train, candidate.unsqueeze(0), locations))
                        scores = victim.get_raw_predictions(x)
                        values.append(objective(scores, spec.get('cap')).mean())
                    current = float(torch.stack(values).mean())
                evaluated = candidate
            queries += cost
            fits.append(current)
            if spec['method'] != 'gradient_tanh' and current > best:
                best, patch = current, evaluated.clone()
            if spec['method'] == 'gradient_tanh':
                # White-box records the latest iterate, not stochastic-best
                # selection. Its reported score is on the saved evaluated image.
                best = current
            history.append(dict(queries=queries, fitness=current, selected_fitness=best))
            if queries in spec['budgets']:
                if spec['method'] == 'gradient_tanh':
                    next_patch = patch
                    patch = evaluated.clone()
                    checkpoint(queries)
                    patch = next_patch
                else:
                    checkpoint(queries)
        if queries < maximum:
            if spec['method'] == 'random_search':
                population = torch.rand_like(population)
            elif spec['method'] != 'gradient_tanh':
                top = torch.tensor(fits).topk(5).indices
                elites = population[top].clone()
                parents1 = elites[torch.randint(0, 5, (15,))]
                parents2 = elites[torch.randint(0, 5, (15,))]
                children = ga._crossover(parents1, parents2)
                mask = (torch.rand(15, 1, 1, 1) < .1)
                children = (children+mask*torch.randn_like(children)*.2).clamp(0, 1)
                population = torch.cat([elites, children])
        temporary = state_file.with_suffix('.tmp')
        torch.save(dict(spec=spec, population=population, patch=patch, best=best,
                        queries=queries, generation=generation, history=history, checkpoints=checkpoints,
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=rng.getstate(),
                        wall_seconds=resumed_wall+time.perf_counter()-started), temporary)
        temporary.replace(state_file)
        print(f'{spec["name"]} seed {seed}: {queries}/{maximum} image forwards; score {best:.3f}', flush=True)
    result = dict(seed=seed, spec=spec, checkpoints=checkpoints, history=history,
                  queries=queries, wall_seconds=resumed_wall+time.perf_counter()-started)
    save_json(completed, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='outputs/extended_q1_20261002')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    train_paths = sorted((ROOT / 'outputs/datasets/coco128/images/train2017').glob('*.jpg'))[:4]
    assert len(train_paths) == 4
    train = [load_image(p) for p in train_paths]
    specs = specifications()
    if args.pilot:
        specs = [{**specs[0], 'budgets': [80]}, {**specs[-1], 'budgets': [16]}]
    protocol = dict(seeds=args.seeds, specifications=specs, training_images=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in train_paths],
                    query_unit='one image forwarded once; gradient backward passes reported separately',
                    population=20, elites=5, mutation_probability=.1, mutation_strength=.2,
                    input_size=320, optimization_threshold=.01, independent_eot_draws_for_later_evaluation=16,
                    device='cpu', torch_threads=2, test_selection='COCO val2017 subset specified before optimization',
                    primary_hypothesis='paired seed-level mean test candidate ratio SG-GA minus GA center at 2500 forwards',
                    minimum_relevant_candidate_ratio_difference=.1,
                    secondary_metrics_exploratory=True,
                    statistical_unit='optimizer seed; images nested within seed, timing repeats not independent seeds',
                    whitebox_baseline='Overload-inspired tanh score ascent adapted to local overwrite patches and YOLOv8, not original reproduction',
                    limitations=['COCO val subset not full 5000', 'upstream checkpoint selection on COCO val unknown',
                                 'one physical laptop, no camera or printed patch in this program',
                                 '10 seeds predeclared resource-limited design, not a guaranteed powered study'])
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
                    script_sha256=sha256(__file__), weights_sha256=sha256(ROOT/'yolov8n.pt'),
                    sources_sha256={p:sha256(ROOT/p) for p in ['core/victim_model.py', 'core/eot_transforms.py', 'core/sponge_fitness.py', 'attack/genetic_algo.py']},
                    torch_version=torch.__version__, executable=sys.executable)
    mpath = out / 'manifest.json'
    if mpath.exists():
        prior = json.loads(mpath.read_text(encoding='utf-8'))
        if prior['protocol'] != protocol or prior['script_sha256'] != manifest['script_sha256']:
            raise RuntimeError('Changed protocol/source: choose a new run directory')
    else:
        save_json(mpath, manifest)
    victim = VictimModel(device='cpu')
    victim.model.eval()
    for parameter in victim.model.parameters():
        parameter.requires_grad_(False)
    for seed in range(args.seeds):
        order = specs.copy()
        random.Random(98125+seed).shuffle(order)
        for spec in order:
            optimize(victim, train, spec, seed, out / 'optimization' / f'{spec["name"]}_seed{seed}')
    manifest = json.loads(mpath.read_text(encoding='utf-8'))
    manifest['optimization_finished_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['optimizer_runs'] = len(specs)*args.seeds
    save_json(mpath, manifest)
    print('OPTIMIZATION COMPLETE; evaluation is a separate sequential stage', flush=True)


if __name__ == '__main__':
    main()
