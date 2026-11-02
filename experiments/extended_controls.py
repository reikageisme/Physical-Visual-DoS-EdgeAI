"""Additional controls, isolated from the main prospective optimizer protocol.

Noise/module patches have zero optimization queries. The confidence-reducing
gradient patch is a local integrity-control adaptation, not a published attack
reproduction. Test images never select any of these patches.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from extended_validation import ROOT, find_locations, insert, make_eot, objective
from extended_evaluation import COCO, evaluate_case
from local_q1_validation import VictimModel, load_image, save_json, sha256
import numpy as np
import torch
from PIL import Image

CONTROLS = ['uniform_noise', 'gaussian_noise', 'checkerboard', 'binary_modules', 'solid_gray', 'integrity_gradient']


def pattern(name, seed):
    generator = torch.Generator().manual_seed(591715+seed)
    if name == 'uniform_noise':
        return torch.rand(3,64,64, generator=generator)
    if name == 'gaussian_noise':
        return (.5+.25*torch.randn(3,64,64, generator=generator)).clamp(0,1)
    if name == 'solid_gray':
        return torch.full((3,64,64), .5)
    if name == 'checkerboard':
        rows, cols = torch.meshgrid(torch.arange(64),torch.arange(64), indexing='ij')
        return (((rows//4+cols//4)%2).float()).expand(3,-1,-1).clone()
    if name == 'binary_modules':
        grid = torch.randint(0,2,(1,1,16,16), generator=generator).float()
        return torch.nn.functional.interpolate(grid,size=(64,64),mode='nearest')[0].expand(3,-1,-1).clone()
    raise ValueError(name)


def integrity(victim, train, seed, folder):
    completed = folder/'optimization.json'
    if completed.exists():
        return
    torch.manual_seed(seed)
    patch = torch.rand(3,64,64)
    locations = find_locations(list(train.split(1,dim=0)),64,'saliency',seed)
    transform = make_eot('full')
    history = []
    start_step = 0
    evaluated = patch.clone()
    state = folder/'resume.pt'
    if state.exists():
        saved = torch.load(state, weights_only=False, map_location='cpu')
        if saved['seed'] != seed:
            raise RuntimeError('Resume seed mismatch')
        patch, history, start_step = saved['patch'], saved['history'], saved['step']
        evaluated = saved['evaluated']
        torch.set_rng_state(saved['torch_rng'])
    for step in range(start_step,100):
        candidate = patch.detach().clone().requires_grad_(True)
        augmented = transform(insert(train,candidate[None],locations))
        raw = victim.model(augmented)
        raw = raw[0] if isinstance(raw,(tuple,list)) else raw
        scores = raw[:,4:].max(1).values
        loss = scores.tanh().sum(1).mean()
        grad, = torch.autograd.grad(loss,candidate)
        with torch.no_grad():
            history.append(dict(step=step+1,image_forwards=(step+1)*4,
                                smooth_confidence_sum=float(loss),sponge_fitness=float(objective(scores).mean())))
            evaluated = candidate.detach().clone()
            patch = (candidate-grad.sign()/255.).clamp(0,1).detach()
        temporary = state.with_suffix('.tmp')
        torch.save(dict(seed=seed,patch=patch,evaluated=evaluated,step=step+1,history=history,torch_rng=torch.get_rng_state()),temporary)
        temporary.replace(state)
    # Save the patch actually measured at the last query, not the next iterate.
    np.save(folder/'patch.npy',evaluated.numpy())
    Image.fromarray((evaluated.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(folder/'patch.png')
    save_json(completed,dict(seed=seed,history=history,queries=400,backward_calls=100,
                              backward_image_examples=400,locations=locations,objective='minimize sum tanh(max class score)',
                              final_iterate_not_stochastic_best=True,not_exact_published_reproduction=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    args = parser.parse_args()
    run = ROOT/args.run_dir
    optimizer = json.loads((run/'manifest.json').read_text())
    if 'optimization_finished_utc' not in optimizer:
        raise RuntimeError('Finish main optimization before extra work on this CPU')
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'], key=lambda i:i['id'])
    output = run/'controls'
    output.mkdir(exist_ok=True)
    config = dict(script_sha256=sha256(__file__),optimizer_manifest_sha256=sha256(run/'manifest.json'),
                  evaluator_sha256=sha256(ROOT/'experiments/extended_evaluation.py'),
                  data_manifest_sha256=sha256(data/'manifest.json'),controls=CONTROLS,seeds=10,
                  test_ids=[i['id'] for i in infos],size=64,location='saliency',integrity_image_forwards=400,
                  integrity_backward_calls=100,control_order_seed=772017,
                  noise_optimization_queries=0,gaussian_mean=.5,gaussian_std=.25,
                  module_definition='binary 16x16 grid enlarged nearest to 64; QR-like texture, NOT a valid QR code',
                  integrity_definition='local overwrite patch minimizing tanh class-score sum, not published reproduction',
                  interpretation='secondary exploratory controls added before extended held-out evaluation')
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['protocol'] != config:
        raise RuntimeError('Control protocol/source changed')
    if not manifest.exists():
        save_json(manifest,dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol=config))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    victim = VictimModel(device='cpu')
    victim.model.requires_grad_(False)
    train = torch.cat([load_image(ROOT/p['path']) for p in optimizer['protocol']['training_images']])
    cases = [(seed,name) for seed in range(10) for name in CONTROLS]
    random.Random(config['control_order_seed']).shuffle(cases)
    for seed,name in cases:
        folder = output/f'{name}_seed{seed}'
        folder.mkdir(exist_ok=True)
        if name == 'integrity_gradient':
            integrity(victim,train,seed,folder)
        elif not (folder/'patch.npy').exists():
            patch = pattern(name,seed)
            np.save(folder/'patch.npy',patch.numpy())
            Image.fromarray((patch.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(folder/'patch.png')
        case = dict(name=name,seed=seed,budget=400 if name=='integrity_gradient' else 0,
                    spec=dict(location='saliency',budgets=[400] if name=='integrity_gradient' else [0]),
                    patch=folder/'patch.npy')
        evaluate_case(victim,case,infos,coco,data,folder/'evaluation.json')
        print(f'CONTROL {name} seed {seed} complete',flush=True)
    final = json.loads(manifest.read_text())
    final['finished_utc'] = datetime.now(timezone.utc).isoformat()
    final['evaluated_cases'] = len(cases)
    save_json(manifest,final)


if __name__ == '__main__':
    main()
