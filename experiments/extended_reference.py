"""Run unedited public attack functions from the Underload reference repository.

This is a CPU/runtime port of the repository's reference VARIANTS, not a claim
to replicate the original papers' complete experiments. They use additive
full-image UAPs, YOLOv5s, native 640 letterboxing and a norm bound of 70; do not
rank them against local 4%-area YOLOv8 overwrite patches as equal-threat tests.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.machinery
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch as mock_patch

from extended_validation import ROOT
from extended_evaluation import COCO,coco_metrics
from local_q1_validation import save_json,sha256
from prepare_extended_validation import download
import numpy as np
import torch
from PIL import Image

VENDOR = ROOT/'outputs/vendor_underload'
COMMIT = '33fb5f348790535a2c819b2cf38a13bebf8a8e54'
PACKAGE = VENDOR/'robust_pkgs/ultralytics-yolov5-robust-0.0.1/yolov5'
WEIGHT = ROOT/'outputs/weights/reference/yolov5s.pt'


def namespace(name,path):
    # The source package __init__ imports unrelated train/export CLIs. Load
    # only its actual model/data/attack modules, without executing those CLIs.
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path/'__init__.py')
    module.__spec__ = importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    sys.modules[name] = module


def initialize():
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip()
    if revision != COMMIT or subprocess.check_output(['git','diff','--name-only'],cwd=VENDOR,text=True).strip():
        raise RuntimeError('Reference checkout changed; do not silently alter published functions')
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    os.environ['YOLOV5_CONFIG_DIR'] = str(ROOT/'outputs/runtime/reference_config')
    sys.path.insert(0,str(ROOT/'outputs/runtime/reference'))
    sys.path.insert(0,str(PACKAGE))
    namespace('yolov5',PACKAGE)
    import models.yolo as yolo
    # The README links a v7 checkpoint but this source names its model Model.
    # Resolve only the pickle class alias; no network/loss parameters change.
    if not hasattr(yolo,'DetectionModel'):
        yolo.DetectionModel = yolo.Model
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.dataloaders import LoadImages
    from yolov5.utils.general import non_max_suppression,scale_coords
    original_load = torch.load

    def fixed_checkpoint_load(path,*args,**kwargs):
        # Full module pickle is restricted to this one observed official file.
        if Path(path).resolve() != WEIGHT.resolve():
            raise RuntimeError('Unexpected reference checkpoint path')
        kwargs['weights_only'] = False
        return original_load(path,*args,**kwargs)

    with mock_patch.object(torch,'load',fixed_checkpoint_load):
        model = attempt_load(str(WEIGHT),torch.device('cpu'),False).float().eval()
    model.requires_grad_(False)
    sys.path.insert(0,str(VENDOR))
    import latency_attack as attacks
    attacks.device = torch.device('cpu')
    return model,attacks,LoadImages,non_max_suppression,scale_coords


def prepare():
    WEIGHT.parent.mkdir(parents=True,exist_ok=True)
    url = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
    download(url,WEIGHT)
    save_json(WEIGHT.parent/'manifest.json',dict(url=url,sha256=sha256(WEIGHT),reference_commit=COMMIT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',default='outputs/extended_q1_20261002')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    expected = json.loads((WEIGHT.parent/'manifest.json').read_text())['sha256']
    if sha256(WEIGHT) != expected:
        raise RuntimeError('Reference checkpoint changed')
    model,attacks,LoadImages,nms,scale_coords = initialize()
    if args.smoke:
        files = sorted((ROOT/'outputs/datasets/coco_val2017_extended/images').glob('*.jpg'))
        dataset = LoadImages(str(files[0]),img_size=640,stride=32,auto=False)
        _,im,im0,_,_ = next(iter(dataset))
        tensor = torch.from_numpy(im).float()[None]/255
        with torch.no_grad():
            raw = model(tensor)[0]
            found = nms(raw.clone(),.25,.45)[0]
        scores = raw.clone().requires_grad_(True)
        _,loss = attacks.max_objects(scores)
        grad, = torch.autograd.grad(loss,scores)
        assert torch.isfinite(raw).all() and torch.isfinite(grad).all()
        algorithm_checks = []
        for name in ['phantom_sponges','overload']:
            small = LoadImages(str(files[0].parent),img_size=640,stride=32,auto=False)
            small.files = small.files[:2]
            small.nf = len(small.files)
            small.video_flag = [False]*small.nf
            folder = WEIGHT.parent/f'smoke_{name}_{time.time_ns()}'
            assert not folder.exists() and folder.resolve().is_relative_to(WEIGHT.parent.resolve())
            captured = {}
            original_to_image = attacks.transp

            def capture(tensor):
                captured['perturbation'] = tensor.detach().clone()
                return original_to_image(tensor)

            with mock_patch.object(attacks,'transp',capture):
                if name=='phantom_sponges':
                    attacks.phantom_sponges(model,small,640,1,10,0,str(folder)+os.sep)
                else:
                    folder.mkdir()
                    attacks.overload(model,small,640,str(folder)+os.sep)
            assert torch.isfinite(captured['perturbation']).all()
            algorithm_checks.append(dict(function=name,actual_program_check_images=2,
                                         finite_output=True,not_scientific_benchmark=True))
        save_json(WEIGHT.parent/'smoke.json',dict(reference_commit=COMMIT,raw_shape=list(raw.shape),
                                                 image_sha256=sha256(files[0]),final_detections=len(found),
                                                 algorithm_checks=algorithm_checks,
                                                 not_scientific_latency_results=True,
                                                 compatibility_alias='DetectionModel -> Model for documented v7 weight'))
        print('REFERENCE REAL-IMAGE AND LOSS CHECK PASSED',flush=True)
        return
    run = ROOT/args.run_dir
    optimizer = json.loads((run/'manifest.json').read_text())
    if 'optimization_finished_utc' not in optimizer:
        raise RuntimeError('Finish primary optimization before more CPU work')
    output = run/'reference'
    output.mkdir(exist_ok=True)
    train_folder = ROOT/'outputs/datasets/coco128/images/train2017'
    train_files = sorted(train_folder.glob('*.jpg'))
    data = ROOT/'outputs/datasets/coco_val2017_extended'
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(data/'instances_subset.json'))
    infos = sorted(coco.dataset['images'],key=lambda i:i['id'])[:32]
    categories = sorted(coco.getCatIds())
    config = dict(script_sha256=sha256(__file__),reference_repository='https://github.com/Hill-Wu-1998/underload',
                  reference_commit=COMMIT,original_attack_sha256=sha256(VENDOR/'latency_attack.py'),
                  original_loss_sha256=sha256(VENDOR/'adv_loss.py'),weights_sha256=sha256(WEIGHT),
                  dataset_sha256=sha256(data/'manifest.json'),training_files_sha256={p.name:sha256(p) for p in train_files},
                  model='yolov5s',runtime_port=torch.__version__,device='cpu',input_size=640,
                  evaluation_confidence_threshold=.25,evaluation_nms_iou=.45,evaluation_max_det=300,
                  nominal_training_norm_bound=70,actual_final_norm_reported_separately=True,
                  functions=['phantom_sponges','overload'],seeds=10,train_order_seed_offset=20775,
                  input_boundary='native letterbox; additive full-image UAP clipped with L2 bound 70',
                  phantom_epochs=3,phantom_l1=1,phantom_l2=10,phantom_l3=0,overload_epochs=1,
                  original_functions_unedited=True,test_ids=[i['id'] for i in infos],warmup=1,repeats=3,
                  original_paper_experiment_reproduction=False,equal_local_patch_threat_comparison=False,
                  source_variant_warning='Underload repository reference variants; different model, image size, training set, threat and query budget',
                  compatibility='only selected modules; fixed official pickle load; DetectionModel alias; CPU and current runtime',
                  timing_boundary='prepared native 640 tensor through model and native NMS; excludes camera/IO/letterbox')
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['protocol'] != config:
        raise RuntimeError('Changed reference protocol/source')
    if not manifest.exists():
        save_json(manifest,dict(started_utc=datetime.now(timezone.utc).isoformat(),protocol=config))
    order = [(seed,name) for seed in range(10) for name in config['functions']]
    random.Random(188715).shuffle(order)
    for seed,name in order:
        target = output/f'{name}_seed{seed}.json'
        if target.exists():
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        dataset = LoadImages(str(train_folder),img_size=640,stride=32,auto=False)
        random.Random(20775+seed).shuffle(dataset.files)
        # A fresh, never-existing output path prevents the public function's
        # deletion-on-reuse branch from running. Preserve failed attempts.
        folder = output/f'{name}_seed{seed}_{time.time_ns()}'
        assert folder.resolve().is_relative_to(output.resolve()) and not folder.exists()
        captured = {}
        original_to_image = attacks.transp
        count = [0]

        def capture(tensor):
            captured['perturbation'] = tensor.detach().cpu().clone()
            return original_to_image(tensor)

        def count_forward(module,arguments):
            count[0] += len(arguments[0])

        handle = model.register_forward_pre_hook(count_forward)
        grad_count = [0]
        original_grad = torch.autograd.grad

        def count_gradient(*arguments,**keywords):
            grad_count[0] += 1
            return original_grad(*arguments,**keywords)
        try:
            with mock_patch.object(attacks,'transp',capture),mock_patch.object(torch.autograd,'grad',count_gradient):
                if name=='phantom_sponges':
                    attacks.phantom_sponges(model,dataset,640,1,10,0,str(folder)+os.sep)
                else:
                    folder.mkdir()
                    attacks.overload(model,dataset,640,str(folder)+os.sep)
        finally:
            handle.remove()
        perturbation = captured['perturbation']
        if not torch.isfinite(perturbation).all():
            raise RuntimeError('Reference optimizer returned non-finite perturbation')
        np.save(folder/'perturbation_float32.npy',perturbation.numpy())
        predicted = dict(clean=[],patched=[])
        records = []
        rng = random.Random(191517+seed)
        test = infos.copy()
        rng.shuffle(test)
        for info in test:
            loader = LoadImages(str(data/'images'/info['file_name']),img_size=640,stride=32,auto=False)
            _,im,im0,_,_ = next(iter(loader))
            clean = torch.from_numpy(im).float()[None]/255
            adv = (clean+perturbation[None]).clamp(0,1)
            conditions = [('clean',clean),('patched',adv)]
            rng.shuffle(conditions)
            for condition,image in conditions:
                samples = []
                with torch.no_grad():
                    for repeat in range(4):
                        started = time.perf_counter_ns()
                        raw = model(image)[0]
                        forwarded = time.perf_counter_ns()
                        # Actual class-score confidence, not just objectness.
                        active = ((raw[...,4:5]*raw[...,5:]).max(-1).values>.25).sum().item()
                        found = nms(raw.clone(),.25,.45,max_det=300)[0]
                        ended = time.perf_counter_ns()
                        if repeat:
                            samples.append(dict(forward_ms=(forwarded-started)/1e6,
                                                postprocess_ms=(ended-forwarded)/1e6,total_ms=(ended-started)/1e6,
                                                active_candidates=active,final_detections=len(found)))
                    found[:,:4] = scale_coords(image.shape[2:],found[:,:4],im0.shape)
                for x1,y1,x2,y2,score,cls in found.cpu().tolist():
                    predicted[condition].append(dict(image_id=info['id'],category_id=categories[int(cls)],score=score,
                                                     bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)]))
                records.append(dict(condition=condition,image_id=info['id'],samples=samples))
        metrics = {c:coco_metrics(coco,predicted[c],[i['id'] for i in infos]) for c in predicted}
        save_json(target,dict(method=name,seed=seed,training_image_forwards=count[0],
                               backward_calls=grad_count[0],actual_final_l2_norm=float(perturbation.norm()),
                               perturbation_min=float(perturbation.min()),perturbation_max=float(perturbation.max()),
                               trained_float_perturbation_sha256=sha256(folder/'perturbation_float32.npy'),
                               perturbation_file=str((folder/'perturbation_float32.npy').relative_to(ROOT)),
                               training_order=dataset.files,records=records,predictions=predicted,coco_subset_bbox=metrics,
                               not_original_paper_reproduction=True,not_equal_local_patch_threat=True))
        print(f'REFERENCE {name} seed {seed} complete; actual forwards {count[0]}',flush=True)
        gc.collect()
    final = json.loads(manifest.read_text())
    final['finished_utc'] = datetime.now(timezone.utc).isoformat()
    final['evaluated_cases'] = len(order)
    save_json(manifest,final)


if __name__ == '__main__':
    main()
