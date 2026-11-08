"""Forward replay of prospectively selected exact-input audit cases (not PGD).

Preserve all mismatches; a replay discrepancy never removes the original case.
Full four-image batch shape is retained, with clean fillers for unaudited slots.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


if __name__ == '__main__':
    import numpy as np
    import torch
    import torchvision
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    import engine
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    assert (a.root / 'DONE').exists()
    a.output.mkdir(exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    base = Path('C:/sponge')
    spec = importlib.util.spec_from_file_location('old', base/'gpu_mechanism.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    variants = [('baseline', 300, 30000), ('maxdet50', 50, 30000), ('prenms300', 300, 300), ('prenms1000', 300, 1000)]
    cases = checks = mismatches = 0; max_native_score_error = 0.; raw = a.output/'raw_replay.jsonl'
    with raw.open('x') as stream, torch.no_grad():
        for task, weight in [('y26_many', 'yolo26n-seg.pt'), ('y26_native', 'yolo26n-seg.pt'), ('v8', 'yolov8n-seg.pt'), ('v11', 'yolo11n-seg.pt')]:
            net = YOLO(str(base/weight)).model.eval().float().cuda()
            target = net if weight == 'yolo26n-seg.pt' else YOLO(str(base/'yolo26n-seg.pt')).model.eval().float().cuda()
            target.model[-1].end2end = True
            for stage in [1000, 1500]:
                folder = a.root/f'stage{stage}'/task
                for seed_dir in sorted(folder.glob('seed*')):
                    offsets = sorted({p.name.split('_')[0] for p in seed_dir.glob('batch*')})
                    for stem in offsets:
                        completed = sorted(p for p in seed_dir.glob(stem+'_attempt*') if (p/'DONE').exists())
                        batch = completed[0]
                        if not (batch/'audit_adv.npz').exists(): continue
                        record = json.loads((batch/'record.json').read_text())
                        with np.load(batch/'audit_adv.npz', allow_pickle=False) as snap, np.load(batch/'outputs.npz', allow_pickle=False) as saved:
                            names = snap['image_names'].tolist(); arrays = snap['adv']
                            clean = torch.cat([old.load_img(base/'coco_val_confirm/images/val2017'/n) for n in record['images']])
                            attacked = clean.clone()
                            indices = [record['images'].index(n) for n in names]
                            for j, index in enumerate(indices):
                                assert hashlib.sha256(arrays[j].tobytes()).hexdigest() == record['input_sha256'][index]
                                attacked[index].copy_(torch.from_numpy(arrays[j]).cuda())
                            assert float((attacked-clean).abs().max()) <= 16/255+1e-6
                            for arm, inputs in [('clean', clean), ('attacked', attacked)]:
                                if target is net: net.model[-1].end2end = False
                                many = engine.prediction(net(inputs), 'many')
                                target.model[-1].end2end = True
                                native = engine.prediction(target(inputs), 'native').cpu().numpy()
                                for j in indices:
                                    differences = []
                                    for v, (variant, maxdet, cap) in enumerate(variants):
                                        actual = len(non_max_suppression(many[j:j+1].clone(), .25, .45, nc=80, max_det=maxdet, max_nms=cap)[0])
                                        expected = record[f'{arm}_variants'][v]['k'][j]
                                        differences.append(dict(variant=variant, expected=expected, replay=actual))
                                    actual_native = int((native[j, :, 4] > .25).sum())
                                    expected_native = record[f'{arm}_native_k'][j]
                                    differences.append(dict(variant='native26', expected=expected_native, replay=actual_native))
                                    error = float(np.abs(native[j, :, 4]-saved[f'{arm}_native'][j, :, 4]).max())
                                    max_native_score_error = max(max_native_score_error, error)
                                    mismatches += sum(d['expected'] != d['replay'] for d in differences); checks += len(differences)
                                    stream.write(json.dumps(dict(record_path=batch.relative_to(a.root).as_posix(), task=task,
                                        seed=record['seed'], image=record['images'][j], arm=arm, comparisons=differences,
                                        max_native_score_error=error))+'\n')
                            cases += len(indices)
                print('AUDIT_PROGRESS', task, stage, cases, checks, mismatches, flush=True)
            del net, target
            torch.cuda.empty_cache()
    assert cases == 64*5*4 and checks == cases*10
    receipt = dict(audit_cases=cases, count_comparisons=checks, count_mismatches=mismatches,
                   max_native_score_error=max_native_score_error,
                   raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   torch=torch.__version__, torchvision=torchvision.__version__,
                   note='Only locked64 exact-input audit subset replayed; remaining cases have saved-output verification only.')
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2)); print(json.dumps(receipt), flush=True)
