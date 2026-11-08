"""Post-run readback of actual source, all images, and current model weights."""
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    import torch, torchvision, ultralytics
    import ultralytics.nn.modules.head as head
    root = Path('C:/sponge'); run = root/'nested1500_full_20261106'; assert (run/'DONE').exists()
    provenance = json.loads((root/'large_full_provenance.json').read_text())
    files = {'runner_sha256':'large_run.py', 'engine_sha256':'engine.py',
             'cohort_sha256':'large_cohort1500.json', 'lock_sha256':'large_execution_lock.json'}
    assert all(sha(root/name) == provenance[key] for key,name in files.items())
    cohort = json.loads((root/'large_cohort1500.json').read_text())
    assert all(sha(root/'coco_val_confirm/images/val2017'/r['file_name']) == r['sha256'] for r in cohort['images'])
    manifests = {}; weights = {}
    for stage in [1000, 1500]:
        for task in ['v8','v11','y26_many','y26_native']:
            p = run/f'stage{stage}'/task/'manifest.json'; mf = json.loads(p.read_text())
            assert mf['provenance'] == provenance
            for name, expected in mf['weights'].items():
                actual = sha(root/name); assert actual == expected; weights[name] = actual
            manifests[p.relative_to(run).as_posix()] = sha(p)
    receipt = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), root_done=True,
        images_verified=len(cohort['images']), manifests=manifests, weights=weights, provenance=provenance,
        source_hashes={name:sha(root/name) for name in files.values()},
        replay_script_sha256=sha(root/'replay_audit.py'),
        installed_head_source_sha256=sha(Path(head.__file__)),
        torch=torch.__version__, torchvision=torchvision.__version__, ultralytics=ultralytics.__version__,
        note='Post-run readback, not a new attack or timing experiment. Does not replace saved-input forward replay.')
    output = root/'large_context_readback_20261106.json'
    with output.open('x') as f: json.dump(receipt, f, indent=2)
    print(json.dumps(receipt), flush=True)
