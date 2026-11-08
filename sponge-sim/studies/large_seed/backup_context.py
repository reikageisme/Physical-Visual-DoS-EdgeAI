"""Copy fixed study inputs and source into an independent, hashed ZIP.

Read-only with respect to the running count campaign. No mutable batch folders.
"""
import hashlib
import json
import zipfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    root = Path('C:/sponge'); output = root/'large_context_backup_20261106'; output.mkdir(exist_ok=False)
    cohort = json.loads((root/'large_cohort1500.json').read_text())
    fixed = ['large_run.py', 'engine.py', 'large_cohort1500.json', 'large_execution_lock.json',
             'large_full_provenance.json', 'gpu_mechanism.py', 'yolov8n-seg.pt', 'yolo11n-seg.pt', 'yolo26n-seg.pt']
    paths = [root/name for name in fixed]
    for r in cohort['images']:
        p = root/'coco_val_confirm/images/val2017'/r['file_name']; assert sha(p) == r['sha256']; paths.append(p)
    for name in ['diagnose26_20261106', 'large_engine_pilot_20261106', 'fusion_engine_pilot_20261106']:
        paths.extend(p for p in (root/name).rglob('*') if p.is_file())
    if (root/'head_source_20261106.txt').exists(): paths.append(root/'head_source_20261106.txt')
    archive = output/'large_context_20261106.zip'; manifest = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(set(paths)):
            rel = p.relative_to(root).as_posix(); digest = sha(p)
            z.write(p, rel); manifest.append(dict(path=rel, bytes=p.stat().st_size, sha256=digest))
    receipt = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), files=manifest,
                   cohort_images=len(cohort['images']), note='Fixed source images/weights/code and prior diagnostic artifacts; no in-progress output copied.')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}), flush=True)
