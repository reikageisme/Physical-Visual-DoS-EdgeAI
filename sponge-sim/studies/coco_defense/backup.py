"""Archive only this session's evidence and required model/source files; preserve originals."""
import hashlib
import json
import zipfile
from pathlib import Path


if __name__=='__main__':
    root=Path('C:/sponge')
    runs=['m4_full_20261106','m4b_smoke_20261106','m4b_full_20261106',
          'm2_cpu1_smoke_20261106','m2_cpu1_full_20261106','m2_cpu4_full_20261106','m2_gpu_full_20261106',
          'm3_native_smoke_20261106','m3_native_full_20261106',
          'coco_defense_smoke_20261106','coco_defense_smoke_v2_20261106','coco_defense_full_20261106',
          'cap_smoke_20261106','cap_full_20261106']
    files=[p for run in runs for p in (root/run).rglob('*') if p.is_file()]
    files += [root/p for p in ['yolov8n-seg.pt','yolo11n-seg.pt','yolo26n-seg.pt','gpu_mechanism.py','gpu_full.py',
                              'm4_run.py','m4b_run.py','m2_confirm.py','m3_confirm.py','coco_defense.py','cap_probe.py',
                              'm2_hw.py','validator_source.txt','coco_full.log','coco_smoke_v2.log',
                              'm4_provenance.json','m4b_provenance.json','confirm_provenance.json','cap_provenance.json',
                              'coco_val_confirm/dataset_manifest.json','coco_val_confirm/val.yaml']]
    archive=root/'codex_evidence_20261106.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in files:z.write(p,p.relative_to(root))
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    receipt=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=h.hexdigest(),n_files=len(files))
    (root/'codex_evidence_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
