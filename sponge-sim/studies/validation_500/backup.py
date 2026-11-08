"""Preserve every expanded-cohort artifact plus this continuation's raw calibration."""
import hashlib
import json
import zipfile
from pathlib import Path


if __name__=='__main__':
    root=Path('C:/sponge');paths=[]
    for name in ['val500_full_20261106','val500_smoke_20261106','stability_cal_20261106',
                 'stability_full_20261106','stability_smoke_20261106','conditioning_cal_20261106']:
        paths.extend(p for p in (root/name).rglob('*') if p.is_file())
    paths.extend(root/n for n in ['val500_run.py','val500_cohort.json','val500_provenance.json','val500_replay.py',
        'stability_cal.py','stability_cal_provenance.json','stability_full.py','stability_full_provenance.json',
        'stability_selection.json','conditioning_cal.py','conditioning_provenance.json','gpu_mechanism.py'])
    run=root/'val500_full_20261106';rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    assert rows[-1]['kind']=='done'
    for r in rows:
        if r['kind'] in ['snapshot','native_snapshot']:
            assert hashlib.sha256((run/r['file']).read_bytes()).hexdigest()==r['sha256']
    archive=root/'codex_val500_evidence_20261106.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
        for p in paths:z.write(p,p.relative_to(root))
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    receipt=dict(archive=archive.name,sha256=h.hexdigest(),bytes=archive.stat().st_size,n_files=len(paths),
                 note='Includes all deltas/native outputs; required pretrained weights retained in prior verified codex_evidence_20261106.zip.')
    (root/'codex_val500_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
