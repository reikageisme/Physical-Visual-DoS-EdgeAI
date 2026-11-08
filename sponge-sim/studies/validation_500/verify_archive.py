"""Offline verification of downloaded val500 archive, all hashes and tensor coverage."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path,PureWindowsPath
import numpy as np


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('base',type=Path);a=ap.parse_args()
    receipt=json.loads((a.base/'codex_val500_receipt.json').read_text())
    archive=a.base/receipt['archive'];h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    assert h.hexdigest()==receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==receipt['n_files']
        prefix='val500_full_20261106/'
        rows=[json.loads(s) for s in z.read(prefix+'raw.jsonl').decode().splitlines()]
        mf=json.loads(z.read(prefix+'manifest.json'));assert rows[-1]['kind']=='done'
        cohort={r['file_name'] for r in mf['cohort']['images']};assert len(cohort)==500
        covered={name:[] for name in mf['config']['models']};native=[];hashes=0;maxdelta=0.
        for r in rows:
            if r['kind'] not in ['snapshot','native_snapshot']:continue
            archive_name=prefix+PureWindowsPath(r['file']).as_posix()
            content=z.read(archive_name);assert hashlib.sha256(content).hexdigest()==r['sha256'];hashes+=1
            data=np.load(io.BytesIO(content),allow_pickle=False);names=[str(n) for n in data['image_names']]
            if r['kind']=='snapshot':
                d=data['delta'];assert d.shape==(len(names),3,640,640) and d.dtype==np.float32 and np.isfinite(d).all()
                bound=float(np.abs(d).max());assert bound<=16/255+1e-6;maxdelta=max(maxdelta,bound)
                covered[r['model']].extend(names)
            else:
                for arm in ['clean','attacked']:
                    d=data[arm];assert d.shape==(len(names),300,38) and np.isfinite(d).all()
                    assert (d[:,:,4]>=0).all() and (d[:,:,4]<=1).all()
                native.extend(names)
        for name,names in covered.items():assert len(names)==500 and set(names)==cohort,name
        assert len(native)==500 and set(native)==cohort
        replay=json.loads(z.read(prefix+'replay_receipt.json'))
        assert replay['artifact_hashes_checked']==hashes and replay['images_budget_verified']==1000
        assert replay['native_counts_verified']==1000 and replay['source_count_replays']==8000
    result=dict(archive_sha256=h.hexdigest(),zip_crc_pass=True,n_files=receipt['n_files'],npz_hashes_checked=hashes,
                per_model_coverage={n:len(v) for n,v in covered.items()},native_coverage=len(native),max_delta=maxdelta,
                source_count_replay_mismatches=replay['mismatch_count'],
                note='Original count records retained regardless of numerical replay mismatches.')
    (a.base/'local_archive_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
