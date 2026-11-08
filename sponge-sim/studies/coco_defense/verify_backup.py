"""Verify downloaded evidence archive and extract only non-tensor evidence locally."""
import hashlib
import json
import zipfile
from pathlib import Path


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[3]
    base=repo/'sponge-sim/results/confirmation_20261106'
    receipt=json.loads((base/'codex_evidence_receipt.json').read_text())
    archive=base/receipt['archive']; h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    assert h.hexdigest()==receipt['sha256']
    out=repo/'_provenance/codex_gpu_20261106';out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==receipt['n_files']
        for name in names:
            if name.endswith('.pt'):continue  # complete tensors and weights stay in verified ZIP
            p=(out/name).resolve();assert p.is_relative_to(out.resolve())
            content=z.read(name);p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==content
            else:p.write_bytes(content)
        for mf_name in ['m3_native_full_20261106/manifest.json','m4_full_20261106/manifest.json','m4b_full_20261106/manifest.json']:
            mf=json.loads(z.read(mf_name))
            if 'replay_inputs' in mf:
                for image,digest in mf['replay_inputs'].items():
                    name='m3_native_full_20261106/'+Path(image).stem+'_inputs.pt'
                    assert hashlib.sha256(z.read(name)).hexdigest()==digest
        for mf_name in ['m3_native_full_20261106/manifest.json','coco_defense_full_20261106/manifest.json']:
            for name,digest in json.loads(z.read(mf_name))['weights'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==digest
    result=dict(archive_sha256=receipt['sha256'],zip_crc_pass=True,n_files=len(names),
                replay_input_hashes_pass=True,weights_hashes_pass=True,extracted_non_tensor_dir=str(out),
                note='Tensor inputs, native YOLO26 outputs and pretrained weights retained inside verified local ZIP.')
    (base/'local_backup_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
