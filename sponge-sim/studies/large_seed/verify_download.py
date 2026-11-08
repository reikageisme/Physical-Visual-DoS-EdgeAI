"""Verify local stage archive SHA, CRC, canonical record hashes and NPZ counts."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from export_stage import sha


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('folder', type=Path)
    ap.add_argument('--stage', type=int, choices=[1000, 1500], required=True); a = ap.parse_args()
    receipt = json.loads((a.folder/f'stage{a.stage}_receipt.json').read_text())
    archive = a.folder/f'stage{a.stage}_all_artifacts.zip'
    raw = a.folder/f'stage{a.stage}_records.jsonl'
    assert sha(archive) == receipt['archive_sha256'] and sha(raw) == receipt['raw_sha256']
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    cases = audits = count_checks = native_checks = 0
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == receipt['files'] and z.testzip() is None
        assert f'stage{a.stage}/DONE' in z.namelist()
        for row in rows:
            prefix = row['record_path']+'/'
            record = z.read(prefix+'record.json')
            assert hashlib.sha256(record).hexdigest() == row['record_sha256']
            r = json.loads(record)
            assert all(row[k] == v for k, v in r.items())
            content = z.read(prefix+'outputs.npz')
            assert hashlib.sha256(content).hexdigest() == r['output_sha256']
            with np.load(io.BytesIO(content), allow_pickle=False) as data:
                assert data['image_names'].tolist() == r['images']
                for arm in ['clean', 'attacked']:
                    assert np.array_equal((data[f'{arm}_native'][:, :, 4] > .25).sum(1), r[f'{arm}_native_k'])
                    native_checks += 4
                    for v in range(4):
                        assert np.array_equal((data[f'{arm}_detections'][:, v, :, 4] > .25).sum(1), r[f'{arm}_variants'][v]['k'])
                        count_checks += 4
                cases += 4
            if r['audit_sha256']:
                content = z.read(prefix+'audit_adv.npz')
                assert hashlib.sha256(content).hexdigest() == r['audit_sha256']
                with np.load(io.BytesIO(content), allow_pickle=False) as data:
                    for name, adv in zip(data['image_names'].tolist(), data['adv']):
                        assert hashlib.sha256(adv.tobytes()).hexdigest() == r['input_sha256'][r['images'].index(name)]
                        audits += 1
    assert cases == receipt['cases'] and audits == receipt['audit_cases'] and count_checks == receipt['count_checks'] and native_checks == receipt['native_count_checks']
    result = dict(stage=a.stage, archive_sha256=receipt['archive_sha256'], zip_crc_pass=True,
                  cases=cases, audit_cases=audits, saved_count_checks=count_checks, native_count_checks=native_checks,
                  note='Verifies downloaded saved artifacts; forward replay is separately reported on locked64 audit subset.')
    with (a.folder/f'stage{a.stage}_local_verification.json').open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
