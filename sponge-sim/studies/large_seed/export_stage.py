"""Export a CLOSED stage, verify every saved output, and archive all attempts.

No inference or timing. The archive retains unfinished attempts as well as the
first completed canonical attempt. Never run on a stage without its DONE marker.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def export(root, stage, cohort, output):
    folder = root / f'stage{stage}'
    assert (folder / 'DONE').exists()
    cfg = json.loads(cohort.read_text())
    images = cfg['images'][:1000] if stage == 1000 else cfg['images'][1000:1500]
    audit_ids = {r['file_name'] for r in images if r['audit']}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    audit_cases = checks = native_checks = 0
    for task in ['y26_many', 'y26_native', 'v8', 'v11']:
        task_dir = folder / task
        assert (task_dir / 'DONE').exists()
        manifest = json.loads((task_dir / 'manifest.json').read_text())
        assert manifest['images'] == images and not manifest['smoke']
        for seed in cfg['seeds']:
            seed_dir = task_dir / f'seed{seed}'
            assert (seed_dir / 'DONE').exists()
            for offset in range(0, len(images), 4):
                completed = sorted(p for p in seed_dir.glob(f'batch{offset:04d}_attempt*') if (p / 'DONE').exists())
                assert completed
                batch = completed[0]
                r = json.loads((batch / 'record.json').read_text())
                names = [v['file_name'] for v in images[offset:offset + 4]]
                assert (r['task'], r['stage'], r['seed'], r['offset'], r['images']) == (task, stage, seed, offset, names)
                assert r['max_delta'] <= 16 / 255 + 1e-6
                assert all((v == 0 if seed == 0 else v > 0) for v in r['initial_delta_l2'])
                assert sha(batch / 'outputs.npz') == r['output_sha256']
                with np.load(batch / 'outputs.npz', allow_pickle=False) as data:
                    assert data['image_names'].tolist() == names
                    for arm in ['clean', 'attacked']:
                        det = data[f'{arm}_detections']
                        native = data[f'{arm}_native']
                        scores = data[f'{arm}_score_max']
                        assert det.shape == (4, 4, 300, 6) and native.shape == (4, 300, 38) and scores.shape == (4, 8400)
                        assert all(np.isfinite(v).all() for v in [det, native, scores])
                        assert ((scores >= 0) & (scores <= 1)).all()
                        assert ((native[:, :, 4] >= 0) & (native[:, :, 4] <= 1)).all()
                        assert (native[:, :, 5] == np.floor(native[:, :, 5])).all()
                        assert np.array_equal((native[:, :, 4] > .25).sum(1), r[f'{arm}_native_k'])
                        native_checks += 4
                        ordered = np.sort(native[:, :, 4], axis=1)[:, ::-1]
                        r[f'{arm}_native_confidence_ranks_1_50_100_300'] = ordered[:, [0, 49, 99, 299]].tolist()
                        for j, (variant, cap, maxdet) in enumerate([('baseline', 30000, 300), ('maxdet50', 30000, 50), ('prenms300', 300, 300), ('prenms1000', 1000, 300)]):
                            record = r[f'{arm}_variants'][j]
                            assert record['variant'] == variant and max(record['kernel_n']) <= cap
                            assert np.array_equal((det[:, j, :, 4] > .25).sum(1), record['k'])
                            assert max(record['k']) <= maxdet
                            checks += 4
                    audited = [n for n in names if n in audit_ids]
                    if audited:
                        assert sha(batch / 'audit_adv.npz') == r['audit_sha256']
                        with np.load(batch / 'audit_adv.npz', allow_pickle=False) as snap:
                            assert snap['image_names'].tolist() == audited
                            assert snap['adv'].shape == (len(audited), 3, 640, 640)
                            for name, adv in zip(audited, snap['adv']):
                                assert adv.dtype == np.float32 and np.isfinite(adv).all() and adv.min() >= 0 and adv.max() <= 1
                                assert hashlib.sha256(adv.tobytes()).hexdigest() == r['input_sha256'][names.index(name)]
                            audit_cases += len(audited)
                    else:
                        assert r['audit_sha256'] is None and not (batch / 'audit_adv.npz').exists()
                rows.append(dict(record_path=batch.relative_to(root).as_posix(), record_sha256=sha(batch / 'record.json'), **r))
    raw = output / f'stage{stage}_records.jsonl'
    with raw.open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    archive = output / f'stage{stage}_all_artifacts.zip'
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    receipt = dict(stage=stage, n_images=len(images), cases=len(rows) * 4, batches=len(rows), count_checks=checks, native_count_checks=native_checks,
                   audit_cases=audit_cases, expected_audit_cases=len(audit_ids) * 20, files=len(files),
                   cohort_sha256=sha(cohort), raw_sha256=sha(raw), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    assert audit_cases == receipt['expected_audit_cases']
    (output / f'stage{stage}_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--stage', type=int, choices=[1000, 1500], required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    export(a.root, a.stage, a.cohort, a.output)
