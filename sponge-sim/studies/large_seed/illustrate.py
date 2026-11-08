"""Show four audit images selected by fixed list positions, never by outcomes."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True); a = ap.parse_args()
    base = a.repo/'sponge-sim/results/large_seed_20261106'
    cohort = json.loads((a.repo/'sponge-sim/studies/large_seed/cohort1500.json').read_text())
    audit = [r for r in cohort['images'] if r['audit']]
    selected = [audit[j] for j in [0, 21, 42, 63]]
    tasks = ['v8', 'v11', 'y26_many', 'y26_native']; names = {r['file_name'] for r in selected}
    records = {}
    for stage in [1000, 1500]:
        for line in (base/f'export/stage{stage}_records.jsonl').read_text().splitlines():
            r = json.loads(line)
            if r['seed'] != 0: continue
            for name in names.intersection(r['images']): records[name, r['task']] = r
    context = zipfile.ZipFile(base/'large_context_backup_20261106/large_context_20261106.zip')
    archives = {stage: zipfile.ZipFile(base/f'export/stage{stage}_all_artifacts.zip') for stage in [1000, 1500]}
    fig, axes = plt.subplots(4, 5, figsize=(12, 10), constrained_layout=True)
    evidence = dict(selection='audit IDs in cohort order at positions0,21,42,63; zero start; no outcome screening', cases=[])
    for i, image in enumerate(selected):
        name = image['file_name']; content = context.read('coco_val_confirm/images/val2017/'+name)
        assert hashlib.sha256(content).hexdigest() == image['sha256']
        clean = np.asarray(Image.open(io.BytesIO(content)).convert('RGB').resize((640,640)))
        clean_k = records[name, 'v8']['clean_native_k'][records[name, 'v8']['images'].index(name)]
        axes[i, 0].imshow(clean); axes[i, 0].set_title(f'Clean: native K={clean_k}', fontsize=9)
        axes[i, 0].set_ylabel(name.removesuffix('.jpg'), fontsize=8)
        for j, task in enumerate(tasks, 1):
            r = records[name, task]; b = r['images'].index(name)
            payload = archives[r['stage']].read(r['record_path']+'/audit_adv.npz')
            assert hashlib.sha256(payload).hexdigest() == r['audit_sha256']
            with np.load(io.BytesIO(payload), allow_pickle=False) as snap:
                index = snap['image_names'].tolist().index(name); adv = snap['adv'][index]
                assert hashlib.sha256(adv.tobytes()).hexdigest() == r['input_sha256'][b]
                axes[i, j].imshow(adv.transpose(1,2,0))
            k = r['attacked_native_k'][b]
            label = {'v8':'v8 transfer', 'v11':'11 transfer', 'y26_many':'26 many objective', 'y26_native':'26 native objective'}[task]
            axes[i,j].set_title(f'{label}: native K={k}', fontsize=9)
            evidence['cases'].append(dict(image=name, task=task, seed=0, native_k=k,
                                          record_path=r['record_path'], input_sha256=r['input_sha256'][b]))
        for ax in axes[i]: ax.set(xticks=[], yticks=[])
    destination = a.repo/'docs/reports/NESTED1500_CODEX_20261106_examples'
    fig.savefig(str(destination)+'.png', dpi=160); fig.savefig(str(destination)+'.pdf'); plt.close(fig)
    with Path(str(destination)+'_sources.json').open('x') as f: json.dump(evidence, f, indent=2)
    context.close()
    for z in archives.values(): z.close()
    print(json.dumps(evidence), flush=True)
