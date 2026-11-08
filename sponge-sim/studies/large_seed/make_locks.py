"""Lock nested cohorts and select runtime by correctness/throughput only."""
import hashlib
import json
import random
from pathlib import Path


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[3];folder=Path(__file__).parent
    previous=json.loads((repo/'sponge-sim/studies/validation_500/cohort.json').read_text())
    dataset=repo/'_provenance/codex_gpu_20261106/coco_val_confirm/dataset_manifest.json'
    all_images=json.loads(dataset.read_text())['images'];old=[r['file_name'] for r in previous['images']]
    remaining=sorted(set(all_images)-set(old));new=random.Random(20261107).sample(remaining,1000)
    names=old+sorted(new[:500])+sorted(new[500:])
    assert len(set(names))==1500
    audit=set(random.Random(20261108).sample(sorted(names),64))
    cohort=dict(population=5000,old_cohort_sha256=hashlib.sha256((repo/'sponge-sim/studies/validation_500/cohort.json').read_bytes()).hexdigest(),
        extension_seed=20261107,audit_seed=20261108,stages=[1000,1500],seeds=[0,12345,23456,34567,45678],
        images=[dict(index=i,file_name=n,sha256=all_images[n],audit=n in audit,origin='old500' if i<500 else ('extend1000' if i<1000 else 'extend1500')) for i,n in enumerate(names)])
    (folder/'cohort1500.json').write_text(json.dumps(cohort,indent=2))
    candidates={};base=repo/'sponge-sim/results/large_seed_20261106'
    for p in base.glob('*/*/pilot.json'):
        record=json.loads(p.read_text())
        for row in record['results']:
            if row.get('eligible'):
                candidates.setdefault(record['model'],[]).append(dict(model=record['model'],engine=row['engine'],batch=row['batch'],
                    fused=record.get('fused',False),seconds_per_image=row['seconds_per_image'],pilot_file=str(p.relative_to(repo)),
                    pilot_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    selected={model:min(rows,key=lambda r:(r['seconds_per_image'],r['batch'])) for model,rows in candidates.items()}
    assert set(selected)=={'yolov8n-seg.pt','yolo11n-seg.pt','yolo26n-seg.pt'}
    assert all(r['batch']==4 and not r['fused'] for r in selected.values())
    lock=dict(selected=selected,selection_rule='eligible correctness+memory, then processing seconds/image, then smaller batch',
        note='No attack K/effect selection; all fused candidates failed predeclared score/gradient agreement gate.')
    (folder/'execution_lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps(dict(n_images=1500,audit_images=len(audit),seeds=cohort['seeds'],execution=selected),indent=2))
