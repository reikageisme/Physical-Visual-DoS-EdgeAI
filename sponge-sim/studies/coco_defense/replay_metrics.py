"""Recompute saved full-validation metrics from detection-level arrays, no inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import SegmentMetrics


if __name__=='__main__':
    root=Path('C:/sponge/coco_defense_full_20261106')
    records=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
    checks=[]
    for r in records:
        p=root/f"{r['model']}_{r['variant']}"/'metric_inputs.npz'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['metric_inputs_sha256']
        data=np.load(p,allow_pickle=False)
        metrics=SegmentMetrics()
        metrics.stats={k:[data[k]] for k in data.files}
        metrics.process()
        actual=metrics.results_dict
        errors={k:abs(float(actual[k])-float(v)) for k,v in r['results'].items()}
        assert max(errors.values())<1e-12,errors
        checks.append(dict(model=r['model'],variant=r['variant'],max_abs_error=max(errors.values()),
                           shapes={k:list(data[k].shape) for k in data.files}))
    assert len(checks)==8
    with (root/'metric_replay_receipt.json').open('x') as f:json.dump(checks,f,indent=2)
    print('METRIC_REPLAY_PASS',len(checks),flush=True)
