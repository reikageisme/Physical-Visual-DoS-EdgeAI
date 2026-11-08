"""Read matched-load controls, retaining every conditioning setting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args()
    spec=importlib.util.spec_from_file_location('cal_analysis',Path(__file__).parents[1]/'stability_calibration/analyze.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    results={};manifests={}
    for name in ['w5','w50','w200']:
        path=a.root/name/'raw.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
        results[name]=mod.summarize(rows);results[name]['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        results[name]['telemetry']=[dict(block=r['block'],after_load=r['after_load'],after_warmup=r['after_warmup'],
                                        after_reference=r['after_reference']) for r in rows if r['kind']=='reference']
        manifests[name]=json.loads((a.root/name/'manifest.json').read_text())
    eligible=[n for n,r in results.items() if r['eligible']]
    selected=min(eligible,key=lambda n:(results[n]['flags'],results[n]['mde_frac'],int(n[1:]))) if eligible else None
    out=dict(profiles=results,selected=selected,affinity=manifests['w5']['config']['affinity'],
             forward_warmup=None if selected is None else int(selected[1:]),condition_cycles=10,
             note='Controls-only selection. One bounded diagnostic campaign after M6 failure; not an effect-size selection.')
    with (a.root/'selection.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(selected=selected,forward_warmup=out['forward_warmup'],
        profiles={n:{k:r[k] for k in ['flags','mde_frac','eligible','median_wall_ms','median_device_ms']} for n,r in results.items()}),indent=2))
