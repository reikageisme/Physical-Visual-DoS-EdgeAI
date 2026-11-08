"""Choose a host profile using clean controls only; no attack-effect selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import stats


def summarize(rows):
    references=[r for r in rows if r['kind']=='reference']
    aa=[r for r in rows if r['kind']=='aa']
    complete=(rows[-1]['kind']=='done' and len(references)==22 and len(aa)==20
              and {r['block'] for r in aa}==set(range(20))
              and all(len(r['samples_ns'])==31 for r in references)
              and all(len(r['samples'])==62 and len({(s['label'],s['rep']) for s in r['samples']})==62
                      and all(sum(s['label']==label for s in r['samples'])==31 for label in ['A','B']) for r in aa))
    logs=np.array([np.log(np.median([s['ns'] for s in r['samples'] if s['label']=='A'])/
                         np.median([s['ns'] for s in r['samples'] if s['label']=='B'])) for r in aa])
    mde=float(np.expm1((stats.t.ppf(.975,len(logs)-1)+stats.t.ppf(.8,len(logs)-1))*logs.std(ddof=1)/np.sqrt(len(logs))))
    flags=sum(r['flagged'] for r in references)
    return dict(complete=complete,flags=flags,mde_frac=mde,eligible=bool(complete and flags<=2 and mde<=.01),
                reference_ratios=[r['ratio'] for r in references],
                median_wall_ms=float(np.median([s['ns']/1e6 for r in aa for s in r['samples']])),
                median_device_ms=float(np.median([s for r in aa for s in r['device_forward_ms']])))


def choose(profiles):
    priority={'four':0,'all':1,'one':2}
    eligible=[name for name,r in profiles.items() if r['eligible']]
    return min(eligible,key=lambda n:(profiles[n]['flags'],profiles[n]['mde_frac'],priority[n])) if eligible else None


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_dir',type=Path);a=ap.parse_args()
    profiles={};manifests={}
    for name in ['all','one','four']:
        p=a.run_dir/name/'raw.jsonl';profiles[name]=summarize([json.loads(s) for s in p.read_text().splitlines()])
        profiles[name]['raw_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        manifests[name]=json.loads((a.run_dir/name/'manifest.json').read_text())
    selected=choose(profiles)
    result=dict(profiles=profiles,selected=selected,affinity=None if selected is None else manifests[selected]['config']['affinity'])
    with (a.run_dir/'selection.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
