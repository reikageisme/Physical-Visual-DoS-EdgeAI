"""M2 per-image paired ratios with A/A and quality gates."""
import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats


def analyze(run, exclude_flagged=False):
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    mf=json.loads((run/'manifest.json').read_text()); cfg=mf['config']
    refs=[r for r in rows if r['kind']=='reference']; pins=[r for r in rows if r['kind'] in ('pin','pin_end')]
    flagged_blocks={r['block'] for r in refs if r['flagged']}
    images=[i for ix,i in enumerate(mf['images']) if not exclude_flagged or ix not in flagged_blocks]
    cells=defaultdict(list); copies=defaultdict(list)
    for r in rows:
        if r['kind']=='cell' and r['image'] in images:
            cells[r['image'],r['k'],r['retina']].append(r)
            if r['k']==0: copies[r['image'],r['retina'],r['copy']].append(r['total_ns'])
    meds={k:{s:float(np.median([r[s] for r in rs])) for s in ('total_ns','mask_ns')} for k,rs in cells.items()}
    complete=(rows[-1]['kind']=='done' and len(refs)==cfg['n_images']+2 and all(len(cells[i,k,r])==cfg['repeats']*(2 if k==0 else 1)
              for i in images for k in cfg['K'] for r in cfg['retinas']))
    out=dict(config=cfg,complete=complete,n_images=len(images),exclude_flagged=exclude_flagged,
             raw_sha256=hashlib.sha256((run/'raw.jsonl').read_bytes()).hexdigest(),mde={},ratios={},budget={},k_star={})
    def bootstrap(x):
        rng=np.random.default_rng(20261106); x=np.asarray(x)
        return np.exp(np.quantile(x[rng.integers(0,len(x),(20000,len(x)))].mean(1),[.025,.975])).tolist()
    for retina in (False,True):
        logs=np.array([np.log(np.median(copies[i,retina,0])/np.median(copies[i,retina,1])) for i in images])
        mde=float((stats.t.ppf(.975,len(logs)-1)+stats.t.ppf(.8,len(logs)-1))*logs.std(ddof=1)/np.sqrt(len(logs)))
        out['mde'][str(retina)]=dict(log=mde,frac=float(np.expm1(mde)))
        star=None
        for k in (50,100,300):
            ds=[np.log(meds[i,k,retina]['total_ns']/meds[i,10,retina]['total_ns']) for i in images]
            ratio=float(np.exp(np.mean(ds))); ci=bootstrap(ds)
            out['ratios'][f'{retina}|{k}']=dict(ratio=ratio,ci95=ci,n_images=len(ds))
            if star is None and ci[0]>np.exp(mde): star=k
            pred=1+np.median([meds[i,k,retina]['mask_ns']-meds[i,10,retina]['mask_ns'] for i in images])/np.median([meds[i,10,retina]['total_ns'] for i in images])
            out['budget'][f'{retina}|{k}']=dict(predicted=float(pred),measured=ratio,absolute_log_error=float(abs(np.log(pred/ratio))))
        out['k_star'][str(retina)]=star
    out['flagged']=[r for r in refs if r['flagged']]
    out['pins']=pins
    out['quality_pass']=out['complete'] and len(out['flagged'])<=2 and all(r['verify']['passed'] for r in pins)
    if cfg['device']=='cpu':
        out['quality_pass']=out['quality_pass'] and len(pins)==2
    if not exclude_flagged:
        out['sensitivity_excluding_flagged']=analyze(run,True)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('run',type=Path); ap.add_argument('--out',default='summary.json'); a=ap.parse_args(); out=analyze(a.run)
    with (a.run/a.out).open('x') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
