"""Image-level counts, bootstrap median intervals and Wilson reach intervals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm


def summary(clean,adv):
    clean=np.asarray(clean);adv=np.asarray(adv);n=len(adv)
    rng=np.random.default_rng(20261106);med=[];gain=[]
    for _ in range(20):
        draws=rng.integers(0,n,(1000,n));med.extend(np.median(adv[draws],1));gain.extend(np.median((adv-clean)[draws],1))
    z=float(norm.ppf(.975));reach={}
    for k in [50,100,300]:
        hits=int((adv>=k).sum());p=hits/n;d=1+z*z/n
        center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        reach[str(k)]=dict(count=hits,fraction=p,wilson_ci95=[max(0,float(center-half)),min(1,float(center+half))])
    return dict(n_images=n,clean_median=float(np.median(clean)),attacked_median=float(np.median(adv)),
                attacked_median_ci95=np.quantile(med,[.025,.975]).tolist(),median_gain=float(np.median(adv-clean)),
                median_gain_ci95=np.quantile(gain,[.025,.975]).tolist(),reach=reach)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    rows=[json.loads(s) for s in (a.run/'raw.jsonl').read_text().splitlines()]
    mf=json.loads((a.run/'manifest.json').read_text());n=mf['config']['n_images']
    images=[r['file_name'] for r in mf['cohort']['images'][:n]]
    counts={(r['model'],r['image'],r['arm'],r['variant']):r for r in rows if r['kind']=='counts'}
    target={(r['image'],r['arm']):r for r in rows if r['kind']=='native26'}
    expected=n*2*2*4
    complete=(rows[-1]['kind']=='done' and len(counts)==expected and len(target)==n*2
              and sum(r['kind']=='counts' for r in rows)==expected and sum(r['kind']=='native26' for r in rows)==n*2)
    assert complete,'incomplete count coverage'
    out=dict(complete=complete,smoke=mf['config']['smoke'],n_images=n,models={},native26=None,
             raw_sha256=hashlib.sha256((a.run/'raw.jsonl').read_bytes()).hexdigest())
    for model in mf['config']['models']:
        variants={}
        for variant,maxdet,cap in mf['config']['variants']:
            clean=[counts[model,i,'clean',variant]['k'] for i in images]
            adv=[counts[model,i,'attacked',variant]['k'] for i in images]
            variants[variant]=summary(clean,adv)
            variants[variant]['attacked_kernel_n_range']=[min(counts[model,i,'attacked',variant]['kernel_n'] for i in images),
                                                         max(counts[model,i,'attacked',variant]['kernel_n'] for i in images)]
        out['models'][model]=variants
    out['native26']=summary([target[i,'clean']['k'] for i in images],[target[i,'attacked']['k'] for i in images])
    with (a.run/'summary.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(dict(n_images=n,complete=complete,baseline={m:v['baseline'] for m,v in out['models'].items()},native26=out['native26']),indent=2))
