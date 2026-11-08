"""Summarize final-iterate source and native target counts, per image bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(run):
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    mf=json.loads((run/'manifest.json').read_text()); attacks=[r for r in rows if r['kind']=='attack']
    out=dict(smoke=mf['smoke'],complete=rows[-1]['kind']=='done' and len(attacks)==len(mf['images']),n_images=len(attacks),models={})
    for key in ('k_v8','k26_native'):
        clean=np.array([r['clean'][key] for r in attacks]); adv=np.array([r['attacked'][key] for r in attacks])
        rng=np.random.default_rng(20261106); draws=rng.integers(0,len(adv),(20000,len(adv)))
        out['models'][key]=dict(clean=clean.tolist(),attacked=adv.tolist(),clean_median=float(np.median(clean)),
            attacked_median=float(np.median(adv)),attacked_median_ci95=np.quantile(np.median(adv[draws],axis=1),[.025,.975]).tolist(),
            median_gain=float(np.median(adv-clean)),reach50=int((adv>=50).sum()),reach100=int((adv>=100).sum()),reach300=int((adv>=300).sum()))
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();out=analyze(a.run)
    with (a.run/'summary.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
