"""POST-HOC label-allocation diagnostic on saved identical-call data.

Not a randomized test: original labels were deterministic. Never overwrites
raw, relabels accepted results, or changes old gates. No new inference/timings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]/'results/laptop_total_confirmation_20261107/full_01'


def main():
    p=ROOT/'P1_cpu2';raw=p/'raw.jsonl'
    rows=[json.loads(s) for s in raw.read_text().splitlines()]
    original=json.loads((p/'attack_summary.json').read_text())
    rng=np.random.default_rng(20261109);draws=2000;out={}
    for arm in ('clean','attacked'):
        ids=sorted({r['image'] for r in rows if r['kind']=='timing' and r['arm']==arm})
        simulated=np.empty((draws,len(ids)));observed=[];position=[]
        for j,image in enumerate(ids):
            rs=[r for r in rows if r['kind']=='timing' and r['arm']==arm and r['image']==image]
            first=[];second=[];strata=[]
            for rep in range(48):
                pair=sorted([r for r in rs if r['rep']==rep],key=lambda r:r['position'])
                assert len(pair)==2 and pair[0]['position']//2==pair[1]['position']//2
                first.append(pair[0]['ns']);second.append(pair[1]['ns']);strata.append(pair[0]['position']//2)
            first=np.array(first);second=np.array(second);strata=np.array(strata)
            a=[r['ns'] for r in rs if r['stage']=='aa_A'];b=[r['ns'] for r in rs if r['stage']=='aa_B']
            observed.append(np.log(np.median(a)/np.median(b)))
            position.append(dict(image=image,first_over_second=float(np.median(first)/np.median(second))))
            take_first=np.zeros((draws,48),dtype=bool)
            for stratum in (0,1):
                index=np.flatnonzero(strata==stratum);assert len(index)==24
                order=np.argsort(rng.random((draws,24)),axis=1)[:,:12]
                take_first[np.arange(draws)[:,None],index[order]]=True
            av=np.where(take_first,first,second);bv=np.where(take_first,second,first)
            simulated[:,j]=np.log(np.median(av,axis=1)/np.median(bv,axis=1))
        factor=(stats.t.ppf(.975,19)+stats.t.ppf(.8,19))/np.sqrt(20)
        actual=float(np.expm1(factor*np.std(observed,ddof=1)))
        assert np.isclose(actual,original['aa'][arm]['mde_fraction'],rtol=1e-12,atol=1e-14)
        distribution=np.expm1(factor*simulated.std(axis=1,ddof=1))
        out[arm]=dict(observed_mde_fraction=actual,registered_mde_fraction=original['aa'][arm]['mde_fraction'],
                      counterfactual_mde_q05_q50_q95=np.quantile(distribution,[.05,.5,.95]).tolist(),
                      fraction_counterfactual_below_observed=float((distribution<actual).mean()),
                      per_image_call_position_ratio=position)
    result=dict(status='POSTHOC_CONDITIONAL_LABEL_DIAGNOSTIC_NOT_RANDOMIZATION_TEST',
                seed=20261109,draws=draws,raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),arms=out,
                limits='Reassigned labels are hypothetical, not new data; no gate/P-value/population/causal claim; old FAIL unchanged.')
    with (ROOT/'posthoc_label_diagnostic.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({a:{k:v for k,v in o.items() if k!='per_image_call_position_ratio'} for a,o in out.items()},indent=2))


if __name__=='__main__':main()
