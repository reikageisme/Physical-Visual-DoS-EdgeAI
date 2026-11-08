"""Registered frozen attack replay analysis; exclusive-create summary, no inference."""
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy import stats


STAGES=('forward','nms_tv','mask_native','total','aa_A','aa_B')


def analyze(run):
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    m=json.loads((run/'manifest.json').read_text());cfg=m['config']
    arms=tuple(cfg.get('arms',('clean','attacked')))
    primary=cfg.get('isolated_stage','total')
    calls=cfg.get('calls_per_trial',1)
    expected_stages=('aa_A','aa_B') if cfg.get('total_only',False) else STAGES
    groups=defaultdict(list)
    for r in rows:
        if r['kind']=='timing':groups[r['image'],r['arm'],r['stage']].append(r)
    ids=sorted({k[0] for k in groups})
    complete=bool(rows) and rows[-1]['kind']=='done' and len(ids)==cfg['n_images'] and all(
        sorted(r['rep'] for r in groups[i,a,s])==list(range(cfg['repeats']))
        for i in ids for a in arms for s in expected_stages)
    meds={k:float(np.median([r['ns'] for r in rs])) for k,rs in groups.items()}
    if cfg.get('total_only',False):
        for i in ids:
            for a in arms:
                pool=groups[i,a,'aa_A']+groups[i,a,'aa_B']
                if pool:meds[i,a,primary]=float(np.median([r['ns'] for r in pool]))/calls
    refs=[r for r in rows if r['kind']=='reference'];flags=sum(r['flagged'] for r in refs)
    pins=[r['verify']['passed'] for r in rows if r['kind'] in ('pin_start','pin_end')]
    aa={};contrasts={}
    for arm in arms:
        if not all((i,arm,s) in meds for i in ids for s in ('aa_A','aa_B')) or not ids:continue
        d=np.array([np.log(meds[i,arm,'aa_A']/meds[i,arm,'aa_B']) for i in ids]);n=len(d)
        mde=None if n<2 else float(np.expm1((stats.t.ppf(.975,n-1)+stats.t.ppf(.8,n-1))*d.std(ddof=1)/np.sqrt(n)))
        aa[arm]=dict(n_images=n,geometric_ratio=float(np.exp(d.mean())),mde_fraction=mde)
    for stage in ((primary,) if cfg.get('total_only',False) else STAGES[:4]):
        if not ids or not all((i,a,stage) in meds for i in ids for a in arms):continue
        clean=np.array([meds[i,arms[0],stage] for i in ids]);adv=np.array([meds[i,arms[1],stage] for i in ids])
        d=np.log(adv/clean);rng=np.random.default_rng(20261107)
        ix=rng.integers(0,len(ids),size=(20000,len(ids)))
        contrasts[stage]=dict(n_images=len(ids),geometric_ratio=float(np.exp(d.mean())),
            ci95=np.exp(np.quantile(d[ix].mean(axis=1),[.025,.975])).tolist(),
            image_ids=ids,**{arms[0]+'_medians_ms':(clean/1e6).tolist(),arms[1]+'_medians_ms':(adv/1e6).tolist()},
            mean_paired_delta_ms=float(((adv-clean)/1e6).mean()),
            delta_ci95=np.quantile(((adv-clean)/1e6)[ix].mean(axis=1),[.025,.975]).tolist())
    feasible=complete and len(refs)==cfg['n_images']+2 and flags<=cfg['max_reference_flags'] and pins==[True,True]
    quality=feasible and not cfg['smoke'] and all(aa.get(a,{}).get('mde_fraction') is not None
              and aa[a]['mde_fraction']<=cfg['max_mde_fraction'] for a in arms)
    result=dict(smoke=cfg['smoke'],complete=complete,feasibility_pass=feasible,quality_pass=quality,
                flags=flags,n_references=len(refs),pin_checks=pins,aa=aa,contrasts=contrasts,
                counts=[r for r in rows if r['kind']=='counts'],references=refs,
                raw_sha256=hashlib.sha256((run/'raw.jsonl').read_bytes()).hexdigest(),
                scope=('Prepared CPU stage control; per-call estimate from fixed call count; no pipeline/service/additive claim.'
                       if cfg.get('isolated_stage') else
                       'Frozen whole-image PGD20; defended/baseline on SAME attacked tensor; no clean-accuracy/DoS claim.'
                       if cfg.get('defense_comparison') else
                       'Frozen whole-image PGD20; tensor-to-native-mask CPU latency; no DoS/deadline/camera claim.'))
    if cfg.get('isolated_stage'):
        result['endpoint']=dict(name=primary,calls_per_trial=calls,primary_units='ms per call',
                                raw_ns_units='whole block',cpu_pdh_diagnostics_units='raw whole blocks')
        result['stage_input_checks']=[r for r in rows if r['kind']=='stage_input_check']
        checks=result['stage_input_checks']
        valid=(len(checks)==2*cfg['n_images'] and {(r['image'],r['arm']) for r in checks}=={(i,a) for i in ids for a in arms}
               and all(r['unchanged'] for r in checks))
        result['feasibility_pass'] &= valid
        result['quality_pass'] &= valid
    return result


if __name__=='__main__':
    run=Path(sys.argv[1]);r=analyze(run)
    from analyze_cpu import summarize
    from analyze_pdh import analyze as pdh_analyze
    rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    r['cpu_diagnostics']={};r['pdh_diagnostics']={}
    manifest=json.loads((run/'manifest.json').read_text())
    for arm in manifest['config'].get('arms',('clean','attacked')):
        selected=[x for x in rows if x['kind']!='timing' or x.get('arm')==arm]
        if manifest['config'].get('total_only',False):
            selected=[dict(x,stage=manifest['config'].get('isolated_stage','total'),label=x['stage']) if x['kind']=='timing' else x for x in selected]
        r['cpu_diagnostics'][arm]=summarize(selected)
        r['pdh_diagnostics'][arm]=pdh_analyze(selected)
    name=('stage_summary.json' if manifest['config'].get('isolated_stage') else
          'defense_summary.json' if manifest['config'].get('defense_comparison') else 'attack_summary.json')
    with (run/name).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps({k:r[k] for k in ['smoke','complete','feasibility_pass','quality_pass','flags','aa','contrasts']},indent=2))
