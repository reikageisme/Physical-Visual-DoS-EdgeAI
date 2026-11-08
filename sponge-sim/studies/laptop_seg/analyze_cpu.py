"""Registered HighQoS diagnostic endpoints; saved raw only, no inference."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['kind']=='timing':
            if not isinstance(r.get('process_cpu_ns'),int) or r['process_cpu_ns']<0 or r['ns']<=0:
                raise ValueError('missing/invalid CPU or wall interval')
            groups[r['image'],r['stage']].append(r)
    stages={}
    for stage in sorted({k[1] for k in groups}):
        trials=[];spikes=[]
        for (image,s),rs in groups.items():
            if s!=stage:
                continue
            median=statistics.median(r['ns'] for r in rs)
            for r in rs:
                fraction=r['process_cpu_ns']/r['ns']
                row=dict(image=image,rep=r['rep'],label=r['label'],wall_ns=r['ns'],
                         process_cpu_ns=r['process_cpu_ns'],cpu_over_wall=fraction)
                trials.append(row)
                if r['ns']>1.5*median:
                    spikes.append(row)
        stages[stage]=dict(n_trials=len(trials),wall_median_ms=statistics.median(r['wall_ns'] for r in trials)/1e6,
                           wall_max_ms=max(r['wall_ns'] for r in trials)/1e6,
                           median_cpu_over_wall=statistics.median(r['cpu_over_wall'] for r in trials),
                           n_highwall=len(spikes),highwall_fraction=len(spikes)/len(trials),
                           highwall_low_cpu_count=sum(r['cpu_over_wall']<.75 for r in spikes),
                           highwall_low_cpu_share=None if not spikes else sum(r['cpu_over_wall']<.75 for r in spikes)/len(spikes),
                           highwall_trials=spikes)
    refs=[]
    for line,r in enumerate(rows,1):
        if r['kind']=='reference':
            if len(r.get('cpu_samples_ns',[]))!=len(r['samples_ns']):
                raise ValueError('reference CPU coverage missing')
            fractions=[c/w for c,w in zip(r['cpu_samples_ns'],r['samples_ns'])]
            refs.append(dict(raw_line=line,block=r['block'],wall_ratio=r['ratio'],
                             median_cpu_over_wall=statistics.median(fractions)))
    return dict(stages=stages,references=refs,
                interpretation='Descriptive process CPU-time / wall-time; no CPU clock or scheduler trace; no clipping.')


if __name__=='__main__':
    root=Path(sys.argv[1])
    rows=[json.loads(s) for s in (root/'raw.jsonl').read_text().splitlines()]
    result=summarize(rows)
    with (root/'cpu_summary.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='highwall_trials'} for k,v in result['stages'].items()},indent=2))
