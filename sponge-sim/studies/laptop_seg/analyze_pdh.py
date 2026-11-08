"""Read registered PDH/process CPU diagnostic raw, no inference or timing."""
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

from scipy import stats
from pdh_cpu import NAMES


def describe(values):
    values=[v for v in values if v is not None]
    return dict(valid=len(values),minimum=None if not values else min(values),
                median=None if not values else statistics.median(values),maximum=None if not values else max(values))


def correlation(pairs):
    pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
    if len(pairs)<2 or len({a for a,b in pairs})<2 or len({b for a,b in pairs})<2:
        return dict(n=len(pairs),spearman=None)
    return dict(n=len(pairs),spearman=float(stats.spearmanr(*zip(*pairs)).statistic))


def analyze(rows):
    timings=[r for r in rows if r['kind']=='timing']
    out=dict(stages={},pdh_read_duration_ms=describe([r['pdh_after']['read_duration_ns']/1e6 for r in timings]))
    for stage in ('forward','total'):
        rs=[r for r in timings if r['stage']==stage]
        if not rs:
            continue
        counters={}
        for name in NAMES:
            cs=[r['pdh_after']['counters'][name] for r in rs]
            counters[name]=dict(**describe([c['value'] for c in cs]),
                                status_counts=dict(Counter(f'{c["return_code"]}/{c["status"]}' for c in cs)))
        perf=lambda r:r['pdh_after']['counters']['% Processor Performance']['value']
        out['stages'][stage]=dict(n=len(rs),counters=counters,
            wall_vs_reported_performance=correlation([(r['ns'],perf(r)) for r in rs]),
            cpu_vs_reported_performance=correlation([(r['process_cpu_ns'],perf(r)) for r in rs]),
            wall_vs_cpu=correlation([(r['ns'],r['process_cpu_ns']) for r in rs]))
    out['limits']='PDH intervals include metadata/warmup; two images; correlated repeats; no exact clock or causal inference.'
    return out


if __name__=='__main__':
    run=Path(sys.argv[1]);rows=[json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    result=analyze(rows)
    with (run/'pdh_summary.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
