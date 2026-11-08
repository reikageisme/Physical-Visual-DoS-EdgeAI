"""Post-hoc component diagnostics from saved raw only; never promote a gate.

The frozen legacy PDH aggregator recognizes forward/total, so component
summaries have empty PDH stage tables. This separate report preserves those
original summaries and exposes their already-recorded counters. No timing,
model invocation, counter collection or old-artifact replacement occurs.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def describe(values):
    values = list(values)
    return dict(n=len(values), minimum=min(values) if values else None,
                median=statistics.median(values) if values else None,
                maximum=max(values) if values else None)


def summarize_rows(rows, endpoint, calls):
    if (endpoint, calls) not in (('nms_kernel', 64), ('mask_native', 1)) or type(calls) is not int:
        raise ValueError('known registered component/call count required')
    timings = [(line, r) for line, r in enumerate(rows, 1) if r['kind'] == 'timing']
    if not timings:
        raise ValueError('no saved timings')
    arms = {}
    for arm in sorted({r['arm'] for _, r in timings}):
        selected = [(line, r) for line, r in timings if r['arm'] == arm]
        counters = {}
        for name in sorted({name for _, r in selected for name in r['pdh_after']['counters']}):
            cs = [r['pdh_after']['counters'].get(name) for _, r in selected]
            valid = [c['value'] for c in cs if c is not None and c['return_code'] == 0
                     and c['status'] in (0, 1) and c['value'] is not None and math.isfinite(c['value'])]
            counters[name] = dict(values=describe(valid), missing_or_invalid=len(cs)-len(valid),
                                  statuses=dict(Counter('missing' if c is None else
                                                f"{c['return_code']}/{c['status']}" for c in cs)))
        intervals = []
        for line, r in selected:
            p = r['pdh_after']
            begin, end = p['interval_start_ns'], p['collected_ns']
            if begin is not None and end > begin:
                intervals.append(dict(raw_line=line, interval_ms=(end-begin)/1e6,
                                      timer_fraction=r['ns']/(end-begin),
                                      timer_contained=begin <= r['t_start_ns'] and r['t_start_ns']+r['ns'] <= end))
        fractions = [r['process_cpu_ns']/r['ns'] for _, r in selected]
        arms[arm] = dict(
            raw_timing_lines=[line for line, _ in selected],
            raw_block_wall_ms=describe(r['ns']/1e6 for _, r in selected),
            pooled_block_average_per_call_ms=describe(r['ns']/calls/1e6 for _, r in selected),
            process_cpu_ns_observed_values=sorted({r['process_cpu_ns'] for _, r in selected}),
            zero_process_cpu_count=sum(r['process_cpu_ns'] == 0 for _, r in selected),
            cpu_over_wall=describe(fractions), cpu_over_wall_above_one_count=sum(v > 1 for v in fractions),
            pdh_counters=counters,
            pdh_read_ms=describe(r['pdh_after']['read_duration_ns']/1e6 for _, r in selected),
            pdh_read_over_raw_block_wall=describe(r['pdh_after']['read_duration_ns']/r['ns'] for _, r in selected),
            pdh_intervals=intervals,
        )
    references = []
    for line, r in enumerate(rows, 1):
        if r['kind'] != 'reference':
            continue
        references.append(dict(raw_line=line, block=r['block'], flagged=r['flagged'], ratio=r['ratio'],
                               wall_median_ms=statistics.median(r['samples_ns'])/1e6,
                               cpu_median_ms=statistics.median(r['cpu_samples_ns'])/1e6,
                               paired_cpu_over_wall=describe(c/w for c, w in zip(r['cpu_samples_ns'], r['samples_ns']))))
    return dict(endpoint=endpoint, calls_per_block=calls, arms=arms, references=references,
                flagged_faster=sum(r['flagged'] and r['ratio'] < 1 for r in references),
                flagged_slower=sum(r['flagged'] and r['ratio'] > 1 for r in references),
                limits=[
                    'Post-hoc descriptive analysis; frozen quality decisions are unchanged.',
                    'Pooled block summaries are not the registered image-unit primary estimator.',
                    'Per-call values divide a whole cached block; no individual-call latency distribution is observed.',
                    'Zero or greater-than-one processCPU/wall values are retained; short-block CPU quantization prevents off-CPU attribution.',
                    'PDH intervals include between-timer work and may include references, preparation and warmup. They are not exact timer-local clocks.',
                    'PDH read/wall ratios describe observer cadence, not an empty-loop correction or measured causal overhead.',
                    'No temperature, scheduler trace or uniquely identified cause is available.',
                ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.run/'raw.jsonl'
    summary = args.run/'stage_summary.json'
    data = raw.read_bytes()
    old = json.loads(summary.read_text())
    digest = hashlib.sha256(data).hexdigest()
    if digest != old['raw_sha256']:
        raise ValueError('saved raw does not match frozen summary')
    result = summarize_rows([json.loads(x) for x in data.splitlines()], old['endpoint']['name'],
                            old['endpoint']['calls_per_trial'])
    assert sum(r['flagged'] for r in result['references']) == old['flags']
    result.update(status='EXPLORATORY_SAVED_RAW_NO_NEW_MEASUREMENT',
                  raw_path=raw.as_posix(), raw_sha256=digest,
                  frozen_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frozen_quality_pass=old['quality_pass'], frozen_feasibility_pass=old['feasibility_pass'],
                  legacy_pdh_stage_keys={a: list(v['stages']) for a, v in old['pdh_diagnostics'].items()})
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'endpoint', 'frozen_quality_pass', 'flagged_faster', 'flagged_slower')}))


if __name__ == '__main__':
    main()
