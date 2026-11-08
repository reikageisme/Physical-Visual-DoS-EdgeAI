"""Saved P2 full-pipeline analysis; same frozen primary estimator and gates."""
import json
from pathlib import Path
import sys

from analyze_attack import analyze
from analyze_cpu import summarize
from analyze_pdh import analyze as pdh_analyze


def main():
    path = Path(sys.argv[1])
    m = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if m['cpu_set'] != [0, 2] or m['torch_threads'] != 2 or m['interop_threads'] != 1:
        raise ValueError('registered P2 runtime required')
    result = analyze(path)
    rows = [json.loads(x) for x in (path/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    result['cpu_diagnostics'] = {}
    result['pdh_diagnostics'] = {}
    for arm in ('clean', 'attacked'):
        selected = [dict(r, stage='total', label=r['stage']) if r['kind'] == 'timing' else r
                    for r in rows if r['kind'] != 'timing' or r['arm'] == arm]
        result['cpu_diagnostics'][arm] = summarize(selected)
        diagnostic = pdh_analyze(selected)
        diagnostic['limits'] = ('Processor Information(0,0) only; not the whole two-core set. '
                                'Intervals include observer/preparation/warmup; correlated repeats; '
                                'no instantaneous clock or causal inference.')
        result['pdh_diagnostics'][arm] = diagnostic
    result['profile_scope'] = dict(cpu_set=[0, 2], physical_p_cores=2, intra_op=2, inter_op=1,
                                   pdh_counter_cpu=0, process_cpu_over_wall_units='effective CPU cores, not fraction of two-core capacity',
                                   historical_single_core_comparison='descriptive only; different session/app exclusion and possible numerical outputs',
                                   inference='within-P2 attacked/clean ratio on same fixed known inputs; no DoS/ARM/tail claim')
    with (path/'attack_summary.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('complete', 'smoke', 'feasibility_pass', 'quality_pass', 'flags', 'aa', 'contrasts')}, indent=2))


if __name__ == '__main__':
    main()
