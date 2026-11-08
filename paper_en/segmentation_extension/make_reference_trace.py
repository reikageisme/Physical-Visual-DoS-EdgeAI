"""Export the closed forward-only diagnostic, without a cadence-effect claim."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREFIX = Path('sponge-sim/results/laptop_reference_trace_20261107/run_01')


def main():
    s = json.loads((ROOT/PREFIX/'analysis.json').read_text())
    assert s['status'] == 'COMPLETE_DESCRIPTIVE_DIAGNOSTIC_NO_ATTACK_QUALIFICATION'
    a, b = (s['conditions'][c] for c in ('continuous', 'pause2'))
    assert (a['timing_count'], b['timing_count']) == (180, 180)
    assert (a['diagnostic_flags'], b['diagnostic_flags']) == (4, 2)
    sources = {(Path(n).relative_to(ROOT).as_posix() if Path(n).is_absolute() else n): meta
               for n, meta in s['sources'].items()}
    source = dict(sources=sources, selectors=['conditions.*.blocks', 'conditions.*.diagnostic_flags'],
                  interpretation='Descriptive traces only, one process per cadence, fixed order, no attack qualification.')
    for name in ('analysis.json', 'reference_trace.png', 'reference_trace.pdf'):
        p = PREFIX/name
        source['sources'][p.as_posix()] = dict(sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
    for name, meta in source['sources'].items():
        path = Path(name)
        assert hashlib.sha256((path if path.is_absolute() else ROOT/path).read_bytes()).hexdigest() == meta['sha256']
    helper = Path('sponge-sim/studies/laptop_seg/analyze_stage_drift.py')
    source['sources'][helper.as_posix()] = dict(sha256=hashlib.sha256((ROOT/helper).read_bytes()).hexdigest())
    source['sources'][Path(__file__).relative_to(ROOT).as_posix()] = dict(
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for suffix in ('png', 'pdf'):
        with (OUT/f'laptop_reference_trace.{suffix}').open('xb') as f:
            f.write((ROOT/PREFIX/f'reference_trace.{suffix}').read_bytes())
    lines = [
        r'% Generated from saved forward-only traces by make_reference_trace.py.',
        r'\subsection{Forward-reference diagnostic without NMS or masks}',
        r'\label{app:seg-forward-reference}',
        r'A subsequent prospective diagnostic records the forward-reference workload alone on the same CPU, model and one fixed clean image. Two fresh processes each execute 12 blocks of three fixed warmup forwards and 15 timed forwards: one continuous cadence and one with a two-second pause between blocks. The registered order is continuous first. PDH is recorded after every timed forward, and each output is released after both clocks. All 360 samples complete, pin-load checks pass, and all 45 application affinities are restored before analysis.',
        r'The continuous trace flags 4/12 block medians and the paused trace flags 2/12 against their own first-block median using the descriptive 3\% threshold; every flagged block is faster. Thus reference drift exceeding this threshold is observed even without NMS or mask work. This does not uniquely identify the mechanism of the earlier component failures. There is one process per cadence and the fixed order confounds cadence with session/time; the flag counts are not evidence that pausing improves stability. No CI, p-value, A/A MDE or attack qualification is inferred, no baseline is replaced, and the finite protocol stops without retry.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.98\textwidth]{segmentation_extension/laptop_reference_trace.pdf}\caption{Forward-only diagnostic: block wall-median ratios, median paired process-CPU/wall values, and median reported processor-performance counter. Shading marks the descriptive 3\% band. Interval counters and quantized process CPU do not identify thermal, clock, cache or scheduling causes.}\label{fig:seg-forward-reference}\end{figure*}',
        r'The initial runtime, pin-load and cache context differs from the component workload. These controls retain all observations, supply no ARM or service-denial result, and do not promote either previously failed component contrast.',
    ]
    with (OUT/'laptop_reference_trace.tex').open('x', encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    with (OUT/'laptop_reference_trace_sources.json').open('x', encoding='utf-8') as f:json.dump(source, f, indent=2)
    print('Forward-only supplement exported; no attack/cadence qualification.')


if __name__ == '__main__':
    main()
