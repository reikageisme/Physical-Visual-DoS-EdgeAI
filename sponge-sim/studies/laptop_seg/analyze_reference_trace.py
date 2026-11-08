"""Verify and describe the finite reference-only traces; no gate promotion."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from analyze_stage_drift import describe
from reference_trace import validate
from run import REPO, load_lock, sha
from windows_qos import high_readback_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--lock', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    cfglock = load_lock(args.lock)
    cfg = cfglock['config']
    order = validate(cfg)
    campaign = json.loads((root/'campaign.json').read_text())
    status = json.loads((root/'campaign_done.json').read_text())
    assert campaign['order'] == order == cfglock['condition_order']
    assert campaign['lock_sha256'] == sha(args.lock)
    assert status == [dict(condition=c, returncode=0) for c in order]
    assert subprocess.check_output(['git', 'rev-parse', campaign['tag']+'^{commit}'],
                                   text=True, cwd=REPO).strip() == campaign['commit']
    differences = []
    for path in cfglock['source_sha256']:
        data = (REPO/path).read_bytes()
        tagged = subprocess.check_output(['git', 'show', campaign['tag']+':'+path], cwd=REPO)
        if data != tagged:
            assert path == 'sponge-sim/src/sponge_sim/hw.py'
            assert data.replace(b'\r\n', b'\n') == tagged.replace(b'\r\n', b'\n')
            differences.append(dict(path=path, working_sha256=sha(REPO/path),
                                    tag_sha256=hashlib.sha256(tagged).hexdigest(), reason='known CRLF/LF only'))
    journal = root.with_name(root.name+'_affinity.jsonl')
    events = [json.loads(x) for x in journal.read_text().splitlines()]
    assert any(x['kind'] == 'ready' for x in events)
    assert not any(x['kind'].startswith('restore_') for x in events)
    assert all(x['readback'] == x['expected'] for x in events if x['kind'] == 'restored')
    assert {x['pid'] for x in events if x['kind'] == 'applied'} <= {
        x['pid'] for x in events if x['kind'] in ('restored', 'exited')}
    result = dict(status='COMPLETE_DESCRIPTIVE_DIAGNOSTIC_NO_ATTACK_QUALIFICATION',
                  tag=campaign['tag'], prereg_commit=campaign['commit'],
                  conditions={}, restoration=dict(Counter(e['kind'] for e in events)),
                  tag_line_ending_differences=differences, sources={},
                  analysis_code_written_after_trace=True,
                  limits=['One process per cadence; fixed order confounded with session/time. No cadence treatment-effect inference.',
                          'No CI, p-value, A/A MDE, attack gate, baseline substitution or historical FAIL promotion.',
                          'PDH is an interval counter, CPU time is quantized, and no temperature/scheduler trace exists.',
                          'No NMS/MASK in workload; pin-load and runtime initialization differ from historical component context.'])
    paths = [args.lock, root/'campaign.json', root/'campaign_done.json', journal,
             Path(__file__), REPO/'sponge-sim/results/laptop_reference_trace_20261107/registration_check.json']
    for condition in order:
        base = root/condition
        manifest = json.loads((base/'manifest.json').read_text())
        assert manifest['condition'] == condition and manifest['config'] == cfg
        assert manifest['lock_sha256'] == sha(args.lock)
        assert manifest['tag'] == campaign['tag'] and manifest['prereg_commit'] == campaign['commit']
        assert manifest['versions'] == cfglock['versions']
        assert manifest['cpu_set'] == [2] and manifest['torch_threads'] == manifest['interop_threads'] == 1
        assert manifest['power_start']['power_plugged'] is True
        assert manifest['power_mode_start'] == cfglock['power_mode']
        assert manifest['priority_start'] == 32768 and high_readback_ok(manifest['qos_change']['after'])
        rows = [json.loads(x) for x in (base/'raw.jsonl').read_text().splitlines()]
        expected = ['pin_start'] + (['pdh_block_initial']+['timing']*15+['reference'])*12 + ['pin_end', 'done']
        assert [r['kind'] for r in rows] == expected
        assert all(r['verify']['passed'] and r['affinity'] == [2] for r in rows if r['kind'].startswith('pin_'))
        done = rows[-1]
        assert done['affinity'] == [2] and done['power_end']['power_plugged'] is True
        assert done['power_mode_end'] == cfglock['power_mode'] and done['priority_end'] == 32768
        assert high_readback_ok(done['qos_end'])
        timings = [(i, r) for i, r in enumerate(rows, 1) if r['kind'] == 'timing']
        assert [(r['block'], r['sample']) for _, r in timings] == [(b, s) for b in range(12) for s in range(15)]
        for (_, a), (_, b) in zip(timings, timings[1:]):
            assert a['t_start_ns']+a['ns'] <= b['t_start_ns']
        blocks = []
        for block in range(12):
            selected = [(i, r) for i, r in timings if r['block'] == block]
            line, reference = next((i, r) for i, r in enumerate(rows, 1)
                                   if r['kind'] == 'reference' and r['block'] == block)
            median = statistics.median(r['ns'] for _, r in selected)
            ratio = median/(blocks[0]['wall_median_ms']*1e6) if blocks else 1.0
            assert median == reference['median_ns'] and abs(ratio-reference['initial_ratio']) < 1e-14
            assert (abs(ratio-1) > .03) == reference['diagnostic_flag_3pct']
            counters = {}
            for name in selected[0][1]['pdh_after']['counters']:
                cs = [r['pdh_after']['counters'][name] for _, r in selected]
                valid = [c['value'] for c in cs if c['return_code'] == 0 and c['status'] in (0, 1) and c['value'] is not None]
                counters[name] = dict(values=describe(valid), statuses=dict(Counter(
                    f"{c['return_code']}/{c['status']}" for c in cs)), invalid=len(cs)-len(valid))
            blocks.append(dict(block=block, raw_timing_lines=[i for i, _ in selected], raw_reference_line=line,
                               wall_median_ms=median/1e6, wall_ms=describe(r['ns']/1e6 for _, r in selected),
                               initial_ratio=ratio, diagnostic_flag_3pct=reference['diagnostic_flag_3pct'],
                               process_cpu_median_ms=statistics.median(r['process_cpu_ns'] for _, r in selected)/1e6,
                               paired_cpu_over_wall=describe(r['process_cpu_ns']/r['ns'] for _, r in selected),
                               pdh_counters=counters))
        result['conditions'][condition] = dict(blocks=blocks, timing_count=len(timings),
            diagnostic_flags=sum(b['diagnostic_flag_3pct'] for b in blocks),
            flagged_faster=sum(b['diagnostic_flag_3pct'] and b['initial_ratio'] < 1 for b in blocks),
            flagged_slower=sum(b['diagnostic_flag_3pct'] and b['initial_ratio'] > 1 for b in blocks),
            raw_wall_ms=describe(r['ns']/1e6 for _, r in timings))
        paths.extend([base/'manifest.json', base/'raw.jsonl'])
    for path in paths:
        result['sources'][path.as_posix()] = dict(sha256=sha(path))
    with (root/'analysis.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({c: {k: v for k, v in r.items() if k not in ('blocks', 'raw_wall_ms')}
                      for c, r in result['conditions'].items()}, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for condition, c in result['conditions'].items():
        blocks = c['blocks']
        x = [b['block'] for b in blocks]
        axes[0].plot(x, [b['initial_ratio'] for b in blocks], 'o-', label=condition)
        axes[1].plot(x, [b['paired_cpu_over_wall']['median'] for b in blocks], 'o-', label=condition)
        axes[2].plot(x, [b['pdh_counters']['% Processor Performance']['values']['median'] for b in blocks], 'o-', label=condition)
    axes[0].axhspan(.97, 1.03, color='gray', alpha=.12)
    axes[0].axhline(1, color='gray', linewidth=.7)
    axes[0].set(ylim=(.94, 1.04), ylabel='Forward block median / first block')
    axes[1].set(ylim=(.94, 1.04), ylabel='Median paired process CPU / wall')
    axes[2].set(ylim=(240, 255), ylabel='Median reported processor performance (%)')
    for ax in axes:
        ax.set(xlabel='Fixed block index')
        ax.legend(fontsize=8)
    fig.suptitle('Forward only: one fresh process per cadence; descriptive, no attack qualification', fontsize=10)
    for suffix in ('png', 'pdf'):
        path = root/f'reference_trace.{suffix}'
        assert not path.exists()
        fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
