"""Offline P2 source/schedule/environment audit; no new model invocation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

from analyze_attack import analyze
from attack_replay import carryover_orders
from pcores import CPUS, EXCLUDED, verify_topology
from run import REPO, load_lock, sha
from windows_qos import high_readback_ok


def verify_profile(manifest, done, lock):
    assert manifest['cpu_set'] == done['affinity'] == lock['profiles']['P2'] == CPUS
    assert manifest['torch_threads'] == 2 and manifest['interop_threads'] == 1
    assert manifest['hardware'] == lock['pcores_topology']
    verify_topology(manifest['hardware'])
    assert manifest['power_start']['power_plugged'] is True and done['power_end']['power_plugged'] is True
    assert manifest['power_mode_start'] == done['power_mode_end'] == lock['power_mode']
    assert manifest['priority_start'] == 32768
    for value in (manifest['qos_start'], done['qos_end']):
        assert high_readback_ok(value) and value['priority_class'] == 32768


def main():
    root, lockpath = (Path(p).resolve() for p in sys.argv[1:])
    lock = load_lock(lockpath)
    run = root/'P2'
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    saved = json.loads((run/'attack_summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (run/'raw.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[-1]['kind'] == 'done'
    verify_profile(manifest, rows[-1], lock)
    assert json.loads((root/'campaign_done.json').read_text()) == [dict(profile='P2', returncode=0)]
    tag = manifest['tag']
    commit = subprocess.check_output(['git', 'rev-parse', tag+'^{commit}'], cwd=REPO, text=True).strip()
    assert commit == manifest['prereg_commit']
    assert manifest['lock_sha256'] == sha(lockpath)
    assert subprocess.check_output(['git', 'show', tag+':'+lockpath.relative_to(REPO).as_posix()], cwd=REPO) == lockpath.read_bytes()
    differences = []
    for name, expected in lock['source_sha256'].items():
        working = (REPO/name).read_bytes()
        tagged = subprocess.check_output(['git', 'show', tag+':'+name], cwd=REPO)
        assert hashlib.sha256(working).hexdigest() == expected
        if working != tagged:
            assert name == 'sponge-sim/src/sponge_sim/hw.py'
            assert working.replace(b'\r\n', b'\n') == tagged.replace(b'\r\n', b'\n')
            differences.append(dict(path=name, working_sha256=expected,
                                    tag_sha256=hashlib.sha256(tagged).hexdigest(), reason='known CRLF/LF only'))
    recomputed = analyze(run)
    assert all(saved[k] == value for k, value in recomputed.items())
    assert saved['profile_scope']['pdh_counter_cpu'] == 0
    journal = root.with_name(root.name+'_affinity.jsonl')
    events = [json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()]
    assert any(e['kind'] == 'ready' and e['excluded'] == EXCLUDED and e['profile_cpu_set'] == CPUS for e in events)
    assert any(e['kind'] == 'profile_topology' and e['topology'] == lock['pcores_topology'] for e in events)
    assert all(not set(e['new']) & set(EXCLUDED) for e in events if e['kind'] == 'intent')
    assert not any(e['kind'].startswith('restore_') for e in events)
    resolved = {e['pid'] for e in events if e['kind'] in ('restored', 'exited')}
    assert {e['pid'] for e in events if e['kind'] in ('intent', 'applied')} <= resolved
    assert all(e['readback'] == e['expected'] for e in events if e['kind'] == 'restored')
    references = []
    baseline = None
    for line, r in enumerate(rows, 1):
        if r['kind'] != 'reference':continue
        median = statistics.median(r['samples_ns'])
        if baseline is None:baseline = median
        assert r['ratio'] == median/baseline and r['flagged'] == (abs(r['ratio']-1) > .03)
        references.append(dict(raw_line=line, block=r['block'], ratio=r['ratio'], flagged=r['flagged']))
    ids = [Path(p).stem for p in lock['input_paths'][:lock['config']['n_images']]]
    counts = saved['counts']
    assert len(counts) == 2*len(ids)
    assert {(r['image'], r['arm']) for r in counts} == {(i, a) for i in ids for a in ('clean', 'attacked')}
    for r in counts:
        assert 0 <= r['k'] <= min(r['n_tau'], 300) and r['n_tau'] <= 8400
        assert r['proto_shape'] == [32, 160, 160] and len(r['prediction_sha256']) == 64
    jobs = [('clean', 'aa_A'), ('clean', 'aa_B'), ('attacked', 'aa_A'), ('attacked', 'aa_B')]
    timing_lines = {}
    for ix, image in enumerate(ids):
        selected = [(line, r) for line, r in enumerate(rows, 1) if r['kind'] == 'timing' and r['image'] == image]
        assert len(selected) == 192
        observed = [[jobs.index((r['arm'], r['stage'])) for _, r in selected if r['rep'] == rep] for rep in range(48)]
        assert observed == carryover_orders(48, ix, 20261107+100*ix)
        assert [r['position'] for _, r in selected] == list(range(4))*48
        assert all(r['ns'] > 0 and r['process_cpu_ns'] >= 0 for _, r in selected)
        timing_lines[image] = [line for line, _ in selected]
    p1path = REPO/'sponge-sim/results/laptop_session_confirm_20261107/full_01/P1_cpu2/attack_summary.json'
    p1 = json.loads(p1path.read_text(encoding='utf-8'))
    previous = {(r['image'], r['arm']): r for r in p1['counts']}
    cross_profile = [dict(image=r['image'], arm=r['arm'], prediction_sha_changed=r['prediction_sha256'] != previous[r['image'], r['arm']]['prediction_sha256'],
                          n_tau_delta=r['n_tau']-previous[r['image'], r['arm']]['n_tau'],
                          k_delta=r['k']-previous[r['image'], r['arm']]['k']) for r in counts]
    receipt = dict(status='SAVED_P2_SOURCE_COVERAGE_AND_ENVIRONMENT_VERIFIED_NO_NEW_INFERENCE',
                   tag=tag, commit=commit, lock_sha256=sha(lockpath), raw_sha256=sha(run/'raw.jsonl'),
                   summary_sha256=sha(run/'attack_summary.json'), auditor_sha256=sha(Path(__file__)),
                   quality_pass=saved['quality_pass'], feasibility_pass=saved['feasibility_pass'],
                   restoration=dict(Counter(e['kind'] for e in events)), tag_line_ending_differences=differences,
                   total_trials=sum(r['kind'] == 'timing' for r in rows), references=references,
                   timing_lines=timing_lines, counts_vs_historical_p1_descriptive_only=cross_profile,
                   duration_raw_s=rows[-1]['wall_utc']-rows[0]['wall_utc'])
    with (root/'verification_receipt.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps({k: receipt[k] for k in ('status', 'quality_pass', 'feasibility_pass', 'total_trials', 'restoration', 'raw_sha256')}, indent=2))


if __name__ == '__main__':
    main()
