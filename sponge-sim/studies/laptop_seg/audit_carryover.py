"""Offline audit of saved carryover trials; never loads a model or measures it."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from analyze_attack import analyze
from attack_replay import carryover_orders

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, lockpath):
    run = root/'P1_cpu2'
    rows = [json.loads(s) for s in (run/'raw.jsonl').read_text().splitlines()]
    manifest = json.loads((run/'manifest.json').read_text())
    saved = json.loads((run/'attack_summary.json').read_text())
    lock = json.loads(lockpath.read_text())
    tag = manifest['tag']
    commit = subprocess.check_output(['git', 'rev-parse', tag], text=True).strip()
    assert commit == manifest['prereg_commit']
    assert sha(lockpath) == manifest['lock_sha256']
    assert subprocess.check_output(['git', 'show', f'{tag}:{lockpath.relative_to(REPO).as_posix()}']) == lockpath.read_bytes()
    tag_line_endings = []
    for name, expected in lock['source_sha256'].items():
        tagged = subprocess.check_output(['git', 'show', f'{tag}:{name}'])
        working = (REPO/name).read_bytes()
        assert hashlib.sha256(working).hexdigest() == expected
        if hashlib.sha256(tagged).hexdigest() != expected:
            # Historical shared hw.py was normalized by existing git attributes.
            # The prospective lock still binds the exact executed working bytes.
            assert name == 'sponge-sim/src/sponge_sim/hw.py'
            assert tagged.replace(b'\r\n', b'\n') == working.replace(b'\r\n', b'\n')
            tag_line_endings.append(dict(path=name, working_sha256=expected,
                                        tag_sha256=hashlib.sha256(tagged).hexdigest(),
                                        difference='CRLF/LF only; exact working SHA bound before measurement'))
    for item in lock['files']:
        assert sha(REPO/item['path']) == item['sha256']
    recomputed = analyze(run)
    assert all(saved[k] == v for k, v in recomputed.items())
    journal = root.with_name(root.name+'_affinity.jsonl')
    events = [json.loads(s) for s in journal.read_text().splitlines()]
    assert any(e['kind'] == 'ready' for e in events)
    assert not any(e['kind'].startswith('restore_') for e in events)
    resolved = {e['pid'] for e in events if e['kind'] in ('restored', 'exited')}
    assert {e['pid'] for e in events if e['kind'] == 'applied'} <= resolved
    assert all(e['readback'] == e['expected'] for e in events if e['kind'] == 'restored')
    done = rows[-1]
    assert done['kind'] == 'done' and done['affinity'] == manifest['cpu_set'] == [2]
    assert manifest['power_start']['power_plugged'] and done['power_end']['power_plugged']
    assert manifest['power_mode_start'] == done['power_mode_end'] == lock['power_mode']
    for qos in (manifest['qos_start'], done['qos_end']):
        assert qos['control_mask'] & 1 and not qos['state_mask'] & 1
        assert qos['priority_class'] == 32768
    references = []
    baseline = None
    for line, r in enumerate(rows, 1):
        if r['kind'] != 'reference':
            continue
        median = statistics.median(r['samples_ns'])
        if baseline is None:
            baseline = median
        assert median/baseline == r['ratio']
        assert (abs(r['ratio']-1) > .03) == r['flagged']
        references.append(dict(line=line, block=r['block'], ratio=r['ratio'], flagged=r['flagged']))
    ids = [Path(p).stem for p in lock['input_paths'][:lock['config']['n_images']]]
    schedule = []
    for ix, image in enumerate(ids):
        trials = [r for r in rows if r['kind'] == 'timing' and r['image'] == image]
        jobs = [('clean', 'aa_A'), ('clean', 'aa_B'), ('attacked', 'aa_A'), ('attacked', 'aa_B')]
        expected = carryover_orders(48, ix, 20261107+100*ix)
        observed = [[jobs.index((r['arm'], r['stage'])) for r in trials if r['rep'] == rep] for rep in range(48)]
        assert observed == expected
        previous = 'clean' if ix % 2 == 0 else 'attacked'
        contexts = Counter()
        positions = Counter()
        for r in trials:
            contexts[r['arm'], r['stage'], previous] += 1
            positions[r['arm'], r['stage'], r['position']] += 1
            previous = r['arm']
        for arm in ('clean', 'attacked'):
            for prev in ('clean', 'attacked'):
                assert contexts[arm, 'aa_A', prev] == contexts[arm, 'aa_B', prev]
            for label in ('aa_A', 'aa_B'):
                assert [positions[arm, label, p] for p in range(4)] == [12]*4
        schedule.append(dict(image=image, exact_order=True, equal_contexts=True, equal_positions=True))
    old = REPO/'sponge-sim/results/laptop_total_confirmation_20261107/full_01/P1_cpu2/attack_summary.json'
    previous = json.loads(old.read_text())
    count_keys = ('image', 'arm', 'n_tau', 'k', 'prediction_sha256', 'proto_shape')
    assert [{k: r[k] for k in count_keys} for r in saved['counts']] == [
        {k: r[k] for k in count_keys} for r in previous['counts']]
    receipt = dict(status='OFFLINE_VERIFIED_NO_NEW_INFERENCE', tag=tag, commit=commit,
                   lock_sha256=sha(lockpath), raw_sha256=sha(run/'raw.jsonl'),
                   manifest_sha256=sha(run/'manifest.json'), summary_sha256=sha(run/'attack_summary.json'),
                   audit_source_sha256=sha(Path(__file__)), quality_pass=saved['quality_pass'],
                   restored_events=dict(Counter(e['kind'] for e in events)),
                   duration_raw_s=done['wall_utc']-rows[0]['wall_utc'],
                   references=references, schedule=schedule, predictions_match_previous=True,
                   tag_line_ending_differences=tag_line_endings,
                   total_trials=sum(r['kind'] == 'timing' for r in rows))
    with (root/'verification_receipt.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
    print(json.dumps({k: receipt[k] for k in ('status', 'quality_pass', 'raw_sha256', 'total_trials', 'restored_events', 'duration_raw_s')}, indent=2))
    return saved, rows, references


def plot(root, saved, rows, references):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    total = saved['contrasts']['total']
    x = np.arange(len(total['image_ids']))
    for arm, key in [('clean', 'clean_medians_ms'), ('attacked', 'attacked_medians_ms')]:
        axes[0, 0].plot(x, total[key], 'o-', label=arm)
    axes[0, 0].set(xlabel='Frozen image index', ylabel='Median total latency (ms)')
    axes[0, 0].legend()
    for arm in ('clean', 'attacked'):
        ratios = []
        for image in total['image_ids']:
            med = {label: statistics.median(r['ns'] for r in rows if r['kind'] == 'timing'
                   and r['image'] == image and r['arm'] == arm and r['stage'] == label)
                   for label in ('aa_A', 'aa_B')}
            ratios.append(med['aa_A']/med['aa_B'])
        axes[0, 1].plot(x, ratios, 'o-', label=arm)
    axes[0, 1].axhline(1, color='gray', ls='--')
    axes[0, 1].set(xlabel='Frozen image index', ylabel='Per-image A/A median ratio')
    axes[0, 1].legend()
    axes[1, 0].plot([r['ratio'] for r in references], 'o-')
    for y in (.97, 1.03):
        axes[1, 0].axhline(y, color='red', ls='--')
    axes[1, 0].set(xlabel='Reference index (start / 20 images / end)', ylabel='Forward reference / baseline')
    for arm in ('clean', 'attacked'):
        trials = [r for r in rows if r['kind'] == 'timing' and r['arm'] == arm]
        axes[1, 1].scatter([r['ns']/1e6 for r in trials],
                           [r['process_cpu_ns']/r['ns'] for r in trials], s=5, alpha=.25, label=arm)
    axes[1, 1].set(xlabel='Individual wall latency (ms)', ylabel='Process CPU / wall', xscale='log')
    axes[1, 1].legend()
    fig.suptitle('Saved CPU carryover-balanced run — all trials retained; one session')
    for suffix in ('png', 'pdf'):
        path = root/f'carryover_diagnostic.{suffix}'
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    root, lock = (Path(p).resolve() for p in sys.argv[1:])
    plot(root, *audit(root, lock))
