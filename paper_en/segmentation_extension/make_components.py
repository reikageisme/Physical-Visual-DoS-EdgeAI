"""Report the closed component campaign; failed gates remain failed."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREFIX = Path('sponge-sim/results/laptop_components_20261107')


def main():
    sources = {}
    runs = {}
    for name in ('nms_kernel/smoke_01', 'nms_kernel/full_01', 'mask_native/smoke_01'):
        base = PREFIX / name
        summary = base / 'P1_cpu2/stage_summary.json'
        s = json.loads((ROOT / summary).read_text())
        receipt = json.loads((ROOT / base / 'verification_receipt.json').read_text())
        raw = base / 'P1_cpu2/raw.jsonl'
        raw_bytes = (ROOT / raw).read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == s['raw_sha256'] == receipt['raw_sha256']
        assert s['complete'] and not s['quality_pass']
        assert receipt['restoration']['restored'] == receipt['restoration']['applied'] == 45
        for path in (summary, raw, base / 'verification_receipt.json'):
            sources[path.as_posix()] = {'sha256': hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
        # One-based JSONL lines retain direct links to the original observations.
        rows = [json.loads(x) for x in raw_bytes.splitlines()]
        sources[raw.as_posix()]['lines'] = {
            kind: [i for i, row in enumerate(rows, 1) if row['kind'] == kind]
            for kind in ('reference', 'timing', 'counts', 'stage_input_check', 'pin_start', 'pin_end', 'done')
        }
        flags = [r for r in s['references'] if r['flagged']]
        runs[name] = dict(
            complete=s['complete'], feasibility_pass=s['feasibility_pass'], quality_pass=s['quality_pass'],
            flags=s['flags'], references=s['n_references'], aa=s['aa'],
            flagged_faster=sum(r['ratio'] < 1 for r in flags),
            flagged_slower=sum(r['ratio'] > 1 for r in flags),
            reference_ratio_range=[min(r['ratio'] for r in s['references']), max(r['ratio'] for r in s['references'])],
            first_reference_median_ms=statistics.median(s['references'][0]['samples_ns']) / 1e6,
            last_reference_median_ms=statistics.median(s['references'][-1]['samples_ns']) / 1e6,
            n_trials=receipt['n_trials'], raw_sha256=s['raw_sha256'],
            diagnostic_only_contrast=s['contrasts'],
        )
    assert runs['nms_kernel/smoke_01']['feasibility_pass']
    assert runs['nms_kernel/full_01']['flags'] == 20
    assert not runs['mask_native/smoke_01']['feasibility_pass']
    assert not (ROOT / PREFIX / 'mask_native/full_01').exists()
    family = dict(
        status='CLOSED_NO_QUALIFIED_COMPONENT_CONTRAST',
        tag='prereg-laptop-components-20261107', commit='0b468db6130714be0521bc3d8bf547b206dbfcef',
        runs=runs, mask_full_status='CANCELLED_AFTER_FAILED_SMOKE_NO_RETRY',
        comparisons={endpoint: dict(eligible=False, testing_p_placeholder=1.0, holm_placeholder=1.0)
                     for endpoint in ('nms_kernel', 'mask_native')},
        interpretation='Both p=1 entries are family-preservation placeholders, not tests or evidence of no effect. No eligible exact sign-flip test performed.',
        scope='One laptop, fixed prepared inputs from the same known PGD pairs. No additive stage, service latency, ARM, physical attack or DoS inference.',
    )
    family_path = PREFIX / 'family_analysis.json'
    with (ROOT / family_path).open('x', encoding='utf-8') as f:
        json.dump(family, f, indent=2, allow_nan=False)
    sources[family_path.as_posix()] = {'sha256': hashlib.sha256((ROOT / family_path).read_bytes()).hexdigest()}
    sources['selectors'] = {'summary': ['flags', 'n_references', 'aa.*.mde_fraction', 'references', 'contrasts', 'endpoint'],
                            'family': ['runs', 'comparisons', 'mask_full_status', 'interpretation']}
    with (OUT / 'laptop_components_sources.json').open('x', encoding='utf-8') as f:
        json.dump(sources, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, name in zip(axes, runs):
        s = json.loads((ROOT / PREFIX / name / 'P1_cpu2/stage_summary.json').read_text())
        refs = s['references']
        ax.axhspan(.97, 1.03, color='tab:green', alpha=.12, label='Registered ±3% band')
        ax.axhline(1, color='gray', linewidth=.7)
        ax.plot(range(len(refs)), [r['ratio'] for r in refs], 'o-', markersize=4)
        for i, r in enumerate(refs):
            if r['flagged']:
                ax.plot(i, r['ratio'], 'o', color='tab:red', markersize=5)
        ax.set(title=name.replace('/', '\n') + f"\n{s['flags']}/{len(refs)} flagged", xlabel='Reference index',
               ylabel='Forward-reference median / initial median', ylim=(.92, 1.07))
        ax.legend(fontsize=7, loc='lower left')
    for suffix in ('png', 'pdf'):
        path = OUT / f'laptop_components.{suffix}'
        assert not path.exists()
        fig.savefig(path, dpi=180)
    plt.close(fig)
    n = runs['nms_kernel/full_01']
    lines = [
        r'% Generated from the closed, failed component campaign by make_components.py.',
        r'\subsection{Prepared-input component controls: qualification failures}',
        r'\label{app:seg-laptop-components}',
        r'Following the accepted total-pipeline sessions, a separate prospective protocol fixes two prepared-input endpoints on the same laptop and frozen image pairs: blocks of 64 original torchvision NMS calls on captured boxes/scores, and one native mask-renderer call on cached prototypes and retained detections. NMS per-call estimates divide whole-block timing by 64 and include the Python facade/loop/index release; mask output release occurs after timing. Preparation is outside timing. These different cache and lifetime scopes are not an additive decomposition of the earlier pipeline.',
        r'The NMS two-image smoke completes 384 trials with zero of four reference blocks flagged, permitting one 20-image full run. The full run completes all 3840 trials and all input-immutability, prediction/count, affinity and source audits, but fails its registered reference-drift criterion: 20 of 22 blocks are flagged (maximum allowed: two). Nineteen flagged references are faster than the initial reference and one is slower. Therefore this failure cannot be described simply as progressive slowing; the physical cause is unresolved.',
        f"The full NMS duplicate-label A/A MDEs are {100*n['aa']['clean']['mde_fraction']:.6f}\\% (clean) and {100*n['aa']['attacked']['mde_fraction']:.6f}\\% (attacked). Passing these two precision bounds does not override the failed reference gate.",
        r'The distinct, predetermined native-mask smoke completes 384 trials but flags its final reference (ratio 1.0328367896175703), exceeding the smoke allowance of zero flags. Its full run is cancelled. No component is repeated, no gate is relaxed, and neither contrast is eligible for primary inference. Testing placeholders of one preserve the declared two-comparison family but are not evidence of no effect. Apparent component ratios remain in diagnostic artifacts only.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.98\textwidth]{segmentation_extension/laptop_components.pdf}\caption{All reference blocks in the closed component campaign. Shading marks the registered 3\% band; red points are flagged. Both full component contrasts remain unqualified.}\label{fig:seg-laptop-components}\end{figure*}',
        r'All three completed runs restore the original affinity of 45 application processes. Failed component controls do not invalidate separately qualified total-pipeline or paired-defense sessions, and do not supply additional evidence of service denial, tail stability, physical attacks or ARM performance.',
    ]
    with (OUT / 'laptop_components.tex').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('Closed family saved: no qualified contrasts; all three runs preserved.')


if __name__ == '__main__':
    main()
