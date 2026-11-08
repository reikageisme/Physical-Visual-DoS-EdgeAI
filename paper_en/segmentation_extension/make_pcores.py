"""Export the closed two-P-core smoke failure, never a qualified effect."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREFIX = Path('sponge-sim/results/laptop_pcores_20261107')


def main():
    run = ROOT/PREFIX/'smoke_01'
    summary = json.loads((run/'P2/attack_summary.json').read_text(encoding='utf-8'))
    audit = json.loads((run/'verification_receipt.json').read_text(encoding='utf-8'))
    raw = [json.loads(x) for x in (run/'P2/raw.jsonl').read_text(encoding='utf-8').splitlines()]
    assert summary['complete'] and not summary['feasibility_pass'] and summary['flags'] == 2
    assert audit['total_trials'] == 384 and not (ROOT/PREFIX/'full_01').exists()
    assert audit['raw_sha256'] == summary['raw_sha256'] == hashlib.sha256((run/'P2/raw.jsonl').read_bytes()).hexdigest()
    pins = [(i, r) for i, r in enumerate(raw, 1) if r['kind'] in ('pin_start', 'pin_end')]
    assert len(pins) == 2 and all(r['verify']['passed'] for _, r in pins)
    closure = dict(status='CLOSED_AFTER_FAILED_SMOKE_FULL_CANCELLED_NO_RETRY', tag=audit['tag'],
                   commit=audit['commit'], raw_sha256=audit['raw_sha256'],
                   complete=True, feasibility_pass=False, quality_pass=False,
                   reference_flags=2, references=4, full_status='CANCELLED_NOT_STARTED',
                   note='Pin gate passed; timing reference gate failed. No qualified multi-core attack estimate.')
    with (ROOT/PREFIX/'protocol_status.json').open('x', encoding='utf-8') as f:json.dump(closure, f, indent=2)
    paths = ['registration_check.json', 'smoke_01/P2/raw.jsonl', 'smoke_01/P2/manifest.json',
             'smoke_01/P2/attack_summary.json', 'smoke_01/verification_receipt.json', 'protocol_status.json']
    source = dict(sources={(PREFIX/p).as_posix(): dict(sha256=hashlib.sha256((ROOT/PREFIX/p).read_bytes()).hexdigest()) for p in paths},
                  selectors=dict(summary=['references', 'aa', 'profile_scope'], audit=['restoration', 'counts_vs_historical_p1_descriptive_only']),
                  raw_pin_lines=[i for i, _ in pins], primary_attack_contrast_eligible=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    refs = summary['references']
    axes[0].axhspan(.97, 1.03, color='tab:green', alpha=.12)
    axes[0].plot(range(4), [r['ratio'] for r in refs], 'o-')
    for i, r in enumerate(refs):
        if r['flagged']:axes[0].plot(i, r['ratio'], 'o', color='tab:red')
    axes[0].set(xticks=range(4), xticklabels=['Start', 'Image 1', 'Image 2', 'End'],
                ylabel='Forward reference / initial median', ylim=(.94, 1.12), title='Timing gate failed: 2/4 flags')
    axes[1].bar(['Start', 'End'], [r['verify']['effective_cores'] for _, r in pins], width=.5)
    axes[1].axhline(1.4, color='tab:red', ls='--', label='Registered lower bound')
    axes[1].axhline(2.15, color='gray', ls='--', label='Registered upper bound')
    axes[1].set(ylabel='Process CPU / wall under pin-check load', ylim=(0, 2.4), title='Pin checks passed on CPUs [0,2]')
    axes[1].legend(fontsize=8)
    for suffix in ('png', 'pdf'):
        p = OUT/f'laptop_pcores.{suffix}'
        assert not p.exists()
        fig.savefig(p, dpi=180)
    plt.close(fig)
    lines = [r'% Generated from the closed P2 smoke by make_pcores.py.',
        r'\subsection{Two physical P cores: smoke rejected}',
        r'\label{app:seg-laptop-pcores}',
        r'A separate prospective protocol fixes logical CPUs [0,2], one thread from each physical P core, with two intra-op and one inter-op threads. The same corrected whole-pipeline worker, fixed NMS backend, model and saved input pairs are reused. The owned application temporarily excludes both P cores and their SMT siblings [0,1,2,3]. This changes the application placement relative to the historical one-core sessions, so cross-profile speedup is not an identified treatment effect.',
        r'The single two-image smoke completes all 384 timing trials and passes both pin-load checks (approximately 1.95 effective cores). All 45 application affinities are restored. However, two of four forward references are flagged: ratios 1.0953159727928494 before the first image and 1.0317927329528502 at the end. The registered smoke allows no flags, so the full 20-image run is cancelled without retry, additional cores or a warmup sweep. Apparent latency ratios and the two-image A/A estimates remain diagnostic only; no qualified multi-core attack estimate follows.',
        r'All four clean/attacked prediction hashes differ from the historical one-core outputs, while the candidate and retained-mask counts are unchanged for these two images. These numerical observations are retained rather than treated as an image-exclusion rule. Process-CPU/wall values on two cores represent effective CPU use and may exceed one; the PDH query covers CPU0 only. This rejected smoke does not establish that multi-core pinning is impossible, identify a thermal or scheduling cause, or support a service-denial conclusion.',
        r'\begin{figure*}[t]\centering\includegraphics[width=.90\textwidth]{segmentation_extension/laptop_pcores.pdf}\caption{Two-P-core smoke: reference-timing failure alongside passing start/end pin-load checks. These are distinct gates; passing affinity verification does not override timing instability.}\label{fig:seg-laptop-pcores}\end{figure*}']
    with (OUT/'laptop_pcores.tex').open('x', encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    with (OUT/'laptop_pcores_sources.json').open('x', encoding='utf-8') as f:json.dump(source, f, indent=2)
    print('P2 failure exported; full cancelled and protocol closed.')


if __name__ == '__main__':
    main()
