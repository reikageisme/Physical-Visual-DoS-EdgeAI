"""Generate laptop supplement from two saved, independently gated sessions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    records = []
    for family in ('laptop_carryover_20261107', 'laptop_session_confirm_20261107'):
        path = Path('sponge-sim/results')/family/'full_01/P1_cpu2/attack_summary.json'
        summary = json.loads((ROOT/path).read_text())
        raw = ROOT/path.with_name('raw.jsonl')
        assert summary['quality_pass'] and summary['complete']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == summary['raw_sha256']
        records.append(dict(file=path.as_posix(), sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                            raw=path.with_name('raw.jsonl').as_posix(), raw_sha256=summary['raw_sha256'],
                            selectors=['contrasts.total', 'aa', 'flags', 'cpu_diagnostics'], summary=summary))
    lines = [r'% Generated from saved CPU summaries by make_laptop.py.',
             r'\subsection{Repeated laptop CPU measurement with real attacked inputs}',
             r'\label{app:seg-laptop}',
             r'After the earlier rejected timing campaigns, a source audit identified a carryover imbalance: duplicate A/A labels had equal absolute positions but unequal preceding clean/attacked workload distributions. The caller also retained the previous output during the next invocation. A new prospective protocol balanced labels jointly by preceding workload and pair position, used the same callable for both labels, and released output after both clocks stopped. These changes form a bundle; the observations do not isolate an allocator, cache or scheduler mechanism. All earlier failures remain rejected.',
             r'Two prospectively tagged sessions reuse the same 20 frozen whole-image PGD pairs ($L_\infty=16/255$, 100 steps) and YOLOv8n-seg, with FP32 unfused inference, $640\times640$ native masks, confidence 0.25, IoU 0.45 and a fixed torchvision NMS backend. The i5-1335U laptop worker is pinned to logical CPU 2, one hardware thread of a P-core, with one intra-op and one inter-op thread. AboveNormal priority and HighQoS are process-local; the assistant application tree temporarily avoids CPU 2 and its SMT sibling 3, with journaled restoration. No power-plan or thermal-protection changes are made.',
             r'Each image/arm has 48 trials per A/A label; the primary median pools all 96 calls. All trials and spikes are retained. The endpoint spans prepared tensor to native-mask return, excluding output cleanup, decoding, transport, serialization and queueing. Each session independently requires at most two of 22 forward-reference medians outside $\pm3\%$ of baseline, both A/A MDEs at most 1\%, valid load pinning and environment/readback/restoration checks. Intervals below resample the 20 images 20,000 times, RNG seed 20261107; repeats are not independent images.',
             r'\begin{table*}[t]\centering',
             r'\caption{Two separately gated sessions on the same 20 frozen image pairs. Ratios are geometric means of per-image attack/clean median total latency. Both sessions pass; they are not independent image or device replications.}',
             r'\label{tab:seg-laptop}',
             r'\begin{tabular}{lrrrr}\toprule',
             r'Session & Total ratio [95\% CI] & Clean A/A MDE (\%) & Attacked A/A MDE (\%) & Drift flags\\\midrule']
    for index, record in enumerate(records, 1):
        s = record['summary']; t = s['contrasts']['total']
        lines.append(f"{index} & {t['geometric_ratio']:.6f} [{t['ci95'][0]:.6f}, {t['ci95'][1]:.6f}] & "
                     f"{100*s['aa']['clean']['mde_fraction']:.6f} & {100*s['aa']['attacked']['mde_fraction']:.6f} & {s['flags']}/22 \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table*}',
              r'\begin{figure*}[t]\centering\includegraphics[width=.9\textwidth]{segmentation_extension/laptop_sessions.pdf}\caption{Saved per-image median latency and separate session ratio intervals. The same images and deterministic schedule are shared; no pooled cross-session interval is constructed.}\label{fig:seg-laptop}\end{figure*}',
              r'The first session clean MDE is close to the 1\% gate; the unrounded value was used for eligibility. Passing this median-based gate does not establish stable tail latency. Session 1 retains 24/1920 clean and 42/1920 attacked calls exceeding 1.5 times their image-specific median; session 2 retains 14/1920 and 52/1920. Each of these spikes has process-CPU/wall below 0.75. Their cause is unresolved. Process CPU time is coarsely quantized on this Windows installation; PDH nominal frequency is not treated as measured instantaneous clock.',
              r'The second session was registered after inspecting the first and reused its configuration without searching additional repeats or cores. Similar estimates are descriptive repeatability evidence, not an equivalence test. Neither session isolates NMS from mask work because isolated-stage profiling was omitted from this corrected total-only schedule. The installed native renderer chunks floating-point interpolation and returns uint8 masks; a monolithic full-float mask allocation is not assumed. Legacy shared hardware-helper line-ending differences between Git and executed working bytes are disclosed in the audit receipt; the prospective lock binds the exact executed SHA.',
              r'This result supports increased prepared-tensor computation cost in two sessions on one laptop and one model. It does not establish service denial, deadline misses, camera robustness, universal/localized patch effectiveness, ARM/Pi behavior, or timing validity for the separate 1500-image count cohort. No failed session is pooled into these estimates.']
    (OUT/'laptop_cpu.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    sources=[dict({k:v for k,v in record.items() if k!='summary'},
                  selected_values={k:record['summary'][k] for k in
                                   ('quality_pass','aa','contrasts','flags','n_references')})
             for record in records]
    (OUT/'laptop_cpu_sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True)
    for i, record in enumerate(records, 1):
        t = record['summary']['contrasts']['total']
        for arm in ('clean', 'attacked'):
            axes[0].plot(t[arm+'_medians_ms'], marker='o', markersize=3,
                         ls='-' if i == 1 else '--', label=f'{arm}, session {i}')
        point = t['geometric_ratio']; lo, hi = t['ci95']
        axes[1].errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt='o', capsize=5)
    axes[0].set(xlabel='Frozen image index', ylabel='Median total latency (ms)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Attack / clean total-time ratio (95% CI)', yticks=[1, 2],
                yticklabels=['Session 1', 'Session 2'], ylim=(.5, 2.5))
    axes[1].axvline(1, color='gray', ls='--')
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT/f'laptop_sessions.{suffix}', dpi=180)
    plt.close(fig)
    print('Generated two-session supplement and exact-value source map from saved raw hashes')


if __name__ == '__main__':
    main()
