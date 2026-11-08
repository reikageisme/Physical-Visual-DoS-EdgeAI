"""Create the defense supplement from the closed, saved two-comparison family."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent


def main():
    prefix=Path('sponge-sim/results/laptop_defense_20261107')
    family=json.loads((ROOT/prefix/'family_analysis.json').read_text())
    source={}
    for path in (prefix/'family_analysis.json',prefix/'pre300/full_01/P1_cpu2/defense_summary.json',
                 prefix/'out50/smoke_01/P1_cpu2/defense_summary.json'):
        source[path.as_posix()]=dict(sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest())
    pre=json.loads((ROOT/prefix/'pre300/full_01/P1_cpu2/defense_summary.json').read_text())
    failed=json.loads((ROOT/prefix/'out50/smoke_01/P1_cpu2/defense_summary.json').read_text())
    assert pre['quality_pass'] and not failed['feasibility_pass']
    assert not (ROOT/prefix/'out50/full_01').exists()
    for sub,s in [('pre300/full_01',pre),('out50/smoke_01',failed)]:
        raw=prefix/sub/'P1_cpu2/raw.jsonl'
        assert hashlib.sha256((ROOT/raw).read_bytes()).hexdigest()==s['raw_sha256']
        source[raw.as_posix()]=dict(sha256=s['raw_sha256'])
    t=pre['contrasts']['total'];f=family['pre300'];a=pre['aa']
    lines=[r'% Generated from saved defense family by make_defense.py.',
           r'\subsection{Paired pre-NMS defense timing}',
           r'\label{app:seg-laptop-defense}',
           r'After the two accepted CPU sessions, a separate prospective protocol fixes two interventions: cap candidates before NMS at 300, or cap retained outputs at 50. Each comparison uses the identical frozen attacked tensor in both arms and a newly measured, interleaved baseline (candidate cap 30000, output cap 300). The device, model, fixed NMS backend, resolution and carryover-balanced duplicate-label schedule follow the laptop method above. Each block has its own smoke and conditional full run, independent quality gate, SHA-bound source/input lock and app-affinity restoration. A temporary probe records the actual compiled-kernel input count outside timing and restores the original function before warmup and measurement.',
           r'\begin{table*}[t]\centering',
           r'\caption{Predetermined defense comparisons. Only the pre-NMS intervention reached full timing qualification. The failed output-cap smoke is retained and its full run was cancelled, without retry.}',
           r'\label{tab:seg-laptop-defense}',
           r'\begin{tabular}{lrrrr}\toprule',
           r'Intervention & Defended/baseline [95\% CI] & Baseline A/A MDE (\%) & Defended A/A MDE (\%) & Drift flags\\\midrule',
           f"Pre-NMS cap 300 & {t['geometric_ratio']:.6f} [{t['ci95'][0]:.6f}, {t['ci95'][1]:.6f}] & {100*a['baseline']['mde_fraction']:.6f} & {100*a['defended']['mde_fraction']:.6f} & {pre['flags']}/22 \\\\",
           r'Output cap 50 & Not qualified; full cancelled & --- & --- & 1/4 (smoke) \\',
           r'\bottomrule\end{tabular}\end{table*}',
           f"For the pre-NMS cap, total latency decreases by {f['reduction_percent']:.6f}\\% [pointwise 95\\% CI {f['reduction_ci95_percent'][0]:.6f}, {f['reduction_ci95_percent'][1]:.6f}]. The mean paired difference of image medians is {t['mean_paired_delta_ms']:.6f} ms [95\\% CI {t['delta_ci95'][0]:.6f}, {t['delta_ci95'][1]:.6f}]. Each interval uses the registered 20000 image-bootstrap replicates; these intervals are not simultaneous family-wise intervals.",
           r'All 20 image log ratios have the same negative sign. The two-sided exact sign-flip calculation enumerates all $2^{20}$ assignments, giving $p=2/2^{20}$; Holm adjustment over the two registered comparisons gives $p_{\mathrm{Holm}}=4/2^{20}$. The unmeasured/failed output-cap comparison contributes a testing placeholder of one to preserve family size; that value is not a finding of no effect. Sign-flip inference assumes sign exchangeability under the null and does not create random-population sampling or additional image independence.',
           r'Observed kernel inputs fall from 1726--3744 to exactly 300 for every image. Retained masks change from a median of 270.5 (range 177--300) to 98 (84--118). Thus the intervention bounds kernel input but retains nonzero mask work; this total-pipeline contrast does not separately attribute savings to NMS and mask rendering. All raw spikes remain, including 50/1920 baseline and 29/1920 defended calls exceeding 1.5 times their image-specific median, each with process-CPU/wall below 0.75.',
           r'\begin{figure*}[t]\centering\includegraphics[width=.98\textwidth]{segmentation_extension/laptop_defense.pdf}\caption{Qualified pre-NMS intervention: per-image median total latency, observed compiled-NMS input count, and retained mask count. Counts and timing have separate axes and units.}\label{fig:seg-laptop-defense}\end{figure*}',
           r'The output-cap smoke had a forward-reference ratio of 1.0376245029873103 at its first image, exceeding the registered 3\% bound; its apparently lower timings remain diagnostic and are not used as a qualified defense estimate. Both blocks restored all 45 application processes. The study tests fixed saved perturbations, not an adaptive adversary trained against the cap. The earlier clean-accuracy evaluation used confidence 0.001 and letterboxing, unlike this confidence-0.25 square-tensor latency protocol; no jointly matched accuracy--latency curve or availability guarantee is inferred.']
    (OUT/'laptop_defense.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    source['selectors']=dict(pre300=['quality_pass','aa','contrasts.total','counts','cpu_diagnostics'],
                             family=['pre300.raw_p','pre300.holm_p','pre300.reduction_percent','out50.full_status'],
                             out50=['feasibility_pass','flags','references'])
    (OUT/'laptop_defense_sources.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.5),constrained_layout=True)
    for arm,label in [('baseline','Baseline attacked'),('defended','Pre-NMS cap 300')]:
        axes[0].plot(t[arm+'_medians_ms'],'o-',markersize=3,label=label)
        counts=[r for r in pre['counts'] if r['arm']==arm]
        for ax,key in zip(axes[1:],('kernel_n','k')):
            ax.plot([r[key] for r in counts],'o-',markersize=3,label=label)
    for ax,ylabel in zip(axes,('Median total latency (ms)','Observed NMS kernel input count','Retained mask count K')):
        ax.set(xlabel='Frozen image index',ylabel=ylabel)
        ax.legend(fontsize=7)
    for suffix in ('png','pdf'):fig.savefig(OUT/f'laptop_defense.{suffix}',dpi=180)
    plt.close(fig)
    print('Generated qualified pre300 result; output-cap smoke failure explicitly retained')


if __name__=='__main__':main()
