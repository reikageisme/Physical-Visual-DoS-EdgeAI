"""English tables for paper_en/main.tex, generated from the same summary files as paper_vi/tables/make_tables.py.

    python paper_en/tables_q3/make_tables_en.py          # writes paper_en/tables_q3/tab_*.tex and numbers_en.json
    python paper_en/tables_q3/make_tables_en.py --check  # report tables that differ from a fresh build (hand edits) -> exit 1

Data access, selectors and rounding are shared with the Vietnamese builder (imported); only labels and the decimal mark differ.
Do not edit tab_*.tex by hand.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
_spec = importlib.util.spec_from_file_location('mt', ROOT / 'paper/vi/tables/make_tables.py')
mt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mt)
j = mt.j
BE = mt.BE
GEOM = mt.GEOM
BE_EN = {'torch_pure': 'pure PyTorch', 'tv_nms': 'torchvision \\texttt{nms}', 'tv_batched': 'torchvision batched', 'ultra_native': 'Ultralytics native'}
GEOM_EN = {'disjoint': 'disjoint', 'mixed': 'mixed', 'dense_cluster': 'dense cluster'}


class Tab(mt.Tab):
    def n(self, x, src, d=None, sig=None):
        """Format a number (decimal point) and record its source."""
        if isinstance(x, str):
            s = '$>$' + x[1:] if x.startswith('>') else x
        elif isinstance(x, int) and d is None and sig is None:
            s = f'{x}'
        else:
            if sig is not None:
                d = max(sig - 1 - int(math.floor(math.log10(abs(x)))), 0) if x else 0
            s = f'{x:.{d}f}'
        if s.startswith('-'):
            s = '$-$' + s[1:]
        self.cells.append((s, src))
        return s


REG = {}


def write(t, body):
    REG[t.name] = [dict(text=s, source=src) for s, src in t.cells]
    return t.name, body


END = '\n\\bottomrule\n\\end{tabular}\n'


def t_aa():
    t = Tab('tab_aa')
    f320, f640, f640a = 'e5/AA_P1cpu2_20261101/summary.json', 'e5/AA640_P1cpu2_20261102b/summary.json', 'e5/AA640_P1cpu2_20261102/summary.json'
    a, b, c = (j(f)['profiles']['P1_cpu2'] for f in (f320, f640, f640a))
    rows = [('pre', 'Pre-processing'), ('forward', 'Forward pass'), ('tau', 'Score filter ($\\tau$)'), ('nms_torch_pure', 'NMS (pure PyTorch)'),
            ('nms_tv_nms', 'NMS (torchvision)'), ('total', 'End-to-end')]
    lines = []
    for k, lab in rows:
        cells = []
        for f, p in ((f320, a), (f640, b)):
            st = p['stages'][k]
            cells += [t.n(st['mde_ratio_pct'], f'{f}:profiles.P1_cpu2.stages.{k}.mde_ratio_pct', d=2),
                      t.n(st['median_of_image_medians_ms'], f'{f}:profiles.P1_cpu2.stages.{k}.median_of_image_medians_ms', sig=3)]
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    st = c['stages']['total']
    lines.append('End-to-end, first 640-px A/A (noisy) & & & ' + t.n(st['mde_ratio_pct'], f'{f640a}:profiles.P1_cpu2.stages.total.mde_ratio_pct', d=2)
                 + ' & ' + t.n(st['median_of_image_medians_ms'], f'{f640a}:profiles.P1_cpu2.stages.total.median_of_image_medians_ms', sig=3) + ' \\\\')
    fl = lambda p, f: t.n(len(p['drift']['flagged']), f'{f}:profiles.P1_cpu2.drift.flagged(len)') + '/' + t.n(p['drift']['reference_points'], f'{f}:profiles.P1_cpu2.drift.reference_points')  # noqa: E731
    lines.append('Flagged reference points / total & ' + fl(a, f320) + ' & & ' + fl(b, f640) + ' & \\\\')
    eff = t.n(a['pin']['effective_cores'], f'{f320}:profiles.P1_cpu2.pin.effective_cores', d=2)
    body = rf"""\begin{{table}}[t]
\centering\small
\caption{{A/A calibration on the pinned laptop profile (logical CPU 2; {eff} effective cores under load at 320 px): MDE (\%) and median time of each stage, 100 images $\times$ 20 repetitions. The A/A end-to-end time starts from the loaded image and includes pre-processing (resizing); the patch and injection measurements start from prepared tensors and therefore have a smaller $t_{{\mathrm{{total}}}}$. 640 px: the repeated A/A. Flagged blocks remain in the analysis.}}
\label{{tab:aa}}
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{320 px}} & \multicolumn{{2}}{{c}}{{640 px}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Stage & MDE (\%) & Median (ms) & MDE (\%) & Median (ms) \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


def t_dose():
    t = Tab('tab_dose')
    f = 'e1/E1_P1cpu2_20261101/summary.json'
    s = j(f)
    aa = j('e5/AA_P1cpu2_20261101/summary.json')['profiles']['P1_cpu2']['stages']
    mde = max(aa['nms_torch_pure']['mde_ratio_pct'], aa['nms_tv_nms']['mde_ratio_pct'])
    lines = []
    for be in BE:
        for g in GEOM:
            k = f'P1_cpu2|{be}|{g}'
            v = s['dose_1e2_to_1e4'][k]
            sel = f'{f}:dose_1e2_to_1e4.{k}'
            ok = v['ci95'][0] > 1 + mde / 100
            lines.append(f'{BE_EN[be]} & {GEOM_EN[g]} & ' + t.n(v['median_ms_N1e2'], sel + '.median_ms_N1e2', sig=3) + ' & '
                         + t.n(v['median_ms_N1e4'], sel + '.median_ms_N1e4', sig=3) + ' & ' + t.n(v['ratio'], sel + '.ratio', sig=3)
                         + ' & ' + t.ci(v['ci95'][0], v['ci95'][1], sel + '.ci95', sig=3) + ' & ' + ('yes' if ok else 'no') + ' \\\\')
    m = t.n(mde, 'e5/AA_P1cpu2_20261101/summary.json:max(stages.nms_torch_pure.mde_ratio_pct,stages.nms_tv_nms.mde_ratio_pct)', d=2)
    body = rf"""\begin{{table*}}[t]
\centering\small
\caption{{Median NMS latency at $\Ntau=10^2$ and $10^4$ (10\,000-anchor head, 80 classes, no cap) and their ratio, for all four implementations $\times$ three geometries. 95\% confidence interval: bootstrap over repetitions (30 per cell). Last column: lower bound above $1+\mathrm{{MDE}}$, with the MDE taken as the larger of the two A/A MDEs of the NMS stage ({m}\%).}}
\label{{tab:dose}}
\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\toprule
Implementation & Geometry & $t_{{\mathrm{{NMS}}}}(10^2)$ (ms) & $t_{{\mathrm{{NMS}}}}(10^4)$ (ms) & Ratio & 95\% CI & Above MDE \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


def t_cap():
    t = Tab('tab_cap')
    f = 'e1/E1_P1cpu2_20261101/summary.json'
    s = j(f)
    lines = []
    for g in GEOM:
        for i, be in enumerate(BE):
            k0, k3 = f'P1_cpu2|10000|{be}|{g}|80|None|10000', f'P1_cpu2|10000|{be}|{g}|80|300|10000'
            kc = f'P1_cpu2|10000|{be}|{g}|80|10000'
            c = s['contrasts']['cap300_over_capNone'][kc]
            sel = f'{f}:1/contrasts.cap300_over_capNone.{kc}'
            lines.append((GEOM_EN[g].capitalize() if i == 0 else '') + f' & {BE_EN[be]} & '
                         + t.n(s['curves'][k0]['median_ms'], f'{f}:curves.{k0}.median_ms', sig=3) + ' & '
                         + t.n(s['curves'][k3]['median_ms'], f'{f}:curves.{k3}.median_ms', sig=3) + ' & '
                         + t.n(1 / c['ratio'], sel + '.ratio', sig=3) + ' (' + t.ci(1 / c['ci95'][1], 1 / c['ci95'][0], sel + '.ci95(inverted)', sig=3) + ') \\\\')
    body = r"""\begin{table}[t]
\centering\small
\caption{Median NMS latency (ms) at $\Ntau=10^4$ (10\,000-anchor head, 80 classes) without a cap and with a top-300 cap, and the reduction factor (uncapped divided by capped; 95\% CI, bootstrap over repetitions).}
\label{tab:cap}
\begin{tabular}{@{}llccc@{}}
\toprule
Geometry & Implementation & No cap & Cap 300 & Reduction \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


def t_nstar():
    t = Tab('tab_nstar')
    fa, fs, fm = 'e9/E9_P1cpu2_20261102/summary_part_a.json', 'e9/E9_P1cpu2_20261102/summary_part_a_sens_aa640b.json', 'amdahl/amdahl_vs_e9_posthoc.json'
    a, s, m = j(fa), j(fs), j(fm)
    lines = []
    for be in BE:
        for i, g in enumerate(GEOM):
            k3, k6 = f'320|{be}|{g}|capNone', f'640|{be}|{g}|capNone'
            row = [BE_EN[be] if i == 0 else '', GEOM_EN[g],
                   t.n(a['n_star'][k3], f'{fa}:n_star.{k3}'), t.n(m['n_star'][k3]['predicted'], f'{fm}:n_star.{k3}.predicted'),
                   t.n(m['n_star_unflagged_blocks'][k3]['measured'], f'{fm}:n_star_unflagged_blocks.{k3}.measured'),
                   t.n(a['n_star'][k6], f'{fa}:n_star.{k6}'), t.n(s['n_star'][k6], f'{fs}:n_star.{k6}'),
                   t.n(m['n_star'][k6]['predicted'], f'{fm}:n_star.{k6}.predicted'),
                   t.n(m['n_star_unflagged_blocks'][k6]['measured'], f'{fm}:n_star_unflagged_blocks.{k6}.measured')]
            lines.append(' & '.join(row) + ' \\\\')
    m3 = t.n(a['mde_total_pct']['320'], f'{fa}:mde_total_pct.320', d=2)
    m6 = t.n(a['mde_total_pct']['640'], f'{fa}:mde_total_pct.640', d=2)
    m6b = t.n(s['mde_total_pct']['640'], f'{fs}:mde_total_pct.640', d=2)
    nf3 = t.n(m['n_star_unflagged_blocks']['320|torch_pure|disjoint|capNone']['n_images'], f'{fm}:n_star_unflagged_blocks.320|torch_pure|disjoint|capNone.n_images')
    nf6 = t.n(m['n_star_unflagged_blocks']['640|torch_pure|disjoint|capNone']['n_images'], f'{fm}:n_star_unflagged_blocks.640|torch_pure|disjoint|capNone.n_images')
    fb = t.n(a['drift']['flagged'], f'{fa}:drift.flagged') + '/' + t.n(a['drift']['points'], f'{fa}:drift.points')
    body = rf"""\begin{{table*}}[t]
\centering\small
\caption{{Threshold $N^{{*}}$: the smallest grid level at which end-to-end time rises above the MDE (lower bound of the 95\% CI, bootstrap over images, $>1+\mathrm{{MDE}}$); full-pipeline injection, 20 images, no top-$k$ cap; ``$>$'' means that no grid level exceeds it. ``Measured'' is the preregistered computation (primary result) with the preregistered MDE ({m3}\% at 320 px; {m6}\% at 640 px, from the first 640-px A/A); the column ``MDE {m6b}\%'' uses the MDE of the repeated 640-px A/A (not preregistered). $^{{\dagger}}$Post hoc: $N^{{*}}$ predicted by the Amdahl model from the injection curves (2100-anchor head at 320 px, 8400-anchor head at 640 px; the 10\,000-anchor head is not used) with the clean $t_{{\mathrm{{total}}}}$ of the same session and the preregistered MDE. $^{{\S}}$Preregistered sensitivity analysis: blocks with a flagged reference point removed ({fb} points flagged; {nf3} images remain at 320 px and {nf6} at 640 px), preregistered MDE.}}
\label{{tab:nstar}}
\begin{{tabular}}{{@{{}}llccccccc@{{}}}}
\toprule
 & & \multicolumn{{3}}{{c}}{{320 px (grid 100, 300, 1000, 2100)}} & \multicolumn{{4}}{{c}}{{640 px (grid 300, 1000, 3000, 8400)}} \\
\cmidrule(lr){{3-5}}\cmidrule(lr){{6-9}}
Implementation & Geometry & Measured & Predicted$^{{\dagger}}$ & Unflagged$^{{\S}}$ & Measured & MDE {m6b}\% & Predicted$^{{\dagger}}$ & Unflagged$^{{\S}}$ \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


def t_amdahl():
    t = Tab('tab_amdahl')
    fm, fe = 'amdahl/amdahl_vs_e9_posthoc.json', 'e10/E10_P1cpu2_20261102/summary.json'
    m, e = j(fm), j(fe)
    lines = []
    for cv, nmax in ((320, 2100), (640, 8400)):
        lines.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{A. Full-pipeline injection, \\textbf{{post hoc}}; {cv} px, disjoint boxes, no cap, $\\Ntau={nmax}$}}}} \\\\')
        for be in BE:
            k = f'{cv}|{be}|disjoint|capNone'
            r = [x for x in m['cells'][k]['rows'] if x['n'] == nmax][0]
            i = m['cells'][k]['rows'].index(r)
            sel = f'{fm}:cells.{k}.rows[{i}]'
            lines.append(f'{BE_EN[be]} & ' + t.n(r['r_pred'], sel + '.r_pred', sig=3) + ' & ' + t.n(r['r_meas'], sel + '.r_meas', sig=3) + ' & '
                         + t.n(r['abs_log_err'], sel + '.abs_log_err', d=3) + ' & ' + t.n(m['n_star'][k]['predicted'], f'{fm}:n_star.{k}.predicted') + ' / '
                         + t.n(m['n_star'][k]['measured'], f'{fm}:n_star.{k}.measured') + ' & -- \\\\')
    lines.append('\\midrule')
    lines.append('\\multicolumn{6}{@{}l}{\\textit{B. Whole-image attack variants (YOLOv5s, 640 px), preregistered model and rule}} \\\\')
    for mth, mv in (('overload', 'Overload'), ('phantom_sponges', 'Phantom Sponges')):
        for be in ('tv_nms', 'tv_batched', 'ultra_native'):
            k = f'{mth}|{be}'
            am = e['amdahl'][k]
            sel = f'{fe}:amdahl.{k}'
            lines.append(f'{mv}, {BE_EN[be]} & ' + t.n(am['r_pred_mean'], sel + '.r_pred_mean', d=3) + ' & ' + t.n(am['r_meas_mean'], sel + '.r_meas_mean', d=3) + ' & '
                         + t.n(am['median_abs_log_err'], sel + '.median_abs_log_err', d=3) + ' (' + t.ci(*am['ci95'], sel + '.ci95', d=3) + ') & -- & '
                         + ('pass' if am['passes_prereg_rule'] else '\\textbf{fail}') + ' \\\\')
    pa = j('e9/E9_P1cpu2_20261102/summary_part_a.json')['n_star']
    diff = [k for k, v in m['n_star'].items() if v['measured'] != pa[k]]
    nd = t.n(len(diff), f'{fm}:n_star[*].measured != e9/E9_P1cpu2_20261102/summary_part_a.json:n_star[*] (count)') + '/' + t.n(len(m['n_star']), f'{fm}:n_star(len)')
    ku = '320|ultra_native|disjoint|capNone'
    du = t.n(m['n_star'][ku]['measured'], f'{fm}:n_star.{ku}.measured') + ' vs.\\ ' + t.n(pa[ku], f'e9/E9_P1cpu2_20261102/summary_part_a.json:n_star.{ku}')
    sm = m['summary']
    e1 = t.n(sm['median_abs_log_err'], f'{fm}:summary.median_abs_log_err', d=3)
    e9 = t.n(sm['p90_abs_log_err'], f'{fm}:summary.p90_abs_log_err', d=3)
    nc = t.n(sm['n_cells'], f'{fm}:summary.n_cells')
    ag = t.n(sm['n_star_exact_agreement'], f'{fm}:summary.n_star_exact_agreement')
    npass = sum(v['passes_prereg_rule'] for v in e['amdahl'].values())
    np_ = t.n(npass, f'{fe}:amdahl[*].passes_prereg_rule (count)') + '/' + t.n(len(e['amdahl']), f'{fe}:amdahl(len)')
    body = rf"""\begin{{table*}}[t]
\centering\small
\caption{{Amdahl-type model $R_{{\mathrm{{pred}}}} = 1+\Delta t_{{\mathrm{{NMS}}}}/t_{{\mathrm{{total}}}}$ against the measured end-to-end ratio $R_{{\mathrm{{meas}}}}$. Error is $|\ln(R_{{\mathrm{{meas}}}}/R_{{\mathrm{{pred}}}})|$. \textbf{{Part A is post hoc}}: the model was written after the full-pipeline injection had been seen; over all {nc} cells (four implementations, three geometries, two caps, two resolutions) the median error is {e1}, the 90th percentile {e9}, and the predicted $N^{{*}}$ equals the measured $N^{{*}}$ in {ag} cells. The measured $N^{{*}}$ in part A is a \textbf{{sensitivity analysis}} (non-preregistered aggregation): it uses the mean log ratio against the median of the three $\Ntau=0$ cells (the same computation as the prediction), so it differs from Table~\ref{{tab:nstar}} (mean ratio) in {nd} cells, e.g.\ Ultralytics, disjoint, 320 px: {du}. Part B is computed for YOLOv5s alone: $t_{{\mathrm{{total}}}}$ is the clean end-to-end time of the YOLOv5s session itself, and the NMS curve is the synthetic injection curve (it depends only on the candidate set, not on the model); specification and rule (upper bound of the 95\% CI of the median error $<0.05$) were preregistered before measurement; {np_} cells pass, and the three failing cells are reported in full.}}
\label{{tab:amdahl}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
 & $R_{{\mathrm{{pred}}}}$ & $R_{{\mathrm{{meas}}}}$ & Error (95\% CI) & $N^{{*}}$ predicted / measured & Rule \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


def t_attacks():
    t = Tab('tab_attacks')
    f = 'e10/E10_P1cpu2_20261102/summary.json'
    s = j(f)
    fd = 'e10/E10_P1cpu2_20261102/stage_medians_posthoc.json'
    sd = j(fd)
    lines = []
    for m, mv in (('overload', 'Overload'), ('phantom_sponges', 'Phantom Sponges')):
        nr, nm = s['n_ratio'][f'{m}|tau0.25'], s['n_median'][f'{m}|tau0.25']
        for i, be in enumerate(('tv_nms', 'tv_batched', 'ultra_native')):
            k = f'{m}|{be}'
            tr = s['total_ratio'][k]
            head = (f'{mv} & ' + t.n(nm['clean'], f'{f}:n_median.{m}|tau0.25.clean', d=1) + ' $\\to$ ' + t.n(nm['patched'], f'{f}:n_median.{m}|tau0.25.patched', d=1)
                    + ' & ' + t.n(nr['mean'], f'{f}:n_ratio.{m}|tau0.25.mean', sig=3) + ' (' + t.ci(*nr['ci95'], f'{f}:n_ratio.{m}|tau0.25.ci95', sig=3) + ')') if i == 0 else ' & & '
            tc = sd['total_ms'][f'clean|{be}']['median']
            dstar = s['in_session_mde_pct'][be] / 100 * tc
            dn = sd['nms_ms'][f'{m}|{be}']['median'] - sd['nms_ms'][f'clean|{be}']['median']
            lines.append(head + f' & {BE_EN[be]} & ' + t.n(tr['mean'], f'{f}:total_ratio.{k}.mean', d=3) + ' (' + t.ci(*tr['ci95'], f'{f}:total_ratio.{k}.ci95', d=3) + ') & '
                         + t.n(s['in_session_mde_pct'][be], f'{f}:in_session_mde_pct.{be}', d=2) + ' & ' + ('yes' if tr['exceeds_in_session_mde'] else 'no') + ' & '
                         + t.n(dstar, f'{f}:in_session_mde_pct.{be}/100 * {fd}:total_ms.clean|{be}.median', sig=2) + ' & '
                         + t.n(dn, f'{fd}:nms_ms.{m}|{be}.median - nms_ms.clean|{be}.median', sig=2) + ' \\\\')
    ni = t.n(s['n_images'], f'{f}:n_images')
    body = rf"""\begin{{table*}}[t]
\centering\small
\caption{{Adaptive variants of two whole-image attacks, after Overload and Phantom Sponges (YOLOv5s, 640 px, additive whole-image perturbation, white box; {ni} images $\times$ 5 seeds), measured on one pinned CPU core, float32. Median $\Ntau$ clean $\to$ perturbed and mean $\Ntau$ ratio ($\tau=0.25$); end-to-end ratio (95\% CI, two-level bootstrap); in-session end-to-end MDE (two clean arms); ``above MDE'' when the lower CI bound is $>1+\mathrm{{MDE}}$. All quantities come from this YOLOv5s session; nothing is compared with the YOLOv8n threshold $N^{{*}}$. $^{{\dagger}}$\textbf{{Post hoc}}, descriptive: the YOLOv5s-specific threshold $\Delta t^{{*}}=\mathrm{{MDE}}\times t_{{\mathrm{{total}}}}$ (median clean end-to-end time of the same session) and the median increase of the NMS stage. The Amdahl model for these cells is in Table~\ref{{tab:amdahl}}.}}
\label{{tab:attacks}}
\begin{{tabular}}{{@{{}}lcclccccc@{{}}}}
\toprule
Variant & $\Ntau$ clean $\to$ attack & $\Ntau$ ratio (95\% CI) & Implementation & End-to-end ratio (95\% CI) & MDE (\%) & Above MDE & $\Delta t^{{*}}$ (ms)$^{{\dagger}}$ & $\Delta t_{{\mathrm{{NMS}}}}$ (ms)$^{{\dagger}}$ \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


METHOD_EN = {('sg_ga', 'saliency'): 'GA, saliency placement', ('ga_center', 'center'): 'GA, centre placement', ('ga_random_location', 'random'): 'GA, random placement',
             ('random_search', 'saliency'): 'random search, saliency', ('gradient_tanh', 'saliency'): 'gradient ascent (tanh), saliency'}


def t_spec():
    t = Tab('tab_spec')
    lines = []
    seeds = set()
    for c in mt.CONDS:
        sp = mt.spec(c)
        for p in mt.ckpts(c, 62500):
            seeds.add(int(p.parent.name[4:]))
        src = f'{mt.E7}/{c}/seed0/{mt.ckpts(c, 62500, 0)[0].name}:specification'
        meth = METHOD_EN[(sp['method'], sp['location'])]
        lines.append(f'{c.split("_")[0]} & {meth} & ' + t.n(sp['canvas'], src + '.canvas') + ' & ' + t.n(sp['size'], src + '.size') + ' & '
                     + t.n(sp['threshold'], src + '.threshold', d=2) + ' & ' + t.n(sp['n_train'], src + '.n_train') + ' \\\\')
    sd = f'{min(seeds)}--{max(seeds)}'
    body = rf"""\begin{{table}}[t]
\centering\small
\caption{{Conditions of the case study, read from the specification stored in the checkpoints of the patch generation run. $\tau_{{\mathrm{{obj}}}}$ is the threshold in the optimization objective. All conditions use software EOT, seeds {sd}, and three budget levels.}}
\label{{tab:spec}}
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\toprule
Condition & Method & Input (px) & Patch (px) & $\tau_{{\mathrm{{obj}}}}$ & Training images \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


def t_nk():
    t = Tab('tab_nk')
    f = 'e9/E9_P1cpu2_20261102/summary_part_b.json'
    s = j(f)
    lines = []
    for key, lab in (('test2_e6a1_640_confirmatory', '640 px, confirmatory evaluation'), ('test1_e6_320_posthoc', '320 px, \\textbf{post hoc}')):
        v0 = s[key]['torch_pure']
        lines.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{lab} (' + t.n(v0['n_units'], f'{f}:{key}.torch_pure.n_units') + ' units, ' + t.n(v0['n_images'], f'{f}:{key}.torch_pure.n_images') + ' images)}} \\\\')
        for be in BE:
            v = s[key][be]
            sel = f'{f}:{key}.{be}'
            lines.append(f'{BE_EN[be]} & ' + t.n(v['median_abs_log_err_M_N'], sel + '.median_abs_log_err_M_N', d=2) + ' & ' + t.n(v['median_abs_log_err_M_NK'], sel + '.median_abs_log_err_M_NK', d=2)
                         + ' & ' + t.n(v['delta'], sel + '.delta', d=2) + ' (' + t.ci(v['delta_ci95'][0], v['delta_ci95'][1], sel + '.delta_ci95', d=2, sep=' to ') + ') \\\\')
    body = r"""\begin{table}[t]
\centering\small
\caption{Median error $|\ln(t_{\mathrm{meas}}/t_{\mathrm{pred}})|$ of $M_N$ and $M_{NK}$ on real candidate sets, and the difference $\Delta$ ($M_N$ minus $M_{NK}$) with a 95\% two-level bootstrap CI (images, units within images). $\Delta>0$ means that adding $K$ reduces the error. The preregistered confirmation uses the first 640-px patch measurement (high noise).}
\label{tab:nk}
\begin{tabular}{@{}lccc@{}}
\toprule
Implementation & $M_N$ & $M_{NK}$ & $\Delta$ (95\% CI) \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


ARM_EN = [('C1', '320 px, 64-px patch, $\\tau_{\\mathrm{obj}}$ 0.01'), ('C3', '320 px, 64-px patch, $\\tau_{\\mathrm{obj}}$ 0.25'), ('B3', 'Random search (320 px)'),
          ('B1', 'GA, centre placement (320 px)'), ('B2', 'GA, random placement (320 px)'), ('B4', 'Gradient ascent (320 px)'),
          ('C2', '640 px, 128-px patch, $\\tau_{\\mathrm{obj}}$ 0.01'), ('C4', '640 px, 64-px patch, $\\tau_{\\mathrm{obj}}$ 0.01')]


def t_patches():
    t = Tab('tab_patches')
    tc = j(mt.TC)
    lines = []
    for arm, lab in ARM_EN:
        f = mt.E6_SRC[arm]
        a = j(f)['arms'][arm]
        sel = f'{f}:arms.{arm}'
        cv = mt.CANV[arm]
        mde = mt.mde_total(cv)
        cells = [t.n(a['N_tau0.01']['ratio_mean'], sel + '.N_tau0.01.ratio_mean', d=2) + ' (' + t.ci(*a['N_tau0.01']['ratio_ci95'], sel + '.N_tau0.01.ratio_ci95', d=2) + ')',
                 t.n(a['N_tau0.25']['ratio_mean'], sel + '.N_tau0.25.ratio_mean', d=2) + ' (' + t.ci(*a['N_tau0.25']['ratio_ci95'], sel + '.N_tau0.25.ratio_ci95', d=2) + ')',
                 t.n(a['nms_tau0.01_tv_nms']['ratio_mean'], sel + '.nms_tau0.01_tv_nms.ratio_mean', d=3) + ' (' + t.ci(*a['nms_tau0.01_tv_nms']['ratio_ci95'], sel + '.nms_tau0.01_tv_nms.ratio_ci95', d=3) + ')']
        for be in ('tv_nms', 'torch_pure'):
            v = tc['arms'][f'{cv}|{arm}|{be}']
            s2 = f'{mt.TC}:arms.{cv}|{arm}|{be}'
            cells.append(t.n(v['r_pred'], s2 + '.r_pred', d=4) + (' (above)' if v['r_pred'] > 1 + mde / 100 else ''))
        cells.append(t.n(1 + mde / 100, f'1 + {mt.MDE_SRC[cv]}:profiles.P1_cpu2.stages.total.mde_ratio_pct/100', d=4))
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    body = r"""\begin{table*}[t]
\centering\small
\caption{Effect of the frozen patches (SHA-256) relative to clean images of the same canvas, 100 images $\times$ 5 seeds (0--4, chosen in advance). The $\Ntau$ ratios and the NMS-stage ratio (torchvision, $\tau=0.01$) are \textbf{measurements} of the patch re-measurement on the pinned profile; 95\% CI, two-level bootstrap over images $\times$ seeds. The two $R_{\mathrm{pred}}$ columns are \textbf{predictions, not measurements}: $R_{\mathrm{pred}} = 1 + \Delta t_{\mathrm{NMS}}/t_{\mathrm{total}}$ (Amdahl-type budget model, post hoc), with $\Delta t_{\mathrm{NMS}}$ measured per image $\times$ seed pair and $t_{\mathrm{total}}$ the clean end-to-end time of the same session. No measured end-to-end column is reported: the end-to-end timing of this run included the patch insertion for patched arms but not for clean images, and the re-measurement after the fix was rejected for drift (Table~\ref{tab:dev}). ``(above)'': $R_{\mathrm{pred}} > 1+\mathrm{MDE}$, with the end-to-end A/A MDE at the same resolution. 640-px rows are compared with clean 640-px images (second addendum run); baselines B1--B4 pool images 1--50 and 51--100 from two sessions, each image compared with the clean image of its own session.}
\label{tab:patches}
\begin{tabular}{@{}lcccccc@{}}
\toprule
Condition & $\Ntau$ ratio ($\tau$=0.01) & $\Ntau$ ratio ($\tau$=0.25) & NMS ratio (measured) & $R_{\mathrm{pred}}$, \texttt{nms} & $R_{\mathrm{pred}}$, pure PyTorch & $1+\mathrm{MDE}$ \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


def t_threshold():
    t = Tab('tab_threshold')
    tc = j(mt.TC)
    TC = mt.TC
    lines = []
    for cv in ('320', '640'):
        lines.append(f'Clean $t_{{\\mathrm{{total}}}}$, {cv} px (ms) & ' + ' & '.join(t.n(tc['threshold_ms'][be][cv]['t_total_clean_ms'], f'{TC}:threshold_ms.{be}.{cv}.t_total_clean_ms', sig=3) for be in ('tv_nms', 'torch_pure')) + ' \\\\')
        lines.append(f'A/A end-to-end MDE, {cv} px (\\%) & ' + ' & '.join(t.n(tc['mde_total_pct'][cv], f'{TC}:mde_total_pct.{cv}', d=2) for _ in ('tv_nms', 'torch_pure')) + ' \\\\')
        lines.append(f'Threshold $\\Delta t^{{*}}=\\mathrm{{MDE}}\\times t_{{\\mathrm{{total}}}}$, {cv} px (ms) & ' + ' & '.join(t.n(tc['threshold_ms'][be][cv]['threshold_ms'], f'{TC}:threshold_ms.{be}.{cv}.threshold_ms', sig=2) for be in ('tv_nms', 'torch_pure')) + ' \\\\')
    lines.append('\\midrule')
    b = {be: tc['arms'][f'320|B4|{be}'] for be in ('tv_nms', 'torch_pure')}
    sb = lambda be, k: f'{TC}:arms.320|B4|{be}.{k}'  # noqa: E731
    lines.append('$\\Delta t_{\\mathrm{NMS}}$ of the gradient baseline, measured (ms) & ' + ' & '.join(t.n(b[be]['delta_nms_ms_mean'], sb(be, 'delta_nms_ms_mean'), sig=2) + ' (' + t.ci(*b[be]['delta_nms_ms_ci95'], sb(be, 'delta_nms_ms_ci95'), sig=2) + ')' for be in b) + ' \\\\')
    lines.append('Above $\\Delta t^{*}$ (320 px) & ' + ' & '.join('yes' if b[be]['delta_nms_ms_mean'] > tc['threshold_ms'][be]['320']['threshold_ms'] else 'no' for be in b) + ' \\\\')
    lines.append('$R_{\\mathrm{pred}} = 1+\\Delta t_{\\mathrm{NMS}}/t_{\\mathrm{total}}$, predicted & ' + ' & '.join(t.n(b[be]['r_pred'], sb(be, 'r_pred'), d=4) + ' (' + t.ci(*b[be]['r_pred_ci95'], sb(be, 'r_pred_ci95'), d=4) + ')' for be in b) + ' \\\\')
    body = r"""\begin{table}[t]
\centering\small
\caption{\textbf{Post hoc.} Threshold $\Delta t^{*}$ for the end-to-end ratio to exceed the MDE, and the prediction for the gradient baseline (B4, 320 px, 100 images $\times$ 5 seeds). $t_{\mathrm{total}}$ is the clean end-to-end time of the patch session itself (prepared tensors, no insertion step); the MDE comes from the separate A/A because this session has no two clean arms. $\Delta t_{\mathrm{NMS}}$ is a micro-benchmark measurement; $R_{\mathrm{pred}}$ is a prediction, not a measurement.}
\label{tab:threshold}
\begin{tabular}{@{}lcc@{}}
\toprule
 & torchvision \texttt{nms} & pure PyTorch \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


def t_x2():
    t = Tab('tab_x2')
    f = 'x2_counts/summary_posthoc.json'
    s = j(f)['detectors']
    name = dict(yolov8n='YOLOv8n', yolov8s='YOLOv8s', yolo11n='YOLO11n', yolo11s='YOLO11s', yolov12n='YOLOv12n')
    lines = []
    for d, v in s.items():
        sel = f'{f}:detectors.{d}'
        lines.append(f'{name[d]} & ' + t.n(v['images'], sel + '.images') + ' & ' + t.n(v['clean_median'], sel + '.clean_median', d=1) + ' & '
                     + t.n(v['patched_median'], sel + '.patched_median', d=1) + ' & ' + t.n(v['ratio_mean'], sel + '.ratio_mean', d=3)
                     + ' (' + t.ci(*v['ratio_ci95'], sel + '.ratio_ci95', d=3) + ') \\\\')
    body = r"""\begin{table}[t]
\centering\small
\caption{\textbf{Post hoc, historical data}: pre-NMS candidate counts ($\tau=0.01$, 320 px) of five YOLO detectors with five earlier SG-GA patches inserted (seeds 0--4, optimized on YOLOv8n in an earlier run, budget 2500 queries). Counts only, deterministic, no timing; no preregistration. Ratios are computed per image and then averaged; 95\% CI, two-level bootstrap over images $\times$ seeds. The image set differs from the 100 images of Table~\ref{tab:patches}. The two-stage and SSD detectors of the same run count after their internal NMS and are not reported.}
\label{tab:x2}
\begin{tabular}{@{}lcccc@{}}
\toprule
Detector & Images & Clean $\Ntau$ (median) & Patched $\Ntau$ (median) & $\Ntau$ ratio (95\% CI) \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table}\n'
    return write(t, body)


def t_decomp():
    t = Tab('tab_decomp')
    f = 'e10/E10_P1cpu2_20261102/stage_decomp_posthoc.json'
    s = j(f)['cells']
    lines = []
    for m, mv in (('overload', 'Overload'), ('phantom_sponges', 'Phantom Sponges')):
        for be in ('tv_nms', 'tv_batched', 'ultra_native'):
            v = s[f'{m}|{be}']
            sel = f'{f}:cells.{m}|{be}'
            lines.append(f'{mv} & {BE_EN[be]} & ' + t.n(v['d_total_ms'], sel + '.d_total_ms', sig=3) + ' (' + t.ci(*v['d_total_ci95'], sel + '.d_total_ci95', sig=2) + ') & '
                         + t.n(v['d_nms_ms'], sel + '.d_nms_ms', sig=3) + ' (' + t.ci(*v['d_nms_ci95'], sel + '.d_nms_ci95', sig=2) + ') & '
                         + t.n(v['d_rest_ms'], sel + '.d_rest_ms', sig=3) + ' (' + t.ci(*v['d_rest_ci95'], sel + '.d_rest_ci95', sig=2) + ') & '
                         + t.n(100 * v['rest_share_of_d_total'], sel + '.rest_share_of_d_total*100', d=0) + ' \\\\')
    body = r"""\begin{table*}[t]
\centering\small
\caption{\textbf{Post hoc}, YOLOv5s (640 px, one pinned CPU core): decomposition of the latency increase of the adaptive variants of the two whole-image attacks, per image $\times$ seed pair against the mean of the two clean arms of the same image (ms; 95\% CI, two-level bootstrap). $\Delta t_{\mathrm{NMS}}$ is measured as a micro-benchmark at $\tau=0.25$. The ``remainder'' $=\Delta t_{\mathrm{total}}-\Delta t_{\mathrm{NMS}}$ comprises the forward pass, decoding, score filtering and building the NMS input; the raw records do not separate these stages. No new measurement.}
\label{tab:decomp}
\begin{tabular}{@{}llcccc@{}}
\toprule
Variant & Implementation & $\Delta t_{\mathrm{total}}$ & $\Delta t_{\mathrm{NMS}}$ & Remainder & Remainder / $\Delta t_{\mathrm{total}}$ (\%) \\
\midrule
""" + '\n'.join(lines) + END + '\\end{table*}\n'
    return write(t, body)


def t_text():
    """Numbers used in the running text: the Vietnamese registry, with decimal points."""
    mt.Tab = Tab
    mt.t_text()
    REG['text'] = mt.REG['text']
    return 'text', None


BUILDERS = [t_aa, t_dose, t_cap, t_nstar, t_amdahl, t_attacks, t_spec, t_nk, t_patches, t_threshold, t_x2, t_decomp, t_text]


def main():
    check = '--check' in sys.argv
    tabs = dict(b() for b in BUILDERS)
    hdr = '% GENERATED by paper_en/tables_q3/make_tables_en.py -- DO NOT EDIT.\n'
    bad = []
    for name, body in tabs.items():
        if body is None:
            continue
        p = OUT / f'{name}.tex'
        new = hdr + body
        if check:
            if not p.exists() or p.read_text(encoding='utf-8') != new:
                bad.append(name)
        else:
            p.write_text(new, encoding='utf-8')
    if check:
        print('tables differing from a fresh build:', bad or 'none')
        sys.exit(1 if bad else 0)
    (OUT / 'numbers_en.json').write_text(json.dumps(REG, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', sum(1 for v in tabs.values() if v), 'tables,', sum(len(v) for v in REG.values()), 'numbers')


if __name__ == '__main__':
    main()
