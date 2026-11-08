"""Hình bổ sung cho bản thảo v3, vẽ từ các tệp tóm tắt (chỉ đọc). python make_figs_v3.py OUT_DIR

fig4_pipeline_nstar   : tỷ số toàn trình theo N khi tiêm vào toàn pipeline (E9, hộp rời nhau, không giới hạn), 320 và 640 px.
fig5_cap_effect       : độ trễ NMS ở N = 10^4 không giới hạn so với top-300 (E1).
fig6_patch_effects    : tỷ số N, NMS (torchvision) và toàn trình của các điều kiện patch 320 px so với MDE (E6).
fig7_training_budget  : fitness huấn luyện theo ngân sách, tương đối so với mốc 10.000 (sinh patch trên GPU).
fig8_historical       : tỷ số ứng viên của các cohort lịch sử (số chép từ bản thảo v2, xem docs/NGUON_SO_LIEU.md).
"""
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

SIM = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
BE = [('torch_pure', 'Pure PyTorch'), ('tv_nms', 'torchvision nms'), ('tv_batched', 'torchvision batched'), ('ultra_native', 'Ultralytics native')]
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    plt.close(fig)


# ---- fig4: E9 phần A
a = json.loads((SIM / 'results/e9/E9_P1cpu2_20261102/summary_part_a.json').read_text(encoding='utf-8'))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for k, cv in enumerate((320, 640)):
    mde = a['mde_total_pct'][str(cv)] if str(cv) in a['mde_total_pct'] else a['mde_total_pct'][cv]
    for be, lab in BE:
        s = a['ratios'][f'{cv}|{be}|disjoint|capNone']
        ns = sorted(int(n) for n in s)
        r = [s[str(n)]['ratio'] for n in ns]
        lo = [s[str(n)]['ci95'][0] for n in ns]
        hi = [s[str(n)]['ci95'][1] for n in ns]
        ax[k].plot(ns, r, marker='o', ms=3, label=lab)
        ax[k].fill_between(ns, lo, hi, alpha=.2)
    ax[k].axhline(1 + mde / 100, color='k', ls='--', lw=.8, label=f'1 + MDE ({mde:.2g}%)')
    ax[k].set_xscale('log'); ax[k].set_yscale('log')
    ax[k].set_xticks(ns); ax[k].set_xticklabels([str(n) for n in ns])
    ax[k].xaxis.set_minor_formatter(NullFormatter())
    ax[k].set_yticks([1, 1.5, 2, 3, 5, 7] if cv == 320 else [1, 1.5, 2, 3, 5, 10])
    ax[k].yaxis.set_major_formatter(ScalarFormatter()); ax[k].yaxis.set_minor_formatter(NullFormatter())
    ax[k].set_xlabel(r'injected $N_\tau$'); ax[k].set_title(f'{cv} px, disjoint boxes, no cap')
ax[0].set_ylabel('end-to-end latency ratio vs. $N_\\tau=0$')
ax[0].legend(fontsize=6)
ax[1].legend(handles=ax[1].get_lines()[-1:], fontsize=6)
save(fig, 'fig4_pipeline_nstar')

# ---- fig5: E1 cap
c = json.loads((SIM / 'results/e1/E1_P1cpu2_20261101/summary.json').read_text(encoding='utf-8'))['curves']
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
for k, (g, gl) in enumerate((('disjoint', 'disjoint'), ('mixed', 'mixed'), ('dense_cluster', 'dense cluster'))):
    x = np.arange(len(BE))
    none = [c[f'P1_cpu2|10000|{be}|{g}|80|None|10000']['median_ms'] for be, _ in BE]
    cap = [c[f'P1_cpu2|10000|{be}|{g}|80|300|10000']['median_ms'] for be, _ in BE]
    ax[k].bar(x - .2, none, .4, label='no cap')
    ax[k].bar(x + .2, cap, .4, label='top-300 cap')
    ax[k].set_yscale('log'); ax[k].set_xticks(x); ax[k].set_xticklabels([l.replace(' ', '\n', 1) for _, l in BE], fontsize=6)
    ax[k].set_title(f'{gl}, $N_\\tau=10^4$')
ax[0].set_ylabel('NMS latency (ms)'); ax[0].legend(fontsize=6)
save(fig, 'fig5_cap_effect')

# ---- fig6: E6 patch effects (320 px)
e6 = json.loads((SIM / 'results/e6/E6_P1cpu2_20261102/summary.json').read_text(encoding='utf-8'))
# baseline B1-B4: số gộp 100 ảnh (E6 + E6-A2, PREREG_E6_A2)
_pool = json.loads((SIM / 'results/e6/E6_pooled_baselines_100img/summary.json').read_text(encoding='utf-8'))['arms']
for _a in ('B1', 'B2', 'B3', 'B4'):
    e6['arms'][_a] = _pool[_a]
# toàn trình: KHÔNG dùng số đo (lỗi chèn patch ở E6, E11 lần 1 bị loại vì drift; DEVIATIONS 03/11) mà dùng DỰ ĐOÁN
# Amdahl 1 + dNMS/t_total (results/amdahl/threshold_check_posthoc.json: r_pred), torchvision nms và PyTorch thuần
_thr = json.loads((SIM / 'results/amdahl/threshold_check_posthoc.json').read_text(encoding='utf-8'))['arms']
for _a in ('C1', 'C3', 'B1', 'B2', 'B3', 'B4'):
    for _be in ('tv_nms', 'torch_pure'):
        _v = _thr[f'320|{_a}|{_be}']
        e6['arms'][_a][f'pred_{_be}'] = dict(ratio_mean=_v['r_pred'], ratio_ci95=_v['r_pred_ci95'])
arms = [('C1', 'GA, τ_obj 0.01'), ('C3', 'GA, τ_obj 0.25'), ('B1', 'GA centre'), ('B2', 'GA random loc.'), ('B3', 'random search'), ('B4', 'gradient (tanh)')]
mde_total = e6['mde_aa_pct']['total']
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for k, (key, title) in enumerate((('N_tau0.01', r'$N_\tau$ ($\tau$=0.01)'), ('nms_tau0.01_tv_nms', 'NMS (torchvision)'), ('pred', 'end-to-end'))):
    for i, (arm, lab) in enumerate(arms):
        col = 'tab:red' if arm == 'B4' else 'tab:blue'
        for kk, mk, dy in ((('pred_tv_nms', 'o', -.12), ('pred_torch_pure', 's', .12)) if key == 'pred' else ((key, 'o', 0),)):
            v = e6['arms'][arm][kk]
            r, (lo, hi) = v['ratio_mean'], v['ratio_ci95']
            ax[k].errorbar(r, i + dy, xerr=[[r - lo], [hi - r]], fmt=mk, ms=3, capsize=2, color=col, mfc=col if mk == 'o' else 'none')
    ax[k].axvline(1, color='k', lw=.8)
    if key == 'pred':
        ax[k].axvspan(1 - mde_total / 100, 1 + mde_total / 100, color='grey', alpha=.25)
        title = 'end-to-end, PREDICTED\n(o tv nms, □ PyTorch; grey ±MDE)'
    ax[k].set_title(title); ax[k].set_xlabel('patched / clean')
ax[0].set_yticks(range(len(arms))); ax[0].set_yticklabels([l for _, l in arms]); ax[0].invert_yaxis()
save(fig, 'fig6_patch_effects')

# ---- fig7: fitness theo ngân sách
R = SIM / 'results/e7_gpu/final/runs/e7'
fit = defaultdict(list)
for f in glob.glob(str(R / '*' / 'seed*' / 'checkpoint_q*.json')):
    d = json.loads(Path(f).read_text(encoding='utf-8'))
    fit[(Path(f).parts[-3], d['queries'])].append(d['best_training_fitness'])
conds = sorted({k[0] for k in fit})
# hai khung: điều kiện không gradient (thang 0–2%) và baseline gradient (thang 0–60%), để không bị B4 đè bẹp
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.1), gridspec_kw=dict(width_ratios=[1.6, 1]))
for cnd in conds:
    qs = sorted(q for (cc, q) in fit if cc == cnd)
    base = np.mean(fit[(cnd, qs[0])])
    ax = axs[1] if cnd.startswith('B4') else axs[0]
    ax.plot(range(len(qs)), [100 * (np.mean(fit[(cnd, q)]) / base - 1) for q in qs], marker='o', ms=2.5, lw=1,
            label=cnd.split('_')[0], color='black' if cnd.startswith('B4') else None)
for ax, title in zip(axs, ('gradient-free (GA, random search)', 'gradient (B4)')):
    ax.set_xticks(range(3)); ax.set_xticklabels(['10k', '25k', '62.5k'], fontsize=6)
    ax.set_title(title, fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=.3)
axs[0].set_ylabel('fitness change vs. 10k (%)', fontsize=7)
axs[0].legend(fontsize=5, ncol=2, loc='upper left', handlelength=1.2, frameon=False)
fig.supxlabel('budget (candidate–image forward passes)', fontsize=7)
save(fig, 'fig7_training_budget')

# ---- fig8: cohort lịch sử (số từ bản thảo v2, bảng tỷ số ứng viên tại 2500 lượt ảnh)
hist = {'128 images': {'SG-GA': (1.0278, .0083), 'GA centre': (1.0329, .0093), 'GA random loc.': (1.0519, .0092), 'random search': (1.0268, .0072)},
        '500 images': {'SG-GA': (1.0268, .0074), 'GA centre': (1.0561, .0123), 'GA random loc.': (1.0707, .0143), 'random search': (1.0250, .0040)}}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
for k, (coh, d) in enumerate(hist.items()):
    for i, (m, (mu, sd)) in enumerate(d.items()):
        ax[k].errorbar(mu, i, xerr=sd, fmt='o', ms=3, capsize=2)
    ax[k].axvline(1, color='k', lw=.8); ax[k].set_title(coh); ax[k].set_xlabel('candidate ratio (patched / clean)')
ax[0].set_yticks(range(4)); ax[0].set_yticklabels(list(hist['128 images'])); ax[0].invert_yaxis()
save(fig, 'fig8_historical')
print('ok', sorted(p.name for p in OUT.glob('*.pdf')))

# ---- fig9: tấn công toàn ảnh thật (E10)
e10 = json.loads((SIM / 'results/e10/E10_P1cpu2_20261102/summary.json').read_text(encoding='utf-8'))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 2.2]))
meths = [('overload', 'Overload'), ('phantom_sponges', 'Phantom Sponges')]
for i, (m, lab) in enumerate(meths):
    v = e10['n_ratio'][f'{m}|tau0.25']
    ax[0].errorbar(i, v['mean'], yerr=[[v['mean'] - v['ci95'][0]], [v['ci95'][1] - v['mean']]], fmt='o', capsize=3)
ax[0].set_yscale('log'); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels([l for _, l in meths], fontsize=7)
ax[0].set_xlim(-.5, 1.5); ax[0].set_ylabel(r'$N_\tau$ ratio (patched / clean)'); ax[0].set_title(r'candidates ($\tau$=0.25)')
pbe = [('tv_nms', 'tv nms'), ('tv_batched', 'tv batched'), ('ultra_native', 'Ultralytics')]
mde = max(e10['in_session_mde_pct'].values())
ax[1].axhspan(1, 1 + mde / 100, color='grey', alpha=.25)
for j, (m, lab) in enumerate(meths):
    for k, (be, bl) in enumerate(pbe):
        x = k + (j - .5) * .3
        t = e10['total_ratio'][f'{m}|{be}']
        ax[1].errorbar(x, t['mean'], yerr=[[t['mean'] - t['ci95'][0]], [t['ci95'][1] - t['mean']]], fmt='o', ms=4, capsize=2,
                       color='tab:blue' if j == 0 else 'tab:orange', label=lab if k == 0 else None)
        ax[1].plot(x, e10['amdahl'][f'{m}|{be}']['r_pred_mean'], 'x', color='k', ms=6, label='Amdahl prediction' if (j == 0 and k == 0) else None)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([b for _, b in pbe], fontsize=7)
ax[1].set_ylabel('end-to-end ratio'); ax[1].set_title('end-to-end latency (grey: in-session MDE)'); ax[1].legend(fontsize=6)
save(fig, 'fig9_attacks')
print('fig9 ok')
