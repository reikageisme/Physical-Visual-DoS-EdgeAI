"""Vẽ Hình 2 (đường dose-response E1) và Hình 3 (dự đoán chỉ-từ-N so với đo, E6) từ JSON/JSONL.
python make_figs.py E1_SUMMARY E6_RUN_DIR OUT_DIR   (chỉ đọc; ghi PNG/PDF vào OUT_DIR)"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

PROF = 'P1_cpu2'
BE = [('torch_pure', 'Pure PyTorch'), ('tv_nms', 'torchvision nms'), ('tv_batched', 'torchvision batched'), ('ultra_native', 'Ultralytics native')]
GEOM = [('disjoint', 'disjoint'), ('dense_cluster', 'dense cluster'), ('mixed', 'mixed')]
e1, run, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
curves = json.loads(e1.read_text(encoding='utf-8'))['curves']
out.mkdir(parents=True, exist_ok=True)

def curve(A, be, g, nc=80):
    pts = {}
    for k, v in curves.items():
        p, a, b, gg, c, cp, n = k.split('|')
        if p == PROF and int(a) == A and b == be and gg == g and int(c) == nc and cp == 'None':
            pts[int(n)] = v
    ns = sorted(n for n in pts if n > 0)
    return (np.array(ns), np.array([pts[n]['median_ms'] for n in ns]),
            np.array([pts[n]['ci95'][0] for n in ns]), np.array([pts[n]['ci95'][1] for n in ns]))

fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), sharex=True)
for a, (be, lab) in zip(ax, BE):
    for g, gl in GEOM:
        ns, m, lo, hi = curve(10000, be, g)
        a.plot(ns, m, marker='o', ms=3, label=gl); a.fill_between(ns, lo, hi, alpha=.2)
    a.set_xscale('log'); a.set_yscale('log'); a.set_title(lab, fontsize=9); a.set_xlabel(r'$N_\tau$')
ax[0].set_ylabel('NMS latency (ms)'); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / 'fig2_dose_response.pdf'); fig.savefig(out / 'fig2_dose_response.png', dpi=200)

# Hình 3
micro, nmap = defaultdict(list), {}
for line in next(run.glob('raw_*.jsonl')).read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    if r['kind'] == 'micro' and r['tau'] == 0.01:
        micro[(r['image_id'], r['arm'], r['seed'], r['backend'])].append(r['nms_ns'] / 1e6)
        nmap[(r['image_id'], r['arm'], r['seed'])] = r['nms_input']
def predict(ns, ms, n):
    return float(np.exp(np.interp(np.log(n), np.log(ns), np.log(ms))))
fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
for a, (be, lab) in zip(ax, BE):
    for grp, col in (('clean', 'tab:blue'), ('patched', 'tab:red')):
        xs, ys = [], []
        for (i, arm, s, b), v in micro.items():
            if b != be or arm in ('C2', 'C4') or (arm == 'clean') != (grp == 'clean'):
                continue
            n = nmap[(i, arm, s)]
            if n <= 0:
                continue
            ns, m, _, _ = curve(2100, be, 'mixed')
            xs.append(predict(ns, m, n)); ys.append(float(np.median(v)))
        a.scatter(xs, ys, s=3, alpha=.25, c=col, label=grp)
    lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
    a.plot(lim, lim, 'k--', lw=.8); a.set_xscale('log'); a.set_yscale('log')
    a.set_title(lab, fontsize=9); a.set_xlabel(r'predicted from $N_\tau$ (ms)'); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_ylabel('measured NMS (ms)'); ax[0].legend(fontsize=7, markerscale=3)
fig.tight_layout(); fig.savefig(out / 'fig3_pred_vs_measured.pdf'); fig.savefig(out / 'fig3_pred_vs_measured.png', dpi=200)
print('ok')
