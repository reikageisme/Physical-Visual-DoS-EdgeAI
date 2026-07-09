"""
plot_results.py — Visualization Tools for Sponge Attack Results

Added (per REVIEWED.md):
  - Latency breakdown stacked bar chart (NMS vs forward vs preproc)
  - Multi-seed comparison (mean ± std shaded area)
  - Scenario comparison table visualization
  - Resource summary table as figure
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (safe for headless edge server)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'clean'     : '#2ecc71',
    'attack'    : '#e74c3c',
    'random'    : '#3498db',
    'physical'  : '#e67e22',
    'nms'       : '#c0392b',
    'forward'   : '#2980b9',
    'preproc'   : '#27ae60',
    'render'    : '#8e44ad',
}


# ─────────────────────────────────────────────────────────────────────────────
def plot_performance(csv_file: str, output_path: str = None):
    """
    Plot FPS + CPU over time from a resource log CSV.
    Includes NMS latency subplot if column exists.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[-] File not found: {csv_file}")
        return

    df['Frame'] = range(len(df))
    has_nms = 'NMS_ms' in df.columns

    n_rows = 3 if has_nms else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    fig.suptitle(f'Sponge Attack Performance — {os.path.basename(csv_file)}',
                 fontsize=14, weight='bold')

    # ── FPS ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(df['Frame'], df['FPS_Actual'], color=COLORS['attack'], lw=2, label='Actual FPS')
    ax.axhline(y=10.0, color=COLORS['clean'], linestyle='--', lw=1.5, label='Target FPS (10)')
    ax.fill_between(df['Frame'], df['FPS_Actual'], 10,
                    where=(df['FPS_Actual'] < 10),
                    color=COLORS['attack'], alpha=0.15, label='FPS deficit')
    ax.set_ylabel('FPS', fontsize=11)
    ax.set_ylim(0, max(15, df['FPS_Actual'].max() * 1.1))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── CPU ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(df['Frame'], df['CPU_Percent'], color=COLORS['random'], lw=2, label='CPU %')
    ax.set_ylabel('CPU %', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── NMS latency ───────────────────────────────────────────────────────────
    if has_nms:
        ax = axes[2]
        ax.plot(df['Frame'], df['NMS_ms'], color=COLORS['nms'], lw=1.5, label='NMS latency (ms)')
        ax.set_ylabel('NMS (ms)', fontsize=11)
        ax.set_xlabel('Frame', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Frame', fontsize=11)

    plt.tight_layout()
    _save(fig, csv_file, output_path, suffix='_perf')


# ─────────────────────────────────────────────────────────────────────────────
def plot_latency_breakdown(csv_file: str, output_path: str = None):
    """
    Stacked bar chart of per-stage latency breakdown.
    Shows: preproc / forward / conf_filter / nms / render
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"[-] File not found: {csv_file}")
        return

    required = ['Preproc_ms', 'Forward_ms', 'ConfFilter_ms', 'NMS_ms']
    if not all(c in df.columns for c in required):
        print(f"[-] CSV missing latency columns. Run with profile_latency=True.")
        return

    # Downsample for readability
    step = max(1, len(df) // 100)
    df   = df.iloc[::step].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    frames  = df.index

    bottom = np.zeros(len(df))
    for col, color, label in [
        ('Preproc_ms',    COLORS['preproc'],  'Preproc'),
        ('Forward_ms',    COLORS['forward'],  'Forward pass'),
        ('ConfFilter_ms', COLORS['random'],   'Conf filter'),
        ('NMS_ms',        COLORS['nms'],      'NMS'),
    ]:
        vals = df[col].fillna(0).values
        ax.bar(frames, vals, bottom=bottom, color=color, label=label, width=0.8)
        bottom += vals

    if 'Render_ms' in df.columns:
        vals   = df['Render_ms'].fillna(0).values
        ax.bar(frames, vals, bottom=bottom, color=COLORS['render'], label='Render', width=0.8)

    ax.set_xlabel('Frame (sampled)', fontsize=11)
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Per-Stage Latency Breakdown During Attack', fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, csv_file, output_path, suffix='_latency_breakdown')


# ─────────────────────────────────────────────────────────────────────────────
def plot_multi_seed(results_dir: str, output_path: str = None):
    """
    Plot mean ± std of fitness history across multiple seeds.
    Expects CSV files named seed_*.csv in results_dir.
    """
    files = sorted(glob.glob(os.path.join(results_dir, 'seed_*.csv')))
    if not files:
        print(f"[-] No seed_*.csv files found in {results_dir}")
        return

    all_fitness = []
    for f in files:
        df = pd.read_csv(f)
        if 'best_fitness' in df.columns:
            all_fitness.append(df['best_fitness'].values)

    if not all_fitness:
        print("[-] No 'best_fitness' column found in seed files.")
        return

    max_len = max(len(a) for a in all_fitness)
    padded  = np.array([
        np.pad(a, (0, max_len - len(a)), constant_values=a[-1])
        for a in all_fitness
    ])

    mean = padded.mean(axis=0)
    std  = padded.std(axis=0)
    gens = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, mean, color=COLORS['attack'], lw=2, label=f'Mean (n={len(files)} seeds)')
    ax.fill_between(gens, mean - std, mean + std,
                    color=COLORS['attack'], alpha=0.2, label='±1 std')

    for i, a in enumerate(all_fitness):
        ax.plot(np.arange(1, len(a)+1), a, lw=0.8, alpha=0.4, color=COLORS['random'])

    ax.set_xlabel('Generation', fontsize=11)
    ax.set_ylabel('Best Fitness Score', fontsize=11)
    ax.set_title(f'Multi-Seed Convergence (n={len(files)} runs)', fontsize=13, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = output_path or os.path.join(results_dir, 'multi_seed_convergence.png')
    fig.savefig(out, dpi=200)
    print(f"[+] Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def plot_scenario_comparison(csv_files: dict, output_path: str = None):
    """
    Bar chart comparing FPS and CPU across scenarios.

    Args:
        csv_files: {'Clean': path, 'Digital Attack': path, 'Physical': path, ...}
    """
    scenarios, fps_means, fps_stds, cpu_means, cpu_stds = [], [], [], [], []

    for name, path in csv_files.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[-] {path} not found, skipping.")
            continue
        scenarios.append(name)
        fps_means.append(df['FPS_Actual'].mean())
        fps_stds.append(df['FPS_Actual'].std())
        cpu_means.append(df['CPU_Percent'].mean())
        cpu_stds.append(df['CPU_Percent'].std())

    if not scenarios:
        return

    x   = np.arange(len(scenarios))
    w   = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x, fps_means, w, yerr=fps_stds, capsize=5,
            color=[COLORS['clean'] if 'Clean' in s else COLORS['attack'] for s in scenarios])
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios, rotation=15, ha='right')
    ax1.set_ylabel('FPS (mean ± std)'); ax1.set_title('FPS by Scenario')
    ax1.axhline(10, ls='--', color='gray', lw=1, label='Target FPS')
    ax1.legend(); ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, cpu_means, w, yerr=cpu_stds, capsize=5,
            color=[COLORS['clean'] if 'Clean' in s else COLORS['attack'] for s in scenarios])
    ax2.set_xticks(x); ax2.set_xticklabels(scenarios, rotation=15, ha='right')
    ax2.set_ylabel('CPU % (mean ± std)'); ax2.set_title('CPU Load by Scenario')
    ax2.set_ylim(0, 105); ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Scenario Comparison — Sponge Attack', fontsize=13, weight='bold')
    plt.tight_layout()

    out = output_path or 'outputs/scenario_comparison.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[+] Saved: {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, csv_path: str, output_path: str, suffix: str = ''):
    if output_path is None:
        base = os.path.basename(csv_path).replace('.csv', f'{suffix}.png')
        output_path = os.path.join(os.path.dirname(csv_path), base)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"[+] Saved: {output_path}")
    plt.close(fig)


def get_latest_log(log_dir: str = 'logs') -> str | None:
    files = glob.glob(os.path.join(log_dir, '*.csv'))
    return max(files, key=os.path.getctime) if files else None


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sponge Attack Result Visualizer")
    parser.add_argument('--file',      type=str, help='CSV log file path')
    parser.add_argument('--out',       type=str, help='Output PNG path')
    parser.add_argument('--breakdown', action='store_true',
                        help='Also generate latency breakdown chart')
    parser.add_argument('--multi-seed', type=str, dest='multi_seed',
                        help='Directory with seed_*.csv files for multi-seed plot')
    args = parser.parse_args()

    if args.multi_seed:
        plot_multi_seed(args.multi_seed, args.out)

    else:
        target = args.file or get_latest_log()
        if not target:
            print("[-] No CSV log found. Run simulate_edge_server.py first.")
        else:
            print(f"[*] Plotting: {target}")
            plot_performance(target, args.out)
            if args.breakdown:
                plot_latency_breakdown(target)
