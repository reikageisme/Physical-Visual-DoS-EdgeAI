"""
experiments/multi_seed_experiment.py
─────────────────────────────────────
Runs Sponge Patch GA over multiple seeds and reports mean ± std.
Addresses REVIEWED.md Major Issue 5 (Statistical Rigor).

Usage:
    python experiments/multi_seed_experiment.py [options]

Output:
    outputs/multi_seed/seed_<N>.json        per-seed run summary
    outputs/multi_seed/aggregate_stats.json mean ± std across seeds
    outputs/multi_seed/multi_seed_conv.png  convergence plot
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import math
import argparse
import statistics
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness
from attack.genetic_algo import SpongeGA


def run_single_seed(
    victim,
    seed: int,
    patch_size: int,
    pop_size: int,
    generations: int,
    frame_res: int,
    use_saliency: bool,
) -> dict:
    """Run one GA optimization with a specific seed."""
    print(f"\n{'='*60}")
    print(f"  Seed {seed}")
    print(f"{'='*60}")

    base_image = torch.rand(
        (1, 3, frame_res, frame_res),
        dtype=torch.float32, device=victim.device
    )

    ga = SpongeGA(
        patch_size     = patch_size,
        pop_size       = pop_size,
        generations    = generations,
        seed           = seed,
        use_saliency   = use_saliency,
    )

    def fitness_fn(scores, conf_thresh=0.01):
        return calculate_sponge_fitness(scores, conf_thresh)

    ga.evolve(victim, fitness_fn, base_image)
    return ga.get_run_summary()


def compute_aggregate_stats(summaries: list[dict]) -> dict:
    """Compute mean ± std across runs for key metrics."""
    keys = ['best_fitness', 'gen_converged', 'final_diversity']
    agg  = {}

    for k in keys:
        vals = [s[k] for s in summaries if k in s and s[k] is not None]
        if vals:
            agg[k] = {
                'mean'   : statistics.mean(vals),
                'std'    : statistics.stdev(vals) if len(vals) > 1 else 0.0,
                'min'    : min(vals),
                'max'    : max(vals),
                'values' : vals,
                'n'      : len(vals),
            }

    return agg


def plot_multi_seed(summaries: list[dict], out_path: str):
    """Plot fitness convergence curves with mean ± std band."""
    all_histories = [s['fitness_history'] for s in summaries if s.get('fitness_history')]
    if not all_histories:
        return

    max_len = max(len(h) for h in all_histories)
    padded  = np.array([
        h + [h[-1]] * (max_len - len(h))
        for h in all_histories
    ])

    mean = padded.mean(axis=0)
    std  = padded.std(axis=0)
    gens = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, h in enumerate(all_histories):
        ax.plot(np.arange(1, len(h)+1), h, lw=0.8, alpha=0.4, color='#3498db',
                label='Individual seeds' if i == 0 else None)

    ax.plot(gens, mean, lw=2.5, color='#e74c3c', label=f'Mean (n={len(all_histories)})')
    ax.fill_between(gens, mean-std, mean+std, color='#e74c3c', alpha=0.2, label='±1 std')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness Score', fontsize=12)
    ax.set_title(f'Multi-Seed GA Convergence (n={len(all_histories)} seeds)', fontsize=13, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[+] Convergence plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed GA Experiment")
    parser.add_argument('--n-seeds',    type=int,   default=10,
                        help='Number of seeds to run (default: 10)')
    parser.add_argument('--seeds',      type=int,   nargs='+', default=None,
                        help='Specific seed values (overrides --n-seeds)')
    parser.add_argument('--pop',        type=int,   default=20)
    parser.add_argument('--gen',        type=int,   default=30)
    parser.add_argument('--size',       type=int,   default=64,
                        help='Patch size in pixels (default: 64)')
    parser.add_argument('--resolution', type=int,   default=320)
    parser.add_argument('--no-saliency', action='store_true')
    parser.add_argument('--out-dir',    type=str,   default='outputs/multi_seed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seeds = args.seeds if args.seeds else list(range(args.n_seeds))
    print(f"[Multi-seed] Seeds: {seeds}")
    print(f"[Multi-seed] Pop={args.pop} | Gen={args.gen} | Patch={args.size}px")

    print("\n[1] Loading victim model...")
    victim = VictimModel()

    summaries = []
    for seed in seeds:
        summary = run_single_seed(
            victim       = victim,
            seed         = seed,
            patch_size   = args.size,
            pop_size     = args.pop,
            generations  = args.gen,
            frame_res    = args.resolution,
            use_saliency = not args.no_saliency,
        )
        summaries.append(summary)

        # Save per-seed JSON
        seed_path = os.path.join(args.out_dir, f"seed_{seed}.json")
        with open(seed_path, 'w') as f:
            json.dump({
                k: (v if not isinstance(v, (list, tuple)) or len(v) < 1000 else v)
                for k, v in summary.items()
            }, f, indent=2)
        print(f"  → Saved: {seed_path}")

    # ── Aggregate statistics ───────────────────────────────────────────────────
    agg = compute_aggregate_stats(summaries)
    agg_path = os.path.join(args.out_dir, 'aggregate_stats.json')
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Multi-seed Results (n={len(seeds)} seeds)")
    print(f"{'='*60}")
    for metric, stats in agg.items():
        print(f"  {metric:25s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"[min={stats['min']:.3f}, max={stats['max']:.3f}]")
    print(f"{'='*60}")
    print(f"  Aggregate saved: {agg_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, 'multi_seed_convergence.png')
    plot_multi_seed(summaries, plot_path)


if __name__ == "__main__":
    main()
