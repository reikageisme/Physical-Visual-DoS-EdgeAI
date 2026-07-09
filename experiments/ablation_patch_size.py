"""
experiments/ablation_patch_size.py
────────────────────────────────────
Ablation study: vary patch size from 1% to 16% of frame area.
Addresses REVIEWED.md Major Issue 4 (patch size contradiction).

For each patch area percentage:
  - Runs GA optimization
  - Reports: best_fitness, gen_converged, patch_px, area_pct

Usage:
    python experiments/ablation_patch_size.py [options]

Output:
    outputs/ablation_size/results.json
    outputs/ablation_size/ablation_patch_size.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import math
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness
from attack.genetic_algo import SpongeGA


AREA_PERCENTAGES = [1.0, 2.0, 4.0, 8.0, 16.0]


def pct_to_side(pct: float, frame_h: int, frame_w: int) -> int:
    total  = frame_h * frame_w
    pixels = total * (pct / 100.0)
    return max(int(math.sqrt(pixels)), 8)


def run_one_size(victim, patch_px: int, frame_res: int, pop: int, gen: int, seed: int) -> dict:
    base_image = torch.rand(
        (1, 3, frame_res, frame_res),
        dtype=torch.float32, device=victim.device
    )
    ga = SpongeGA(
        patch_size  = patch_px,
        pop_size    = pop,
        generations = gen,
        seed        = seed,
    )
    def fitness_fn(scores, conf_thresh=0.01):
        return calculate_sponge_fitness(scores, conf_thresh)

    ga.evolve(victim, fitness_fn, base_image)
    return ga.get_run_summary()


def plot_ablation(results: list[dict], out_path: str):
    area_pcts     = [r['area_pct']    for r in results]
    best_fitness  = [r['best_fitness'] for r in results]
    gen_conv      = [r['gen_converged'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Patch Size Ablation Study', fontsize=13, weight='bold')

    # ── Fitness vs area% ──────────────────────────────────────────────────────
    ax1.plot(area_pcts, best_fitness, 'o-', color='#e74c3c', lw=2, ms=8)
    for x, y, px in zip(area_pcts, best_fitness, [r['patch_px'] for r in results]):
        ax1.annotate(f"{px}px", (x, y), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xlabel('Patch Area (% of frame)', fontsize=11)
    ax1.set_ylabel('Best Fitness Score', fontsize=11)
    ax1.set_title('Fitness vs Patch Size')
    ax1.grid(True, alpha=0.3)

    # ── Convergence gen vs area% ──────────────────────────────────────────────
    ax2.bar(area_pcts, gen_conv, color='#3498db', width=0.8)
    ax2.set_xlabel('Patch Area (% of frame)', fontsize=11)
    ax2.set_ylabel('Generation at Convergence', fontsize=11)
    ax2.set_title('Convergence Speed vs Patch Size')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[+] Ablation plot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch Size Ablation Study")
    parser.add_argument('--resolution', type=int, default=320)
    parser.add_argument('--pop',        type=int, default=20)
    parser.add_argument('--gen',        type=int, default=25)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--pcts',       type=float, nargs='+',
                        default=AREA_PERCENTAGES,
                        help='Area percentages to test')
    parser.add_argument('--out-dir',    type=str, default='outputs/ablation_size')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_res = args.resolution

    print(f"[Ablation] Frame: {frame_res}×{frame_res}")
    print(f"[Ablation] Testing patch areas: {args.pcts}%")

    victim = VictimModel()
    results = []

    for pct in args.pcts:
        patch_px = pct_to_side(pct, frame_res, frame_res)
        actual_pct = (patch_px ** 2) / (frame_res ** 2) * 100

        print(f"\n{'─'*50}")
        print(f"  Patch area: {pct:.1f}% → {patch_px}×{patch_px}px (actual {actual_pct:.2f}%)")
        print(f"{'─'*50}")

        summary = run_one_size(victim, patch_px, frame_res, args.pop, args.gen, args.seed)
        result  = {
            'target_pct'   : pct,
            'area_pct'     : actual_pct,
            'patch_px'     : patch_px,
            'best_fitness' : summary['best_fitness'],
            'gen_converged': summary['gen_converged'],
            'final_diversity': summary['final_diversity'],
        }
        results.append(result)

        print(f"  → Fitness: {result['best_fitness']:.2f} | "
              f"Converged: gen {result['gen_converged']}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved: {out_json}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'Area%':>8}  {'Patch px':>10}  {'Best Fitness':>14}  {'Conv. Gen':>10}")
    print(f"{'─'*65}")
    for r in results:
        print(f"  {r['area_pct']:8.2f}%  {r['patch_px']:>8}px  "
              f"{r['best_fitness']:14.2f}  {r['gen_converged']:>10}")
    print(f"{'─'*65}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, 'ablation_patch_size.png')
    plot_ablation(results, plot_path)


if __name__ == "__main__":
    main()
