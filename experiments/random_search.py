"""
experiments/random_search.py — Random Search Baseline for Sponge Patch
──────────────────────────────────────────────────────────────────────
Fair comparison baseline:
  - Same query budget as GA (pop_size × generations evaluations)
  - Same patch size and frame resolution
  - Multiple seeds for statistical rigor

Usage:
    python experiments/random_search.py --n-seeds 5 --n-evals 300 --out-dir outputs/random_search
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import time
import numpy as np
import torch

from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness


def run_random_search(
    victim: VictimModel,
    seed: int,
    n_evals: int = 300,
    patch_size: int = 64,
    resolution: int = 320,
) -> dict:
    """Run random search with a single seed. Returns summary dict."""
    rng = np.random.default_rng(seed)
    base_img = torch.rand(
        (1, 3, resolution, resolution),
        dtype=torch.float32, device=victim.device,
    )

    best_fit = 0.0
    fitness_history = []
    t_start = time.perf_counter()

    for i in range(n_evals):
        # Random patch [0, 1]
        patch = rng.uniform(0, 1, (1, 3, patch_size, patch_size)).astype(np.float32)
        patch_t = torch.from_numpy(patch).to(victim.device)

        # Random location
        y = rng.integers(0, resolution - patch_size)
        x = rng.integers(0, resolution - patch_size)

        img = base_img.clone()
        img[0, :, y:y + patch_size, x:x + patch_size] = patch_t

        with torch.no_grad():
            scores = victim.get_raw_predictions(img)
            fit, _ = calculate_sponge_fitness(scores, conf_thresh=0.01)

        fit_val = float(fit)
        if fit_val > best_fit:
            best_fit = fit_val
        fitness_history.append(best_fit)

    elapsed = time.perf_counter() - t_start

    return {
        'seed': seed,
        'best_fitness': best_fit,
        'n_evals': n_evals,
        'elapsed_s': round(elapsed, 2),
        'fitness_history': fitness_history,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Random Search Baseline for Sponge Patch')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--n-evals', type=int, default=300,
                        help='Evaluations per seed (match GA: pop*gen)')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=320)
    parser.add_argument('--out-dir', type=str, default='outputs/random_search')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[*] Loading victim model...")
    victim = VictimModel()

    print(f"\n[*] Running Random Search ({args.n_seeds} seeds, "
          f"{args.n_evals} evals/seed)")
    print(f"    Patch: {args.patch_size}x{args.patch_size} | "
          f"Resolution: {args.resolution}\n")

    all_results = []

    for seed in range(args.n_seeds):
        result = run_random_search(
            victim, seed,
            n_evals=args.n_evals,
            patch_size=args.patch_size,
            resolution=args.resolution,
        )
        all_results.append(result)
        print(f"  Seed {seed}: best_fitness = {result['best_fitness']:.2f} "
              f"({result['elapsed_s']:.1f}s)")

        # Save per-seed JSON
        seed_path = os.path.join(args.out_dir, f"seed_{seed}.json")
        with open(seed_path, 'w') as f:
            json.dump(result, f, indent=2)

    # Aggregate stats
    fitnesses = [r['best_fitness'] for r in all_results]
    aggregate = {
        'method': 'random_search',
        'n_seeds': args.n_seeds,
        'n_evals_per_seed': args.n_evals,
        'best_fitness': {
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses)),
        },
    }

    agg_path = os.path.join(args.out_dir, 'aggregate_stats.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Random Search Results: {aggregate['best_fitness']['mean']:.2f} "
          f"± {aggregate['best_fitness']['std']:.2f}")
    print(f"{'=' * 60}")
    print(f"\n[+] Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
