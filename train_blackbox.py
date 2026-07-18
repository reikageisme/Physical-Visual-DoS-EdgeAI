"""
train_blackbox.py — Black-box Sponge Patch Training via Observable Fitness

Threat Model: STRICT BLACK-BOX
  - No access to model architecture, weights, or gradients
  - No access to pre-NMS raw anchor scores
  - Only observes: inference latency, number of final detections (post-NMS)

This script demonstrates that Sponge Patch optimization is feasible even
under strict black-box constraints, using latency as the primary fitness signal.

Compare with:
  - main_train.py   : gray-box GA (uses pre-NMS scores)
  - fast_train.py   : white-box PGD (uses gradients)

Usage:
    python train_blackbox.py --pop 20 --gen 30 --seed 42
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import cv2

from core.victim_model import VictimModel
from core.sponge_fitness import ObservableFitness
from core.eot_transforms import apply_eot


class BlackboxGA:
    """
    Genetic Algorithm using only observable (black-box) fitness.

    Unlike the gray-box SpongeGA which reads pre-NMS scores,
    this variant queries the model and measures:
      - total inference latency
      - number of final detections (post-NMS)
    """

    def __init__(
        self,
        patch_size: int = 64,
        pop_size: int = 20,
        generations: int = 30,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.25,
        elite_k: int = 4,
        seed: int = None,
        n_queries_per_eval: int = 3,
    ):
        self.patch_size = patch_size
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_k = min(elite_k, pop_size)
        self.seed = seed
        self.n_queries_per_eval = n_queries_per_eval

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Initialize population
        self.population = torch.rand(
            (self.pop_size, 3, self.patch_size, self.patch_size),
            device=self.device, dtype=torch.float32,
        )

        # Tracking
        self.fitness_history = []
        self.mean_fit_history = []
        self.diversity_history = []
        self.gen_converged = None
        self.total_queries = 0

    def evolve(
        self,
        victim_model: VictimModel,
        obs_fitness: ObservableFitness,
        base_image: torch.Tensor,
        convergence_patience: int = 8,
        convergence_delta: float = 0.3,
    ) -> torch.Tensor:
        """
        Run black-box genetic optimization.

        Args:
            victim_model        : VictimModel instance
            obs_fitness         : ObservableFitness instance (calibrated)
            base_image          : [1, 3, H, W] float32 background image
            convergence_patience: early stopping patience
            convergence_delta   : min improvement to reset patience

        Returns:
            best_patch : [3, patch_size, patch_size] optimized patch tensor
        """
        _, _, H, W = base_image.shape

        # Place patch at center (no saliency needed for black-box)
        y_off = (H - self.patch_size) // 2
        x_off = (W - self.patch_size) // 2

        print(f"\n[BlackboxGA] Start black-box evolution:")
        print(f"  Pop={self.pop_size} | Gen={self.generations} | "
              f"Patch={self.patch_size}x{self.patch_size}px")
        print(f"  Queries per eval: {self.n_queries_per_eval}")
        print(f"  Seed: {self.seed} | Device: {self.device}\n")

        best_patch = self.population[0].clone()
        global_best_fit = -float('inf')
        patience_counter = 0

        for gen in range(self.generations):
            fitness_scores = []

            # Evaluate each individual sequentially (black-box = one at a time)
            for i in range(self.pop_size):
                patch = self.population[i]

                # Apply patch to base image
                adv_image = base_image.clone()
                adv_image[0, :,
                          y_off:y_off + self.patch_size,
                          x_off:x_off + self.patch_size] = patch

                # Apply EOT for robustness
                adv_image = apply_eot(adv_image)

                # Query model and measure observable fitness
                fitness, details = obs_fitness.compute(
                    victim_model, adv_image,
                    conf_thresh=0.25,
                    n_queries=self.n_queries_per_eval,
                )
                fitness_scores.append(fitness)
                self.total_queries += self.n_queries_per_eval

            # Stats
            fit_tensor = torch.tensor(fitness_scores, dtype=torch.float32)
            gen_best = float(fit_tensor.max())
            gen_mean = float(fit_tensor.mean())
            gen_std = float(fit_tensor.std())

            self.fitness_history.append(gen_best)
            self.mean_fit_history.append(gen_mean)
            self.diversity_history.append(gen_std)

            # Track global best
            best_idx = int(fit_tensor.argmax())
            if gen_best > global_best_fit + convergence_delta:
                global_best_fit = gen_best
                best_patch = self.population[best_idx].clone()
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Gen {gen+1:3d}/{self.generations} | "
                  f"Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | "
                  f"Std: {gen_std:6.2f} | Queries: {self.total_queries} | "
                  f"Patience: {patience_counter}/{convergence_patience}")

            # Early stopping
            if patience_counter >= convergence_patience:
                self.gen_converged = gen + 1
                print(f"\n[BlackboxGA] Converged at generation {self.gen_converged}!")
                break

            # Selection + crossover + mutation
            scores_t = torch.tensor(fitness_scores, device=self.device)
            _, top_idx = torch.topk(scores_t, self.elite_k)
            elites = self.population[top_idx]

            num_children = self.pop_size - self.elite_k
            p1_idx = top_idx[torch.randint(0, self.elite_k, (num_children,), device=self.device)]
            p2_idx = top_idx[torch.randint(0, self.elite_k, (num_children,), device=self.device)]

            # Uniform crossover
            mask = (torch.rand(num_children, 1, self.patch_size, self.patch_size,
                               device=self.device) > 0.5).float()
            children = mask * self.population[p1_idx] + (1 - mask) * self.population[p2_idx]

            # Centered mutation (unbiased)
            mut_mask = (torch.rand(num_children, device=self.device) < self.mutation_rate
                       ).view(-1, 1, 1, 1)
            noise = (torch.rand_like(children) - 0.5) * 2 * self.mutation_strength
            mutated = torch.clamp(children + noise, 0, 1)
            children = torch.where(mut_mask, mutated, children)

            self.population = torch.cat([elites, children], dim=0)

        if self.gen_converged is None:
            self.gen_converged = self.generations

        print(f"[BlackboxGA] Done. Best fitness: {global_best_fit:.2f}")
        print(f"[BlackboxGA] Total model queries: {self.total_queries}")
        return best_patch.cpu()

    def get_run_summary(self) -> dict:
        return {
            'seed': self.seed,
            'threat_model': 'strict_black_box',
            'gen_converged': self.gen_converged,
            'best_fitness': max(self.fitness_history) if self.fitness_history else 0,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'total_queries': self.total_queries,
            'fitness_history': self.fitness_history,
            'mean_fit_history': self.mean_fit_history,
            'diversity_history': self.diversity_history,
        }


def main():
    parser = argparse.ArgumentParser(
        description='Black-box Sponge Patch training using observable fitness (latency-based)')
    parser.add_argument('--pop', type=int, default=20, help='Population size')
    parser.add_argument('--gen', type=int, default=30, help='Max generations')
    parser.add_argument('--size', type=int, default=64, help='Patch size (px)')
    parser.add_argument('--resolution', type=int, default=320, help='Input resolution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-queries', type=int, default=3, help='Queries per fitness eval')
    parser.add_argument('--out-dir', type=str, default='outputs/blackbox', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    print("[*] Loading victim model...")
    victim = VictimModel()

    # Create synthetic base image
    frame_h, frame_w = args.resolution, args.resolution
    base_image = torch.rand(1, 3, frame_h, frame_w, device=victim.device, dtype=torch.float32)

    # Calibrate baseline latency on clean image
    print("[*] Calibrating baseline latency...")
    obs_fitness = ObservableFitness(
        latency_weight=2.0,
        det_count_weight=1.0,
        n_warmup_queries=5,
    )
    obs_fitness.calibrate_baseline(victim, base_image)

    # Run black-box GA
    print("[*] Starting black-box GA optimization...")
    t_start = time.perf_counter()

    bbga = BlackboxGA(
        patch_size=args.size,
        pop_size=args.pop,
        generations=args.gen,
        seed=args.seed,
        n_queries_per_eval=args.n_queries,
    )
    best_patch = bbga.evolve(victim, obs_fitness, base_image)

    t_elapsed = time.perf_counter() - t_start
    print(f"\n[*] Training completed in {t_elapsed:.1f}s")

    # Save patch
    patch_np = best_patch.permute(1, 2, 0).numpy()
    patch_np = (patch_np * 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

    patch_path = os.path.join(args.out_dir, f"blackbox_patch_s{args.seed}.png")
    cv2.imwrite(patch_path, patch_bgr)

    # Save A4 print version
    patch_a4 = cv2.resize(patch_bgr, (2480, 2480), interpolation=cv2.INTER_CUBIC)
    a4_path = os.path.join(args.out_dir, f"blackbox_patch_A4_s{args.seed}.png")
    cv2.imwrite(a4_path, patch_a4)

    # Save summary
    summary = bbga.get_run_summary()
    summary['training_time_s'] = round(t_elapsed, 2)
    summary['patch_path'] = patch_path

    summary_path = os.path.join(args.out_dir, f"blackbox_summary_s{args.seed}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[*] Results saved to {args.out_dir}/")
    print(f"    Patch:   {patch_path}")
    print(f"    A4:      {a4_path}")
    print(f"    Summary: {summary_path}")

    # Compare with gray-box result if available
    graybox_summary = 'outputs/multi_seed/aggregate_stats.json'
    if os.path.exists(graybox_summary):
        with open(graybox_summary) as f:
            gb_stats = json.load(f)
        print(f"\n[*] Comparison with gray-box GA:")
        print(f"    Gray-box best fitness (mean): {gb_stats.get('best_fitness', {}).get('mean', 'N/A')}")
        print(f"    Black-box best fitness:       {summary['best_fitness']:.2f}")
        print(f"    Black-box total queries:      {summary['total_queries']}")


if __name__ == '__main__':
    main()
