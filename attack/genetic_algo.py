"""
genetic_algo.py — Saliency-Guided Genetic Algorithm for Sponge Patch Optimization

Changes from original (per REVIEWED.md):
  - Added seed control for reproducibility (Major Issue 5)
  - Added convergence tracking: fitness_history, diversity_history
  - Patch location: center (default) or saliency-guided (image-based, no model gradient)
  - Patch size now specified as pixel size or percentage of frame
  - Multiple crossover modes: horizontal_split (original), uniform, two_point
  - Logs: generation stats, diversity, convergence criterion
"""

import torch
import numpy as np
import cv2
from core.eot_transforms import apply_eot


class SpongeGA:
    """
    Genetic Algorithm for optimizing adversarial Sponge Patches.

    Threat model: Gray-box — fitness computed on pre-NMS scores.
    Saliency: Image-based (Laplacian variance) — does NOT use model gradients.
    """

    def __init__(
        self,
        patch_size: int = 64,
        pop_size: int = 30,
        generations: int = 50,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        crossover_mode: str = 'horizontal_split',  # 'horizontal_split' | 'uniform' | 'two_point'
        elite_k: int = 5,
        seed: int = None,
        use_saliency: bool = True,
        convergence_patience: int = 10,
        convergence_delta: float = 0.5,
    ):
        """
        Args:
            patch_size          : pixel dimension of the square patch
            pop_size            : population size
            generations         : max generations
            mutation_rate       : fraction of children that undergo mutation
            mutation_strength   : noise scale for mutation (0–1)
            crossover_mode      : crossover strategy
            elite_k             : number of elites kept each generation
            seed                : random seed for reproducibility (None = random)
            use_saliency        : if True, place patch at most salient region
            convergence_patience: stop early if best fitness doesn't improve
            convergence_delta   : minimum improvement to reset patience counter
        """
        self.patch_size          = patch_size
        self.pop_size            = pop_size
        self.generations         = generations
        self.mutation_rate       = mutation_rate
        self.mutation_strength   = mutation_strength
        self.crossover_mode      = crossover_mode
        self.elite_k             = min(elite_k, pop_size)
        self.seed                = seed
        self.use_saliency        = use_saliency
        self.convergence_patience = convergence_patience
        self.convergence_delta   = convergence_delta

        # ── Device ───────────────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Seed ─────────────────────────────────────────────────────────────
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"[SpongeGA] Seed set to {seed}")

        # ── Population initialisation ─────────────────────────────────────────
        self.population = torch.rand(
            (self.pop_size, 3, self.patch_size, self.patch_size),
            device=self.device, dtype=torch.float32
        )

        # ── Tracking ─────────────────────────────────────────────────────────
        self.fitness_history   = []   # best fitness per generation
        self.mean_fit_history  = []   # mean fitness per generation
        self.diversity_history = []   # population std (diversity metric)
        self.gen_converged     = None # generation at convergence

        # ── Saliency cache ────────────────────────────────────────────────────
        self._patch_location   = None  # (y_offset, x_offset) — set in evolve()

    # ─────────────────────────────────────────────────────────────────────────
    # Saliency-guided patch placement (image-based, NO model gradient)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_saliency_location(self, base_image: torch.Tensor) -> tuple[int, int]:
        """
        Compute optimal patch location using image-based saliency (Laplacian variance).
        This approach does NOT require model gradient access — compatible with gray-box.

        Returns:
            (y_offset, x_offset) for top-left corner of patch placement
        """
        _, _, H, W = base_image.shape

        # Convert to numpy grayscale for Laplacian
        img_np = base_image[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Sliding window Laplacian variance (higher = more texture/edges)
        step = max(self.patch_size // 4, 8)
        best_score = -1
        best_y, best_x = (H - self.patch_size) // 2, (W - self.patch_size) // 2

        for y in range(0, H - self.patch_size, step):
            for x in range(0, W - self.patch_size, step):
                region = gray[y:y + self.patch_size, x:x + self.patch_size]
                lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
                if lap_var > best_score:
                    best_score = lap_var
                    best_y, best_x = y, x

        print(f"[SpongeGA] Saliency: best patch location = ({best_y}, {best_x}), "
              f"Laplacian score = {best_score:.2f}")
        return best_y, best_x

    def apply_patch_batch(
        self,
        base_image_batch: torch.Tensor,
        patch_batch: torch.Tensor,
        y_offset: int,
        x_offset: int,
    ) -> torch.Tensor:
        """Paste patch_batch onto base_image_batch at (y_offset, x_offset)."""
        adv_images = base_image_batch.clone()
        adv_images[
            :, :,
            y_offset : y_offset + self.patch_size,
            x_offset : x_offset + self.patch_size
        ] = patch_batch
        return adv_images

    # ─────────────────────────────────────────────────────────────────────────
    # Crossover
    # ─────────────────────────────────────────────────────────────────────────
    def _crossover(self, parents1: torch.Tensor, parents2: torch.Tensor) -> torch.Tensor:
        n = parents1.shape[0]

        if self.crossover_mode == 'horizontal_split':
            split = self.patch_size // 2
            children = parents1.clone()
            children[:, :, split:, :] = parents2[:, :, split:, :]

        elif self.crossover_mode == 'uniform':
            mask = (torch.rand(n, 1, self.patch_size, self.patch_size,
                               device=self.device) > 0.5).float()
            children = mask * parents1 + (1 - mask) * parents2

        elif self.crossover_mode == 'two_point':
            p1 = torch.randint(0, self.patch_size, (1,)).item()
            p2 = torch.randint(p1 + 1, self.patch_size + 1, (1,)).item()
            children = parents1.clone()
            children[:, :, p1:p2, :] = parents2[:, :, p1:p2, :]

        else:
            raise ValueError(f"Unknown crossover_mode: {self.crossover_mode}")

        return children

    # ─────────────────────────────────────────────────────────────────────────
    # Main evolution loop
    # ─────────────────────────────────────────────────────────────────────────
    def evolve(
        self,
        victim_model,
        fitness_function,
        base_image: torch.Tensor,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Run genetic optimization.

        Args:
            victim_model     : VictimModel instance
            fitness_function : callable(batch_scores, conf_thresh) → (fitness, n_boxes)
            base_image       : [1, 3, H, W] float32 background image
            batch_size       : mini-batch size for GPU evaluation

        Returns:
            best_patch : [3, H_patch, W_patch] optimized patch tensor (float32, CPU)
        """
        _, _, H, W = base_image.shape

        # ── Compute patch location ────────────────────────────────────────────
        if self.use_saliency:
            y_off, x_off = self._compute_saliency_location(base_image)
        else:
            y_off = (H - self.patch_size) // 2
            x_off = (W - self.patch_size) // 2

        self._patch_location = (y_off, x_off)

        print(f"\n[SpongeGA] Start evolution:")
        print(f"  Pop={self.pop_size} | Gen={self.generations} | "
              f"Patch={self.patch_size}×{self.patch_size}px "
              f"({self.patch_size*self.patch_size/(H*W)*100:.1f}% of {H}×{W} frame)")
        print(f"  Patch location: y={y_off}, x={x_off}")
        print(f"  Crossover: {self.crossover_mode} | Elite-k: {self.elite_k}")
        print(f"  Seed: {self.seed}")
        print(f"  Device: {self.device}\n")

        best_patch       = self.population[0].clone()
        global_best_fit  = -float('inf')
        patience_counter = 0

        base_expanded = base_image.expand(batch_size, -1, -1, -1)  # logical view

        for gen in range(self.generations):
            fitness_scores = []

            with torch.no_grad():
                for i in range(0, self.pop_size, batch_size):
                    end_idx  = min(i + batch_size, self.pop_size)
                    curr_bs  = end_idx - i
                    base_chunk  = base_expanded[:curr_bs]
                    patch_chunk = self.population[i:end_idx]

                    adv_images  = self.apply_patch_batch(
                        base_chunk, patch_chunk, y_off, x_off
                    )
                    eot_images  = apply_eot(adv_images)
                    raw_scores  = victim_model.get_raw_predictions(eot_images)

                    chunk_fit, _ = fitness_function(raw_scores, conf_thresh=0.01)
                    fitness_scores.extend(chunk_fit.tolist())

                    del adv_images, eot_images, raw_scores, base_chunk, patch_chunk

            # ── Stats ─────────────────────────────────────────────────────────
            fit_tensor = torch.tensor(fitness_scores, dtype=torch.float32)
            gen_best   = float(fit_tensor.max())
            gen_mean   = float(fit_tensor.mean())
            gen_std    = float(fit_tensor.std())           # diversity proxy

            self.fitness_history.append(gen_best)
            self.mean_fit_history.append(gen_mean)
            self.diversity_history.append(gen_std)

            # ── Track global best ──────────────────────────────────────────────
            best_idx  = int(fit_tensor.argmax())
            if gen_best > global_best_fit + self.convergence_delta:
                global_best_fit  = gen_best
                best_patch       = self.population[best_idx].clone()
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Gen {gen+1:3d}/{self.generations} | "
                  f"Best: {gen_best:8.2f} | Mean: {gen_mean:8.2f} | "
                  f"Std: {gen_std:6.2f} | Patience: {patience_counter}/{self.convergence_patience}")

            # ── Early stopping ─────────────────────────────────────────────────
            if patience_counter >= self.convergence_patience:
                self.gen_converged = gen + 1
                print(f"\n[SpongeGA] Converged at generation {self.gen_converged}!")
                break

            # ── Selection ─────────────────────────────────────────────────────
            scores_t   = torch.tensor(fitness_scores, device=self.device)
            _, top_idx = torch.topk(scores_t, self.elite_k)
            elites     = self.population[top_idx]

            num_children = self.pop_size - self.elite_k
            p1_idx = top_idx[torch.randint(0, self.elite_k, (num_children,), device=self.device)]
            p2_idx = top_idx[torch.randint(0, self.elite_k, (num_children,), device=self.device)]

            children = self._crossover(self.population[p1_idx], self.population[p2_idx])

            # ── Mutation ───────────────────────────────────────────────────────
            mut_mask = (torch.rand(num_children, device=self.device) < self.mutation_rate
                       ).view(-1, 1, 1, 1)
            noise    = (torch.rand_like(children) - 0.5) * 2 * self.mutation_strength
            mutated  = torch.clamp(children + noise, 0, 1)
            children = torch.where(mut_mask, mutated, children)

            self.population = torch.cat([elites, children], dim=0)

            del children, mutated, noise

        if self.gen_converged is None:
            self.gen_converged = self.generations
            print(f"\n[SpongeGA] Reached max generations ({self.generations}).")

        print(f"[SpongeGA] Done. Global best fitness: {global_best_fit:.2f}")
        return best_patch.cpu()

    # ─────────────────────────────────────────────────────────────────────────
    def get_run_summary(self) -> dict:
        """Return a dict with convergence statistics for reporting."""
        return {
            'seed'              : self.seed,
            'gen_converged'     : self.gen_converged,
            'best_fitness'      : max(self.fitness_history) if self.fitness_history else 0,
            'final_diversity'   : self.diversity_history[-1] if self.diversity_history else 0,
            'fitness_history'   : self.fitness_history,
            'mean_fit_history'  : self.mean_fit_history,
            'diversity_history' : self.diversity_history,
            'patch_location'    : self._patch_location,
        }