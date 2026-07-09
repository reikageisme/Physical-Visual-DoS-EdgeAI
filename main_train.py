"""
main_train.py — Sponge Patch Training with GA (Gray-box)

Usage:
    python main_train.py [options]

Key options:
    --pop       : population size (default 30)
    --gen       : number of generations (default 50)
    --size      : patch pixel size (default 64)
    --area-pct  : patch as % of input frame area (overrides --size)
    --seed      : random seed for reproducibility
    --no-saliency : disable saliency-guided patch placement
    --crossover : crossover strategy [horizontal_split | uniform | two_point]
    --resolution: input frame resolution (e.g. 320 or 640)
"""

import torch
import cv2
import numpy as np
import argparse
import os
import json
import math
from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness
from attack.genetic_algo import SpongeGA


def compute_patch_size_from_pct(frame_h: int, frame_w: int, area_pct: float) -> int:
    """
    Convert area percentage to square patch pixel dimension.
    e.g. 4% of 320×320 = 0.04 * 320*320 = 4096 px² → side = 64px
    """
    total_pixels   = frame_h * frame_w
    patch_pixels   = total_pixels * (area_pct / 100.0)
    patch_side     = int(math.sqrt(patch_pixels))
    return max(patch_side, 8)   # minimum 8px


def main():
    parser = argparse.ArgumentParser(
        description="Train Sponge Patch via Saliency-Guided Genetic Algorithm"
    )
    parser.add_argument('--pop',        type=int,   default=30,
                        help='Population size (default: 30)')
    parser.add_argument('--gen',        type=int,   default=50,
                        help='Max generations (default: 50)')
    parser.add_argument('--size',       type=int,   default=64,
                        help='Patch pixel size (default: 64). Overridden by --area-pct')
    parser.add_argument('--area-pct',   type=float, default=None,
                        help='Patch area as %% of input frame (e.g. 4.0 for 4%%)')
    parser.add_argument('--resolution', type=int,   default=320,
                        help='Input frame resolution in pixels (square, default: 320)')
    parser.add_argument('--seed',       type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-saliency', action='store_true',
                        help='Disable saliency-guided patch placement (use center)')
    parser.add_argument('--crossover',  type=str,   default='horizontal_split',
                        choices=['horizontal_split', 'uniform', 'two_point'],
                        help='Crossover strategy (default: horizontal_split)')
    parser.add_argument('--mutation-rate', type=float, default=0.1,
                        help='Fraction of children mutated (default: 0.1)')
    parser.add_argument('--elite-k',    type=int,   default=5,
                        help='Number of elite individuals kept (default: 5)')
    parser.add_argument('--out-dir',    type=str,   default='outputs',
                        help='Output directory (default: outputs)')
    args = parser.parse_args()

    # ── Derived config ────────────────────────────────────────────────────────
    frame_h = frame_w = args.resolution

    if args.area_pct is not None:
        patch_size = compute_patch_size_from_pct(frame_h, frame_w, args.area_pct)
        actual_pct = (patch_size ** 2) / (frame_h * frame_w) * 100
        print(f"[Config] Patch area: {args.area_pct:.1f}% → {patch_size}×{patch_size}px "
              f"(actual {actual_pct:.2f}% of {frame_h}×{frame_w})")
    else:
        patch_size = args.size
        actual_pct = (patch_size ** 2) / (frame_h * frame_w) * 100
        print(f"[Config] Patch size: {patch_size}×{patch_size}px "
              f"({actual_pct:.2f}% of {frame_h}×{frame_w})")

    use_saliency = not args.no_saliency

    print(f"[Config] Pop={args.pop} | Gen={args.gen} | Seed={args.seed}")
    print(f"[Config] Crossover={args.crossover} | Saliency={'ON' if use_saliency else 'OFF'}")

    # ── CUDA benchmark ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")

    # ── Load victim model ─────────────────────────────────────────────────────
    print("\n[1] Loading victim model (YOLOv8n)...")
    victim = VictimModel()

    # ── Prepare base image ────────────────────────────────────────────────────
    print(f"[2] Preparing base frame ({frame_h}×{frame_w})...")
    base_image = torch.rand(
        (1, 3, frame_h, frame_w),
        dtype=torch.float32, device=victim.device
    )

    # ── Fitness wrapper ───────────────────────────────────────────────────────
    def evaluate_fitness(outputs, conf_thresh=0.01):
        return calculate_sponge_fitness(outputs, conf_thresh)

    # ── Initialize GA ─────────────────────────────────────────────────────────
    print("[3] Initializing Saliency-Guided GA...")
    ga = SpongeGA(
        patch_size       = patch_size,
        pop_size         = args.pop,
        generations      = args.gen,
        mutation_rate    = args.mutation_rate,
        crossover_mode   = args.crossover,
        elite_k          = args.elite_k,
        seed             = args.seed,
        use_saliency     = use_saliency,
    )

    # ── Run evolution ──────────────────────────────────────────────────────────
    best_patch_tensor = ga.evolve(victim, evaluate_fitness, base_image)
    summary = ga.get_run_summary()

    # ── Save patch ────────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = f"s{args.seed}_g{args.gen}_p{args.pop}_sz{patch_size}"

    patch_np  = best_patch_tensor.numpy()
    patch_np  = np.transpose(patch_np, (1, 2, 0))
    patch_np  = (patch_np * 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

    # Save at optimized size
    out_path_small = os.path.join(args.out_dir, f"sponge_patch_{run_tag}.png")
    cv2.imwrite(out_path_small, patch_bgr)
    print(f"\n[4] Saved optimized patch ({patch_size}×{patch_size}px): {out_path_small}")

    # Save A4 print version (2480×2480 @ 300DPI)
    patch_a4 = cv2.resize(patch_bgr, (2480, 2480), interpolation=cv2.INTER_CUBIC)
    out_path_a4 = os.path.join(args.out_dir, f"sponge_patch_A4_{run_tag}.png")
    cv2.imwrite(out_path_a4, patch_a4)
    print(f"[4] Saved A4 print patch (2480×2480): {out_path_a4}")

    # ── Save run summary JSON ─────────────────────────────────────────────────
    summary_out = os.path.join(args.out_dir, f"run_summary_{run_tag}.json")
    summary_serializable = {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in summary.items()
    }
    summary_serializable['patch_size_px']   = patch_size
    summary_serializable['patch_area_pct']  = round(actual_pct, 4)
    summary_serializable['frame_resolution']= f"{frame_h}x{frame_w}"
    summary_serializable['crossover_mode']  = args.crossover
    summary_serializable['use_saliency']    = use_saliency

    with open(summary_out, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    print(f"[5] Run summary saved: {summary_out}")
    print(f"\n=== DONE. Best fitness: {summary['best_fitness']:.2f} | "
          f"Converged at gen: {summary['gen_converged']} ===")


if __name__ == "__main__":
    main()