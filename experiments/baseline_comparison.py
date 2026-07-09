"""
experiments/baseline_comparison.py
────────────────────────────────────
Compares Sponge Patch against baseline textures to verify it's special.
Addresses REVIEWED.md Major Issue 9 (Baseline Controls).

Baselines:
  1. Random RGB noise
  2. Checkerboard (high-frequency)
  3. Uniform solid color (random)
  4. Gaussian blur noise
  5. Adversarial Sponge Patch (from file or re-optimized)

For each texture: measures num_raw_boxes, num_final_dets, NMS latency, FPS.

Usage:
    python experiments/baseline_comparison.py --patch outputs/sponge_patch.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import numpy as np
import torch
import cv2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.victim_model import VictimModel


# ─────────────────────────────────────────────────────────────────────────────
def make_checkerboard(size: int, block: int = 8) -> np.ndarray:
    """Create a black-and-white checkerboard pattern."""
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(0, size, block):
        for j in range(0, size, block):
            if ((i // block) + (j // block)) % 2 == 0:
                board[i:i+block, j:j+block] = 255
    return board


def make_random_noise(size: int) -> np.ndarray:
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def make_solid_color(size: int) -> np.ndarray:
    color = np.random.randint(50, 200, 3)
    return np.full((size, size, 3), color, dtype=np.uint8)


def make_gaussian_noise(size: int) -> np.ndarray:
    base  = np.full((size, size, 3), 128, dtype=np.float32)
    noise = np.random.randn(size, size, 3) * 60
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def load_patch(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Patch not found: {path}")
    return cv2.resize(img, (size, size))


# ─────────────────────────────────────────────────────────────────────────────
def build_frame_with_patch(
    patch_bgr: np.ndarray,
    frame_size: int = 320,
    position: str = 'center',
) -> torch.Tensor:
    """Build a frame tensor with the patch placed at given position."""
    ph, pw = patch_bgr.shape[:2]
    frame  = np.random.randint(0, 256, (frame_size, frame_size, 3), dtype=np.uint8)

    if position == 'center':
        y0 = (frame_size - ph) // 2
        x0 = (frame_size - pw) // 2
    elif position == 'topleft':
        y0, x0 = 0, 0
    else:
        y0 = np.random.randint(0, max(1, frame_size - ph))
        x0 = np.random.randint(0, max(1, frame_size - pw))

    frame[y0:y0+ph, x0:x0+pw] = patch_bgr

    # BGR → RGB, HWC → CHW, [0,1]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def evaluate_texture(
    victim: VictimModel,
    texture_bgr: np.ndarray,
    frame_size: int = 320,
    conf_thresh: float = 0.25,
    n_trials: int = 20,
) -> dict:
    """
    Run N inference trials with the texture patched into frames.
    Return mean stats.
    """
    raw_boxes_list, final_dets_list, nms_ms_list, total_ms_list = [], [], [], []

    for _ in range(n_trials):
        tensor = build_frame_with_patch(texture_bgr, frame_size).to(victim.device)
        result = victim.get_predictions_with_nms(
            tensor, conf_thresh=conf_thresh, profile_latency=True
        )
        raw_boxes_list.append(result['num_raw_boxes'])
        final_dets_list.append(result['num_final_boxes'])
        nms_ms_list.append(result['latency_ms']['nms_ms'])
        total_ms_list.append(result['latency_ms']['total_ms'])

    def _s(lst):
        return {'mean': float(np.mean(lst)), 'std': float(np.std(lst))}

    return {
        'raw_boxes'  : _s(raw_boxes_list),
        'final_dets' : _s(final_dets_list),
        'nms_ms'     : _s(nms_ms_list),
        'total_ms'   : _s(total_ms_list),
        'n_trials'   : n_trials,
    }


# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(results: dict, out_path: str):
    names    = list(results.keys())
    raw_means = [results[n]['raw_boxes']['mean']  for n in names]
    raw_stds  = [results[n]['raw_boxes']['std']   for n in names]
    nms_means = [results[n]['nms_ms']['mean']     for n in names]
    nms_stds  = [results[n]['nms_ms']['std']      for n in names]

    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c']
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Baseline Texture Comparison — Sponge Patch vs Controls',
                 fontsize=13, weight='bold')

    ax1.bar(x, raw_means, yerr=raw_stds, capsize=5, color=colors[:len(names)], width=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha='right')
    ax1.set_ylabel('Raw Boxes (pre-NMS) — mean ± std')
    ax1.set_title('Anchor Activation Count')
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, nms_means, yerr=nms_stds, capsize=5, color=colors[:len(names)], width=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha='right')
    ax2.set_ylabel('NMS Latency (ms) — mean ± std')
    ax2.set_title('NMS Computational Cost')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[+] Comparison plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Baseline Texture Comparison")
    parser.add_argument('--patch',      type=str,   default=None,
                        help='Path to optimized sponge patch PNG')
    parser.add_argument('--patch-size', type=int,   default=64)
    parser.add_argument('--frame-size', type=int,   default=320)
    parser.add_argument('--conf',       type=float, default=0.25)
    parser.add_argument('--n-trials',   type=int,   default=30,
                        help='Number of inference trials per texture (default: 30)')
    parser.add_argument('--out-dir',    type=str,   default='outputs/baseline_comparison')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ps = args.patch_size

    # ── Build textures ────────────────────────────────────────────────────────
    textures = {
        'Random Noise'   : make_random_noise(ps),
        'Checkerboard'   : make_checkerboard(ps),
        'Solid Color'    : make_solid_color(ps),
        'Gaussian Noise' : make_gaussian_noise(ps),
    }

    if args.patch:
        try:
            textures['Sponge Patch'] = load_patch(args.patch, ps)
            print(f"[+] Loaded sponge patch: {args.patch}")
        except FileNotFoundError as e:
            print(f"[-] {e} — skipping Sponge Patch")

    # Save texture images for reference
    for name, tex in textures.items():
        tex_path = os.path.join(args.out_dir, f"texture_{name.replace(' ', '_')}.png")
        cv2.imwrite(tex_path, tex)

    # ── Load victim model ─────────────────────────────────────────────────────
    print("\n[*] Loading victim model...")
    victim = VictimModel()

    # ── Evaluate each texture ─────────────────────────────────────────────────
    results = {}
    print(f"\n{'─'*70}")
    print(f"  {'Texture':20s}  {'Raw Boxes':>12}  {'NMS (ms)':>12}  {'Final Dets':>12}")
    print(f"{'─'*70}")

    for name, texture in textures.items():
        print(f"  Evaluating: {name} ...")
        stats = evaluate_texture(
            victim, texture,
            frame_size  = args.frame_size,
            conf_thresh = args.conf,
            n_trials    = args.n_trials,
        )
        results[name] = stats
        rb = stats['raw_boxes']
        nm = stats['nms_ms']
        fd = stats['final_dets']
        print(f"  {name:20s}  {rb['mean']:>8.1f}±{rb['std']:.1f}  "
              f"  {nm['mean']:>8.2f}±{nm['std']:.2f}  "
              f"  {fd['mean']:>8.1f}±{fd['std']:.1f}")

    print(f"{'─'*70}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, 'comparison_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, 'baseline_comparison.png')
    plot_comparison(results, plot_path)


if __name__ == "__main__":
    main()
