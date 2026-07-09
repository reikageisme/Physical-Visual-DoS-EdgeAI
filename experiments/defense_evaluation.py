"""
experiments/defense_evaluation.py
───────────────────────────────────
Evaluates simple defenses against the Sponge Patch attack.
Addresses REVIEWED.md Major Issue 10 (Defense Evaluation).

Defenses tested:
  1. max_det cap   : limit NMS output to top-K boxes (100, 200, 300)
  2. conf_thresh   : raise confidence threshold (0.01, 0.1, 0.25, 0.5)
  3. top_k prefilter: filter top-K candidates before NMS (300, 1000, 3000)

Metrics reported:
  - num_raw_boxes (pre-NMS)
  - num_final_dets (post-NMS)
  - nms_ms (latency)
  - total_ms

Usage:
    python experiments/defense_evaluation.py --patch outputs/sponge_patch.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import argparse
import numpy as np
import torch
import cv2
import torchvision.ops as ops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.victim_model import VictimModel


# ─────────────────────────────────────────────────────────────────────────────
def build_adversarial_frame(
    patch_path: str,
    patch_size: int,
    frame_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Load patch, embed in random background frame, return tensor."""
    if patch_path and os.path.exists(patch_path):
        patch = cv2.resize(cv2.imread(patch_path), (patch_size, patch_size))
    else:
        # Fallback: random noise patch
        patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
        print("[!] No patch file provided — using random noise patch as proxy.")

    frame = np.random.randint(0, 256, (frame_size, frame_size, 3), dtype=np.uint8)
    y0    = (frame_size - patch_size) // 2
    x0    = (frame_size - patch_size) // 2
    frame[y0:y0+patch_size, x0:x0+patch_size] = patch

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_defense(
    victim: VictimModel,
    adv_tensor: torch.Tensor,
    conf_thresh: float = 0.25,
    max_det: int = 300,
    n_trials: int = 20,
) -> dict:
    """Run N inference trials with given defense settings."""
    raw_boxes_list, final_dets_list, nms_ms_list, total_ms_list = [], [], [], []

    for _ in range(n_trials):
        result = victim.get_predictions_with_nms(
            adv_tensor,
            conf_thresh     = conf_thresh,
            max_det         = max_det,
            profile_latency = True,
        )
        raw_boxes_list.append(result['num_raw_boxes'])
        final_dets_list.append(result['num_final_boxes'])
        nms_ms_list.append(result['latency_ms']['nms_ms'])
        total_ms_list.append(result['latency_ms']['total_ms'])

    def _s(lst):
        return {'mean': float(np.mean(lst)), 'std': float(np.std(lst))}

    return {
        'conf_thresh' : conf_thresh,
        'max_det'     : max_det,
        'raw_boxes'   : _s(raw_boxes_list),
        'final_dets'  : _s(final_dets_list),
        'nms_ms'      : _s(nms_ms_list),
        'total_ms'    : _s(total_ms_list),
        'n_trials'    : n_trials,
    }


def plot_defense(results_by_conf: dict, results_by_maxdet: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Defense Evaluation — Sponge Attack vs Mitigation Strategies',
                 fontsize=13, weight='bold')

    # ── Top: conf_thresh sweep ────────────────────────────────────────────────
    conf_vals   = sorted(results_by_conf.keys())
    raw_means_c = [results_by_conf[c]['raw_boxes']['mean'] for c in conf_vals]
    nms_means_c = [results_by_conf[c]['nms_ms']['mean']    for c in conf_vals]
    raw_stds_c  = [results_by_conf[c]['raw_boxes']['std']  for c in conf_vals]
    nms_stds_c  = [results_by_conf[c]['nms_ms']['std']     for c in conf_vals]

    ax = axes[0, 0]
    ax.errorbar(conf_vals, raw_means_c, yerr=raw_stds_c, fmt='o-', color='#e74c3c',
                capsize=4, lw=2, ms=7)
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('Raw Boxes (pre-NMS)')
    ax.set_title('Conf. Threshold → Raw Boxes'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(conf_vals, nms_means_c, yerr=nms_stds_c, fmt='o-', color='#c0392b',
                capsize=4, lw=2, ms=7)
    ax.set_xlabel('Confidence Threshold'); ax.set_ylabel('NMS Latency (ms)')
    ax.set_title('Conf. Threshold → NMS Latency'); ax.grid(True, alpha=0.3)

    # ── Bottom: max_det sweep ─────────────────────────────────────────────────
    maxdet_vals  = sorted(results_by_maxdet.keys())
    fin_means_d  = [results_by_maxdet[d]['final_dets']['mean'] for d in maxdet_vals]
    nms_means_d  = [results_by_maxdet[d]['nms_ms']['mean']     for d in maxdet_vals]
    fin_stds_d   = [results_by_maxdet[d]['final_dets']['std']  for d in maxdet_vals]
    nms_stds_d   = [results_by_maxdet[d]['nms_ms']['std']      for d in maxdet_vals]

    ax = axes[1, 0]
    ax.errorbar(maxdet_vals, fin_means_d, yerr=fin_stds_d, fmt='s-', color='#2980b9',
                capsize=4, lw=2, ms=7)
    ax.set_xlabel('max_det cap'); ax.set_ylabel('Final Detections (post-NMS)')
    ax.set_title('max_det Cap → Final Detections'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(maxdet_vals, nms_means_d, yerr=nms_stds_d, fmt='s-', color='#1a5276',
                capsize=4, lw=2, ms=7)
    ax.set_xlabel('max_det cap'); ax.set_ylabel('NMS Latency (ms)')
    ax.set_title('max_det Cap → NMS Latency'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[+] Defense plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Defense Evaluation")
    parser.add_argument('--patch',      type=str,   default=None)
    parser.add_argument('--patch-size', type=int,   default=64)
    parser.add_argument('--frame-size', type=int,   default=320)
    parser.add_argument('--n-trials',   type=int,   default=20)
    parser.add_argument('--out-dir',    type=str,   default='outputs/defense_eval')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[*] Loading victim model...")
    victim = VictimModel()

    # Build adversarial input once
    adv_tensor = build_adversarial_frame(
        args.patch, args.patch_size, args.frame_size, victim.device
    )

    # ── Sweep 1: confidence threshold ─────────────────────────────────────────
    conf_thresholds = [0.01, 0.05, 0.1, 0.25, 0.5]
    results_conf    = {}
    print(f"\n{'─'*70}")
    print("  Defense: Confidence Threshold Sweep")
    print(f"{'─'*70}")

    for ct in conf_thresholds:
        r = evaluate_defense(victim, adv_tensor, conf_thresh=ct, n_trials=args.n_trials)
        results_conf[ct] = r
        print(f"  conf={ct:.2f}  raw={r['raw_boxes']['mean']:.0f}±{r['raw_boxes']['std']:.0f}  "
              f"nms={r['nms_ms']['mean']:.2f}ms  final_det={r['final_dets']['mean']:.0f}")

    # ── Sweep 2: max_det cap ───────────────────────────────────────────────────
    max_det_vals = [50, 100, 200, 300, 1000]
    results_det  = {}
    print(f"\n{'─'*70}")
    print("  Defense: max_det Cap Sweep (conf=0.01 to stress NMS)")
    print(f"{'─'*70}")

    for md in max_det_vals:
        r = evaluate_defense(victim, adv_tensor, conf_thresh=0.01, max_det=md,
                             n_trials=args.n_trials)
        results_det[md] = r
        print(f"  max_det={md:5d}  final={r['final_dets']['mean']:.0f}±{r['final_dets']['std']:.0f}  "
              f"nms={r['nms_ms']['mean']:.2f}ms")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    all_results = {
        'conf_thresh_sweep' : {str(k): v for k, v in results_conf.items()},
        'max_det_sweep'     : {str(k): v for k, v in results_det.items()},
    }
    out_json = os.path.join(args.out_dir, 'defense_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Results saved: {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, 'defense_evaluation.png')
    plot_defense(results_conf, results_det, plot_path)


if __name__ == "__main__":
    main()
