"""
simulate_edge_server.py
═══════════════════════════════════════════════════════════════════════════════
Complete Edge Server DoS Simulation Pipeline

Simulates a full IP Camera → Edge Server surveillance pipeline including:
  1. Clean baseline measurement
  2. Digital Sponge Patch attack
  3. Latency breakdown profiling (NMS vs forward vs preproc)
  4. Defense evaluation (optional)
  5. Result plotting and summary report

No physical camera required — uses synthetic frames (SyntheticFrameSource).

Hardware simulation modes:
  --edge-profile raspberry_pi  : 1 CPU thread, no GPU
  --edge-profile intel_nuc     : 2 CPU threads, no GPU
  --edge-profile jetson_nano   : 2 CPU threads, CUDA (limited)
  --edge-profile full_server   : all threads, full GPU

Usage:
    # Quick demo (60 frames clean + 60 frames attack)
    python simulate_edge_server.py --frames 60 --quick

    # Full experiment
    python simulate_edge_server.py --frames 200 --train --pop 20 --gen 20

    # With existing patch
    python simulate_edge_server.py --frames 200 --patch outputs/sponge_patch.png

    # Simulate Raspberry Pi constraints
    python simulate_edge_server.py --frames 100 --edge-profile raspberry_pi
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import cv2

from core.victim_model import VictimModel
from core.sponge_fitness import calculate_sponge_fitness
from attack.genetic_algo import SpongeGA
from utils.monitor import EdgeMonitor
from utils.plot_results import (
    plot_performance,
    plot_latency_breakdown,
    plot_scenario_comparison,
)

# ─────────────────────────────────────────────────────────────────────────────
# Edge Hardware Profiles (CPU thread count)
# ─────────────────────────────────────────────────────────────────────────────
EDGE_PROFILES = {
    'raspberry_pi' : {'threads': 1, 'force_cpu': True,  'desc': 'Raspberry Pi 4 (1 thread, CPU-only)'},
    'intel_nuc'    : {'threads': 2, 'force_cpu': True,  'desc': 'Intel NUC (2 threads, CPU-only)'},
    'jetson_nano'  : {'threads': 2, 'force_cpu': False, 'desc': 'Jetson Nano (2 threads, CUDA)'},
    'full_server'  : {'threads': 0, 'force_cpu': False, 'desc': 'Full server (all threads, GPU)'},
}


# ─────────────────────────────────────────────────────────────────────────────
class SyntheticIPCamera:
    """
    Simulates an IP camera stream generating random scene frames.
    Optionally overlays an adversarial Sponge Patch.
    """
    def __init__(
        self,
        height: int = 720,
        width: int  = 1280,
        patch_path: str = None,
        patch_size: int = 64,
        patch_tensor: torch.Tensor = None,
    ):
        self.H, self.W   = height, width
        self.patch_bgr   = None

        if patch_tensor is not None:
            # Convert torch tensor → numpy BGR
            p_np = patch_tensor.cpu().numpy()          # [3, H, W]
            p_np = np.transpose(p_np, (1, 2, 0))       # [H, W, 3]
            p_np = (p_np * 255).astype(np.uint8)
            p_bgr = cv2.cvtColor(p_np, cv2.COLOR_RGB2BGR)
            self.patch_bgr = cv2.resize(p_bgr, (patch_size, patch_size))

        elif patch_path and os.path.exists(patch_path):
            raw = cv2.imread(patch_path)
            self.patch_bgr = cv2.resize(raw, (patch_size, patch_size))

    def get_frame(self) -> np.ndarray:
        """Generate a synthetic camera frame (random scene + optional patch)."""
        # Realistic scene simulation: gradient background + random objects
        frame = np.random.randint(30, 220, (self.H, self.W, 3), dtype=np.uint8)

        # Add some structure (horizontal gradient for realism)
        for c in range(3):
            frame[:, :, c] = np.clip(
                frame[:, :, c].astype(int) +
                np.linspace(-30, 30, self.W).astype(int),
                0, 255
            )

        if self.patch_bgr is not None:
            ph, pw = self.patch_bgr.shape[:2]
            y0     = (self.H - ph) // 2
            x0     = (self.W - pw) // 2
            frame[y0:y0+ph, x0:x0+pw] = self.patch_bgr

        return frame


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    small = cv2.resize(frame, (size, size))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    t     = torch.from_numpy(rgb).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(
    victim: VictimModel,
    camera: SyntheticIPCamera,
    scenario_name: str,
    n_frames: int,
    yolo_res: int,
    conf_thresh: float,
    max_det: int,
    target_fps: float,
    log_dir: str,
) -> str:
    """
    Run one scenario (clean or attack) for N frames.
    Returns path to CSV log file.
    """
    monitor    = EdgeMonitor(log_dir=log_dir, scenario=scenario_name)
    FRAME_TIME = 1.0 / target_fps

    print(f"\n{'═'*65}")
    print(f"  Scenario: {scenario_name.upper()}")
    print(f"  Frames: {n_frames} | Resolution: {yolo_res}×{yolo_res} | conf={conf_thresh}")
    print(f"{'═'*65}")

    for i in range(n_frames):
        t0 = time.perf_counter()

        frame  = camera.get_frame()
        tensor = preprocess(frame, yolo_res, victim.device)

        result = victim.get_predictions_with_nms(
            tensor,
            conf_thresh     = conf_thresh,
            max_det         = max_det,
            profile_latency = True,
        )

        elapsed    = time.perf_counter() - t0
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        total_elapsed = max(time.perf_counter() - t0, 0.001)
        actual_fps    = 1.0 / total_elapsed

        monitor.log_status(
            current_fps     = actual_fps,
            latency_ms      = result['latency_ms'],
            num_raw_boxes   = result['num_raw_boxes'],
            num_final_boxes = result['num_final_boxes'],
        )

        if (i + 1) % max(1, n_frames // 10) == 0:
            raw_n = result['num_raw_boxes']
            fin_n = result['num_final_boxes']
            nms_t = result['latency_ms']['nms_ms']
            print(f"  [{i+1:5d}/{n_frames}] FPS:{actual_fps:5.1f} | "
                  f"CPU:{monitor._cpu_history[-1]:4.0f}% | "
                  f"RawBox:{raw_n:5d} | FinalDet:{fin_n:4d} | "
                  f"NMS:{nms_t:.2f}ms")

    monitor.print_summary()
    return monitor.log_file


# ─────────────────────────────────────────────────────────────────────────────
def train_patch(
    victim: VictimModel,
    patch_size: int,
    pop_size: int,
    generations: int,
    frame_res: int,
    seed: int,
    out_dir: str,
) -> tuple[torch.Tensor, str]:
    """Run GA to generate a Sponge Patch. Returns (patch_tensor, patch_path)."""
    print(f"\n{'═'*65}")
    print(f"  Training Sponge Patch (GA)")
    print(f"  Patch: {patch_size}×{patch_size}px | Pop: {pop_size} | Gen: {generations}")
    print(f"{'═'*65}")

    base_image = torch.rand(
        (1, 3, frame_res, frame_res),
        dtype=torch.float32, device=victim.device
    )

    ga = SpongeGA(
        patch_size   = patch_size,
        pop_size     = pop_size,
        generations  = generations,
        seed         = seed,
        use_saliency = True,
    )

    def fitness_fn(scores, conf_thresh=0.01):
        return calculate_sponge_fitness(scores, conf_thresh)

    best_patch = ga.evolve(victim, fitness_fn, base_image)
    summary    = ga.get_run_summary()

    os.makedirs(out_dir, exist_ok=True)
    patch_np  = best_patch.numpy()
    patch_np  = np.transpose(patch_np, (1, 2, 0))
    patch_np  = (patch_np * 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

    run_tag   = f"simulated_s{seed}_g{generations}_p{pop_size}_sz{patch_size}"
    patch_path = os.path.join(out_dir, f"sponge_patch_{run_tag}.png")
    cv2.imwrite(patch_path, patch_bgr)

    # Also save A4 version
    patch_a4 = cv2.resize(patch_bgr, (2480, 2480), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(out_dir, f"sponge_patch_A4_{run_tag}.png"), patch_a4)

    # Save run summary
    summ_path = os.path.join(out_dir, f"ga_summary_{run_tag}.json")
    with open(summ_path, 'w') as f:
        json.dump({k: (v if not isinstance(v, list) else v)
                   for k, v in summary.items()}, f, indent=2)

    print(f"\n[+] Patch saved: {patch_path}")
    print(f"[+] GA summary:  {summ_path}")
    return best_patch, patch_path


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Edge Server DoS Simulation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ────────────────────────────────────────────────────────────────
    parser.add_argument('--quick',    action='store_true',
                        help='Quick demo: 60 frames, pop=10, gen=10')
    parser.add_argument('--train',    action='store_true',
                        help='Train a new Sponge Patch with GA before testing')
    parser.add_argument('--patch',    type=str,   default=None,
                        help='Path to pre-trained patch PNG (skip training)')

    # ── Experiment params ────────────────────────────────────────────────────
    parser.add_argument('--frames',   type=int,   default=150,
                        help='Frames per scenario (default: 150)')
    parser.add_argument('--pop',      type=int,   default=20,
                        help='GA population size (default: 20)')
    parser.add_argument('--gen',      type=int,   default=20,
                        help='GA generations (default: 20)')
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--size',     type=int,   default=64,
                        help='Patch pixel size (default: 64)')
    parser.add_argument('--resolution', type=int, default=320,
                        help='YOLO input resolution (default: 320)')
    parser.add_argument('--conf',     type=float, default=0.25,
                        help='NMS confidence threshold (default: 0.25)')
    parser.add_argument('--max-det',  type=int,   default=300,
                        help='NMS max detections (default: 300)')
    parser.add_argument('--target-fps', type=float, default=10.0)

    # ── Edge simulation ──────────────────────────────────────────────────────
    parser.add_argument('--edge-profile', type=str, default='full_server',
                        choices=list(EDGE_PROFILES.keys()),
                        help='Edge hardware simulation profile (default: full_server)')

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument('--out-dir',  type=str,   default='outputs',
                        help='Output directory (default: outputs)')
    parser.add_argument('--log-dir',  type=str,   default='logs',
                        help='Log directory for CSV files (default: logs)')

    args = parser.parse_args()

    # ── Quick mode overrides ──────────────────────────────────────────────────
    if args.quick:
        args.frames  = 60
        args.pop     = 10
        args.gen     = 10
        print("[Quick Mode] frames=60, pop=10, gen=10")

    # ── Apply edge hardware profile ───────────────────────────────────────────
    profile = EDGE_PROFILES[args.edge_profile]
    print(f"\n{'═'*65}")
    print(f"  Edge Server Simulation")
    print(f"  Hardware profile: {profile['desc']}")
    print(f"{'═'*65}")

    if profile['threads'] > 0:
        torch.set_num_threads(profile['threads'])
        print(f"  CPU threads: {profile['threads']}")

    device = None
    if profile['force_cpu'] and torch.cuda.is_available():
        device = 'cpu'
        print(f"  Forcing CPU-only mode (simulating {args.edge_profile})")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load victim model ─────────────────────────────────────────────────────
    print(f"\n[1] Loading victim model...")
    victim = VictimModel(device=device)

    # ── Train or load patch ───────────────────────────────────────────────────
    patch_tensor = None
    patch_path   = args.patch

    if args.train or (not args.patch and not args.quick):
        patch_tensor, patch_path = train_patch(
            victim      = victim,
            patch_size  = args.size,
            pop_size    = args.pop,
            generations = args.gen,
            frame_res   = args.resolution,
            seed        = args.seed,
            out_dir     = args.out_dir,
        )
    elif args.quick and not args.patch:
        # Quick mode: use random noise patch as placeholder
        patch_tensor = torch.rand(3, args.size, args.size)
        print(f"[Quick] Using random noise patch ({args.size}×{args.size}px)")

    # ── SCENARIO 1: Clean baseline ────────────────────────────────────────────
    print(f"\n[2] Running CLEAN BASELINE scenario...")
    clean_cam    = SyntheticIPCamera(patch_path=None)
    log_clean    = run_scenario(
        victim       = victim,
        camera       = clean_cam,
        scenario_name= 'clean',
        n_frames     = args.frames,
        yolo_res     = args.resolution,
        conf_thresh  = args.conf,
        max_det      = args.max_det,
        target_fps   = args.target_fps,
        log_dir      = args.log_dir,
    )

    # ── SCENARIO 2: Digital attack ────────────────────────────────────────────
    print(f"\n[3] Running DIGITAL ATTACK scenario...")
    attack_cam   = SyntheticIPCamera(
        patch_path   = patch_path,
        patch_size   = args.size,
        patch_tensor = patch_tensor,
    )
    log_attack   = run_scenario(
        victim       = victim,
        camera       = attack_cam,
        scenario_name= 'digital_attack',
        n_frames     = args.frames,
        yolo_res     = args.resolution,
        conf_thresh  = args.conf,
        max_det      = args.max_det,
        target_fps   = args.target_fps,
        log_dir      = args.log_dir,
    )

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"\n[4] Generating plots...")

    # Per-scenario performance
    plot_performance(log_clean,  os.path.join(args.out_dir, 'plot_clean.png'))
    plot_performance(log_attack, os.path.join(args.out_dir, 'plot_attack.png'))

    # Latency breakdown
    plot_latency_breakdown(log_clean,  os.path.join(args.out_dir, 'latency_clean.png'))
    plot_latency_breakdown(log_attack, os.path.join(args.out_dir, 'latency_attack.png'))

    # Scenario comparison
    plot_scenario_comparison(
        csv_files   = {'Clean': log_clean, 'Digital Attack': log_attack},
        output_path = os.path.join(args.out_dir, 'scenario_comparison.png'),
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  SIMULATION COMPLETE")
    print(f"{'═'*65}")
    print(f"  Edge profile  : {profile['desc']}")
    print(f"  Patch size    : {args.size}×{args.size}px "
          f"({args.size**2 / args.resolution**2 * 100:.2f}% of {args.resolution}×{args.resolution})")
    print(f"  Seed          : {args.seed}")
    print(f"\n  Outputs:")
    print(f"    Patch PNG         : {patch_path or 'N/A (quick mode)'}")
    print(f"    Clean log         : {log_clean}")
    print(f"    Attack log        : {log_attack}")
    print(f"    Plots             : {args.out_dir}/")
    print(f"\n  Next steps:")
    print(f"    • Run multi-seed: python experiments/multi_seed_experiment.py")
    print(f"    • Run ablation  : python experiments/ablation_patch_size.py")
    print(f"    • Run baselines : python experiments/baseline_comparison.py --patch {patch_path or '<patch_path>'}")
    print(f"    • Run defense   : python experiments/defense_evaluation.py --patch {patch_path or '<patch_path>'}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
