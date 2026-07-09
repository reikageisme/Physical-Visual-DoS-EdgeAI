"""
test_physical_dos.py — Edge-AI Physical DoS Test (Real Camera / Simulation)

Fixes from original:
  - FIXED BUG: outputs was raw tensor [B,8400], code incorrectly accessed outputs[0]['scores']
    → Now uses victim_model.get_predictions_with_nms() which returns proper dict
  - Added full latency breakdown logging (NMS, forward, preproc)
  - Added '--simulate' mode for headless edge server simulation (no camera)
  - Added '--scenario' flag for logging (clean / digital_attack / physical_patch)
  - EdgeMonitor now receives latency_ms dict directly
"""

import cv2
import torch
import time
import argparse
import os
import numpy as np
import threading

from core.victim_model import VictimModel
from utils.monitor import EdgeMonitor


# ─────────────────────────────────────────────────────────────────────────────
class WebcamStream:
    """Threaded webcam reader to decouple I/O from inference."""

    def __init__(self, src: int = 0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread  = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            self.ret, frame = self.cap.read()
            if self.ret:
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap.isOpened():
            self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
class SyntheticFrameSource:
    """
    Simulated camera for headless edge server simulation.
    Generates random noise frames (or solid color) at a given resolution.
    Optionally overlays a loaded Sponge Patch.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (720, 1280),
        patch_path: str = None,
        patch_corner: str = 'center',   # 'center' | 'bottomright' | 'random'
    ):
        self.H, self.W     = resolution
        self.patch_img     = None
        self.patch_corner  = patch_corner

        if patch_path and os.path.exists(patch_path):
            self.patch_img = cv2.imread(patch_path)
            print(f"[SyntheticFrameSource] Loaded patch: {patch_path}")

        print(f"[SyntheticFrameSource] Simulated camera {self.W}×{self.H} — "
              f"patch={'YES' if self.patch_img is not None else 'NO'}")

    def read(self) -> tuple[bool, np.ndarray]:
        # Random noise frame (simulates complex scene)
        frame = np.random.randint(0, 256, (self.H, self.W, 3), dtype=np.uint8)

        if self.patch_img is not None:
            ph = min(self.patch_img.shape[0], self.H)
            pw = min(self.patch_img.shape[1], self.W)
            patch_rs = cv2.resize(self.patch_img, (pw, ph))

            if self.patch_corner == 'center':
                y0 = (self.H - ph) // 2
                x0 = (self.W - pw) // 2
            elif self.patch_corner == 'bottomright':
                y0, x0 = self.H - ph, self.W - pw
            else:
                y0 = np.random.randint(0, max(1, self.H - ph))
                x0 = np.random.randint(0, max(1, self.W - pw))

            frame[y0:y0+ph, x0:x0+pw] = patch_rs

        return True, frame

    def stop(self):
        pass   # no-op


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_frame(
    frame: np.ndarray,
    target_size: int = 320,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert OpenCV BGR frame to float32 RGB tensor [1, 3, H, W]."""
    device = device or torch.device('cpu')
    resized = cv2.resize(frame, (target_size, target_size))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor  = torch.from_numpy(rgb).float() / 255.0      # [H, W, 3]
    tensor  = tensor.permute(2, 0, 1).unsqueeze(0)        # [1, 3, H, W]
    return tensor.to(device)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Edge-AI DoS Test — Real camera or headless simulation"
    )
    parser.add_argument('--cam',        type=int,   default=0,
                        help='Camera ID (ignored in --simulate mode)')
    parser.add_argument('--patch',      type=str,   default=None,
                        help='Path to sponge patch PNG to overlay')
    parser.add_argument('--simulate',   action='store_true',
                        help='Run in headless simulation mode (no real camera needed)')
    parser.add_argument('--frames',     type=int,   default=200,
                        help='Number of frames to run in simulation mode (default: 200)')
    parser.add_argument('--resolution', type=int,   default=320,
                        help='Input resolution fed to YOLO (default: 320)')
    parser.add_argument('--conf',       type=float, default=0.25,
                        help='Confidence threshold for NMS (default: 0.25)')
    parser.add_argument('--max-det',    type=int,   default=300,
                        help='Max detections cap for NMS (defense knob, default: 300)')
    parser.add_argument('--scenario',   type=str,   default='unknown',
                        help='Scenario tag for CSV log (e.g. clean / digital_attack / physical)')
    parser.add_argument('--num-threads', type=int,  default=1,
                        help='CPU thread count (1 = simulate Raspberry Pi, default: 1)')
    parser.add_argument('--target-fps', type=float, default=10.0,
                        help='Target FPS (default: 10)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable OpenCV window (for headless servers)')
    args = parser.parse_args()

    # ── Edge simulation: throttle CPU threads ─────────────────────────────────
    torch.set_num_threads(args.num_threads)
    print(f"[Edge Sim] CPU threads: {args.num_threads} "
          f"({'simulating edge device' if args.num_threads <= 2 else 'full CPU'})")

    print("=== EDGE-AI Visual DoS TEST ===")
    print(f"Scenario : {args.scenario}")
    print(f"Mode     : {'SIMULATION (headless)' if args.simulate else 'REAL CAMERA'}")
    print(f"Patch    : {args.patch or 'None (clean baseline)'}")

    # ── Load model ────────────────────────────────────────────────────────────
    victim = VictimModel()

    # ── Frame source ──────────────────────────────────────────────────────────
    if args.simulate:
        frame_h = 720   # typical IP camera resolution
        source  = SyntheticFrameSource(
            resolution   = (frame_h, 1280),
            patch_path   = args.patch,
            patch_corner = 'center',
        )
    else:
        source = WebcamStream(src=args.cam)
        time.sleep(2)
        if not source.ret:
            print("[-] Camera not found! Use --simulate for headless mode.")
            return

    # ── Monitor ───────────────────────────────────────────────────────────────
    monitor   = EdgeMonitor(log_dir="logs", scenario=args.scenario)
    TARGET_FPS = args.target_fps
    FRAME_TIME = 1.0 / TARGET_FPS

    print(f"\n[Running] Target FPS={TARGET_FPS} | conf={args.conf} | max_det={args.max_det}")
    print(f"[Running] Press Ctrl+C to stop\n")

    frame_count = 0

    try:
        while True:
            if args.simulate and frame_count >= args.frames:
                print(f"\n[Sim] Reached {args.frames} frames. Done.")
                break

            t_frame_start = time.perf_counter()

            # ── Capture ───────────────────────────────────────────────────────
            ret, frame = source.read()
            if not ret:
                continue

            # ── Preprocess ────────────────────────────────────────────────────
            input_tensor = preprocess_frame(frame, args.resolution, victim.device)

            # ── Inference + NMS + Latency profiling ───────────────────────────
            result = victim.get_predictions_with_nms(
                input_tensor,
                conf_thresh     = args.conf,
                iou_thresh      = 0.45,
                max_det         = args.max_det,
                profile_latency = True,
            )

            # ── FPS calculation ────────────────────────────────────────────────
            elapsed     = time.perf_counter() - t_frame_start
            sleep_time  = FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            total_elapsed = max(time.perf_counter() - t_frame_start, 0.001)
            actual_fps    = 1.0 / total_elapsed

            # ── Render time estimate ───────────────────────────────────────────
            t_render_start = time.perf_counter()

            # ── Log ───────────────────────────────────────────────────────────
            stats = monitor.log_status(
                current_fps     = actual_fps,
                latency_ms      = result['latency_ms'],
                num_raw_boxes   = result['num_raw_boxes'],
                num_final_boxes = result['num_final_boxes'],
                render_ms       = 0.0,   # updated below
            )

            # ── Display (skip if headless) ─────────────────────────────────────
            if not args.no_display and not args.simulate:
                disp = cv2.resize(frame, (640, 480))
                color = (0, 0, 255) if result['num_raw_boxes'] > 200 else (0, 255, 0)

                def put(text, y):
                    cv2.putText(disp, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                put(f"Scenario: {args.scenario}", 30)
                put(f"FPS: {actual_fps:.1f} / {TARGET_FPS:.0f}", 60)
                put(f"CPU: {stats['cpu']:.1f}%  RAM: {stats['ram_pct']:.1f}%", 90)
                put(f"Raw Boxes (pre-NMS): {result['num_raw_boxes']}", 120)
                put(f"Final Dets (post-NMS): {result['num_final_boxes']}", 150)
                put(f"NMS: {result['latency_ms']['nms_ms']:.1f}ms  "
                    f"Fwd: {result['latency_ms']['forward_ms']:.1f}ms", 180)
                put(f"Total: {result['latency_ms']['total_ms']:.1f}ms", 210)

                cv2.imshow("Edge-AI DoS Test", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── Console print ─────────────────────────────────────────────────
            if frame_count % 20 == 0 or args.simulate:
                raw_n = result['num_raw_boxes']
                fin_n = result['num_final_boxes']
                nms_t = result['latency_ms']['nms_ms']
                fwd_t = result['latency_ms']['forward_ms']
                print(f"Frame {frame_count:5d} | FPS:{actual_fps:5.1f} | "
                      f"CPU:{stats['cpu']:5.1f}% | "
                      f"RawBox:{raw_n:5d} | FinalDet:{fin_n:4d} | "
                      f"NMS:{nms_t:6.2f}ms | Fwd:{fwd_t:6.2f}ms")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")
    finally:
        source.stop()
        if not args.simulate:
            cv2.destroyAllWindows()
        monitor.print_summary()
        print(f"[*] Log saved to: {monitor.log_file}")


if __name__ == "__main__":
    main()