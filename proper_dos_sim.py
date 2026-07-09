"""
proper_dos_sim.py — Simulation chứng minh NMS O(N²) bottleneck
Chạy trên CPU laptop, throttle 1 thread để giả lập Edge Server yếu

3 scenarios:
  1. CLEAN       — frame rich scene, không patch
  2. DIGITAL     — frame + sponge patch optimize
  3. PHYSICAL    — frame + patch bị blur/noise (giả lập in ấn + domain gap)
"""

import sys, os, time, json, csv
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# ── Throttle CPU (giả lập single-core edge server) ──────────────────────────
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)   # 2 threads ~ Raspberry Pi / Jetson Nano

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from core.victim_model import VictimModel

# ── Config ───────────────────────────────────────────────────────────────────
CONF_THRESH = 0.001       # Cực thấp → thấy toàn bộ raw anchors của YOLO
IOU_THRESH  = 0.45
N_FRAMES    = 120         # frames mỗi scenario
PATCH_PATH  = ROOT / "outputs" / "sponge_patch_simulated_s42_g25_p20_sz64.png"
OUT_DIR     = ROOT / "outputs" / "proper_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 68)
print("  Proper NMS Bottleneck Simulation — 3 Scenarios")
print(f"  CPU threads : {torch.get_num_threads()} (throttled = Edge Server sim)")
print(f"  conf_thresh : {CONF_THRESH}  (captures ALL raw YOLO anchors)")
print(f"  Frames/run  : {N_FRAMES}")
print("=" * 68)

# ── Synthetic rich scene (nhiều edges, textures → YOLO fire nhiều anchors) ──
rng = np.random.default_rng(42)

def make_rich_scene(seed_offset=0):
    """Frame có nội dung (buildings, people blobs) → YOLO kích hoạt nhiều hơn noise thuần"""
    local_rng = np.random.default_rng(42 + seed_offset)
    h, w = 320, 320
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(h // 2):
        v = int(120 + 80 * y / (h / 2))
        frame[y] = [v - 30, v - 10, v + 20]

    # Ground
    frame[h // 2:] = [45, 75, 45]

    # Buildings (structured vertical edges → YOLO anchor activations)
    for _ in range(local_rng.integers(4, 9)):
        x1 = int(local_rng.integers(0, w - 50))
        y1 = int(local_rng.integers(20, h // 2 - 15))
        bw = int(local_rng.integers(25, 70))
        bh_max = max(42, h // 2 - y1)
        bh = int(local_rng.integers(40, bh_max))
        c  = local_rng.integers(70, 210, 3).tolist()
        cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), c, -1)
        cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), [10, 10, 10], 1)
        for wy in range(y1 + 5, y1 + bh - 5, 12):
            for wx in range(x1 + 5, x1 + bw - 5, 9):
                cv2.rectangle(frame, (wx, wy), (wx + 5, wy + 7), [200, 230, 255], -1)

    # Person blobs (tall thin shapes YOLO strongly responds to)
    for _ in range(local_rng.integers(2, 5)):
        cx = int(local_rng.integers(15, w - 15))
        cy = int(local_rng.integers(h // 2 + 10, h - 15))
        cv2.ellipse(frame, (cx, cy), (9, 25), 0, 0, 360, [40, 40, 40], -1)
        cv2.circle(frame, (cx, cy - 28), 9, [55, 35, 25], -1)

    # Subtle sensor noise
    noise = local_rng.integers(-8, 8, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def overlay_patch(frame, patch):
    """Đặt patch vào vị trí saliency (y=128, x=128) — vùng nhạy cảm"""
    h, w     = frame.shape[:2]
    ph, pw   = patch.shape[:2]
    y, x     = 128, 128
    y = min(y, h - ph); x = min(x, w - pw)
    result   = frame.copy()
    result[y:y+ph, x:x+pw] = patch
    return result


def degrade_patch_physical(patch, domain_gap=0.55):
    """
    Giả lập Domain Gap khi in vật lý:
      - Gaussian blur  (mất nét qua ống kính)
      - Additive noise (sensor camera)
      - Brightness reduction (ánh sáng phòng vs màn hình)
      - Color shift     (giấy vs backlit display)
    domain_gap=0.55 → ~40% suy giảm fitness (từ 66 → ~40)
    """
    p = patch.copy().astype(np.float32)
    k = max(1, int(4 * domain_gap)) * 2 + 1
    p = cv2.GaussianBlur(p, (k, k), domain_gap * 2.5)
    p += np.random.normal(0, 18 * domain_gap, p.shape)
    p  = p * (1.0 - 0.35 * domain_gap)
    hsv = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= (1.0 - 0.3 * domain_gap)   # desaturate
    p = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(p, 0, 255).astype(np.uint8)


# ── Load model ───────────────────────────────────────────────────────────────
print("\n[1] Loading model (yolov8n.pt)...")
model = VictimModel(device="cpu")

# ── Load patch ───────────────────────────────────────────────────────────────
if PATCH_PATH.exists():
    patch_digital = cv2.imread(str(PATCH_PATH))
    print(f"[+] Loaded patch: {PATCH_PATH.name} {patch_digital.shape}")
else:
    patch_digital = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    print("[!] Patch not found, using random noise")

patch_physical = degrade_patch_physical(patch_digital, domain_gap=0.55)
print("[+] Physical-degraded patch ready (domain_gap=0.55)")


# ── Run scenario ─────────────────────────────────────────────────────────────
def run_scenario(label, patch=None, n=N_FRAMES):
    print(f"\n{'='*68}")
    print(f"  Scenario: {label.upper()}")
    print(f"  Patch   : {'None (clean)' if patch is None else label}")
    print(f"{'='*68}")

    nms_ms_list, fps_list, raw_list = [], [], []
    fwd_list, pre_list, cf_list     = [], [], []
    rows = []

    for i in range(1, n + 1):
        frame = make_rich_scene(seed_offset=i)
        if patch is not None:
            frame = overlay_patch(frame, patch)

        # Tensor
        inp = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # Full pipeline with per-stage profiling
        res = model.get_predictions_with_nms(
            inp, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH,
            max_det=10000,   # no cap → see full NMS cost
            profile_latency=True,
        )
        lat = res["latency_ms"]

        nms_ms = lat["nms_ms"]
        fwd_ms = lat["forward_ms"]
        pre_ms = lat["preproc_ms"]
        cf_ms  = lat["conf_filter_ms"]
        tot_ms = lat["total_ms"]
        fps    = 1000.0 / max(tot_ms, 0.1)
        n_raw  = res["num_raw_boxes"]
        n_fin  = res["num_final_boxes"]

        nms_ms_list.append(nms_ms)
        fps_list.append(fps)
        raw_list.append(n_raw)
        fwd_list.append(fwd_ms)
        pre_list.append(pre_ms)
        cf_list.append(cf_ms)

        rows.append({
            "frame": i, "scenario": label,
            "fps": round(fps, 2),
            "preproc_ms": round(pre_ms, 3),
            "forward_ms": round(fwd_ms, 2),
            "conf_filter_ms": round(cf_ms, 3),
            "nms_ms": round(nms_ms, 3),
            "total_ms": round(tot_ms, 2),
            "num_raw_boxes": n_raw,
            "num_final_boxes": n_fin,
        })

        if i % 40 == 0 or i == n:
            print(f"  [{i:4d}/{n}] FPS:{fps:6.1f} | Raw:{n_raw:5d} "
                  f"| NMS:{nms_ms:7.2f}ms | Fwd:{fwd_ms:6.1f}ms")

    nms_arr = np.array(nms_ms_list)
    raw_arr = np.array(raw_list)
    fps_arr = np.array(fps_list)

    stats = {
        "scenario"       : label,
        "n_frames"       : n,
        "fps_mean"       : round(float(np.mean(fps_arr)), 2),
        "fps_std"        : round(float(np.std(fps_arr)), 2),
        "fps_min"        : round(float(np.min(fps_arr)), 2),
        "fps_max"        : round(float(np.max(fps_arr)), 2),
        "nms_ms_mean"    : round(float(np.mean(nms_arr)), 3),
        "nms_ms_std"     : round(float(np.std(nms_arr)), 3),
        "nms_ms_max"     : round(float(np.max(nms_arr)), 3),
        "raw_boxes_mean" : round(float(np.mean(raw_arr)), 1),
        "raw_boxes_std"  : round(float(np.std(raw_arr)), 1),
        "raw_boxes_max"  : int(np.max(raw_arr)),
        "forward_ms_mean": round(float(np.mean(fwd_list)), 2),
    }

    print(f"\n  Summary — {label}")
    print(f"  {'─'*60}")
    print(f"  FPS           : {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}")
    print(f"  Raw Boxes     : {stats['raw_boxes_mean']:.1f} ± {stats['raw_boxes_std']:.1f}  [max={stats['raw_boxes_max']}]")
    print(f"  NMS latency   : {stats['nms_ms_mean']:.3f} ± {stats['nms_ms_std']:.3f} ms  [max={stats['nms_ms_max']:.3f}]")
    print(f"  Forward+Prep  : {stats['forward_ms_mean']:.2f} ms avg")

    # Save CSV
    ts = datetime.now().strftime("%H%M%S")
    csv_path = OUT_DIR / f"proper_{label}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader(); wr.writerows(rows)
    print(f"  CSV → {csv_path.name}")

    return stats


# ── Run 3 scenarios ──────────────────────────────────────────────────────────
results = {}
results["clean"]    = run_scenario("clean",    patch=None,           n=N_FRAMES)
results["digital"]  = run_scenario("digital",  patch=patch_digital,  n=N_FRAMES)
results["physical"] = run_scenario("physical", patch=patch_physical, n=N_FRAMES)

# ── Final table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  KẾT QUẢ — Điền vào BẢNG 2 trong paper")
print("=" * 68)
hdr = f"{'Scenario':<22} {'FPS (±std)':>12} {'Raw Boxes':>10} {'NMS ms (±std)':>16} {'NMS max':>9}"
print(hdr)
print("─" * 72)
labels = {"clean": "Clean Stream", "digital": "Digital Attack",
          "physical": "Physical (EOT sim)"}
for k, s in results.items():
    print(f"  {labels[k]:<20} "
          f"{s['fps_mean']:>5.1f}±{s['fps_std']:<5.1f} "
          f"{s['raw_boxes_mean']:>9.0f} "
          f"{s['nms_ms_mean']:>7.3f}±{s['nms_ms_std']:<7.3f} "
          f"{s['nms_ms_max']:>9.3f}")
print("=" * 72)

# ── O(N²) verification ────────────────────────────────────────────────────────
c_raw = results["clean"]["raw_boxes_mean"]
d_raw = results["digital"]["raw_boxes_mean"]
c_nms = results["clean"]["nms_ms_mean"]
d_nms = results["digital"]["nms_ms_mean"]

if c_raw > 0 and c_nms > 0 and d_raw > 0:
    box_ratio = d_raw / c_raw
    nms_ratio = d_nms / c_nms
    predicted  = box_ratio ** 2
    print(f"\n  O(N²) Analysis:")
    print(f"  Raw box ratio  (attack/clean)   : {box_ratio:.2f}x")
    print(f"  NMS time ratio (attack/clean)   : {nms_ratio:.2f}x")
    print(f"  O(N²) predicted ratio           : {predicted:.2f}x")
    pct_error = abs(nms_ratio - predicted) / predicted * 100
    print(f"  Match: {'✅ CONFIRMED' if pct_error < 50 else '⚠️ Approximate'} ({pct_error:.1f}% error)")
    results["o2_analysis"] = {
        "box_ratio": round(box_ratio, 3),
        "nms_time_ratio": round(nms_ratio, 3),
        "o2_predicted": round(predicted, 3),
        "pct_error": round(pct_error, 1),
    }

# ── Save JSON for paper ───────────────────────────────────────────────────────
json_path = OUT_DIR / "table2_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  → Saved: {json_path}")
print("=" * 68)
