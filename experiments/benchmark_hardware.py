"""
experiments/benchmark_hardware.py
─────────────────────────────────────────────────────────────────
Benchmarks the NMS algorithm to prove its O(N^2) time complexity.
Generates a latency curve vs. Number of Bounding Boxes (N).
Essential for demonstrating the theoretical basis of the vulnerability.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_random_boxes(N, device='cpu'):
    """Generates N random bounding boxes (xyxy) and scores."""
    # Centers
    cx = torch.rand(N) * 320
    cy = torch.rand(N) * 320
    # Widths/Heights (10 to 50 pixels)
    w = torch.rand(N) * 40 + 10
    h = torch.rand(N) * 40 + 10
    
    x1 = (cx - w/2).clamp(0, 320)
    y1 = (cy - h/2).clamp(0, 320)
    x2 = (cx + w/2).clamp(0, 320)
    y2 = (cy + h/2).clamp(0, 320)
    
    boxes = torch.stack([x1, y1, x2, y2], dim=1).to(device)
    scores = torch.rand(N).to(device)
    return boxes, scores

def benchmark_nms(device='cpu', iou_thresh=0.45):
    print(f"[*] Benchmarking NMS on {device}...")
    
    # N values to test
    N_list = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 8400, 10000]
    latencies = []
    
    # Warmup
    dummy_b, dummy_s = generate_random_boxes(100, device)
    for _ in range(5):
        _ = torchvision.ops.nms(dummy_b, dummy_s, iou_thresh)
        
    for N in N_list:
        boxes, scores = generate_random_boxes(N, device)
        
        # Measure
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Run NMS 10 times for stable measurement
        runs = 10
        for _ in range(runs):
            _ = torchvision.ops.nms(boxes, scores, iou_thresh)
            
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_ms = ((end - start) / runs) * 1000
        latencies.append(avg_ms)
        print(f"  N = {N:<6} | Latency = {avg_ms:8.2f} ms")
        
    return N_list, latencies

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    N_vals, latencies = benchmark_nms(device=device)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N_vals, latencies, 'o-', color='tab:red', linewidth=2, label='Measured Latency')
    
    # Fit an O(N^2) curve for reference (C * N^2)
    # Use the last point to calculate C
    C = latencies[-1] / (N_vals[-1] ** 2)
    theoretical_curve = [C * (n**2) for n in N_vals]
    ax.plot(N_vals, theoretical_curve, '--', color='gray', label='O(N²) Reference Curve')
    
    ax.set_title(f'NMS Latency vs. Number of Bounding Boxes (N) [{device.upper()}]')
    ax.set_xlabel('Number of Raw Bounding Boxes (N)')
    ax.set_ylabel('NMS Latency (ms)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs('outputs/benchmark', exist_ok=True)
    plt.savefig('outputs/benchmark/nms_complexity.png', dpi=200)
    
    import json
    with open('outputs/benchmark/results.json', 'w') as f:
        json.dump({"N": N_vals, "latency_ms": latencies}, f, indent=2)
        
    print("[+] Benchmark complete. Plot saved to outputs/benchmark/nms_complexity.png")

if __name__ == "__main__":
    main()
