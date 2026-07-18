"""
experiments/defense_tradeoff.py
─────────────────────────────────────────────────────────────────
Evaluates the cost of defending against Sponge Patch.
Increasing confidence threshold or lowering max detections
stops the DoS attack, but how does it impact clean accuracy?
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.victim_model import VictimModel

def synthesize_clean_image(victim: VictimModel) -> torch.Tensor:
    """
    Since we don't have COCO dataset here, we generate a synthetic image
    that triggers some high-confidence bounding boxes to simulate a 
    real-world scene with pedestrians/cars.
    """
    # In a real paper, you would load your validation dataset here.
    # For demonstration, we just use a structured noise pattern that 
    # happens to trigger some YOLO boxes at various confidence levels.
    torch.manual_seed(1337)
    img = torch.rand((1, 3, 320, 320), device=victim.device)
    # Add some structural blocks
    img[0, :, 100:150, 100:150] = 0.9
    img[0, :, 200:280, 50:100] = 0.1
    return img

def main():
    print("[*] Running Defense Trade-off Evaluation...")
    victim = VictimModel()
    
    # We will evaluate across these confidence thresholds
    thresholds = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
    
    # Load 10 "clean" images (simulated)
    images = [synthesize_clean_image(victim) for _ in range(10)]
    
    results = {}
    
    for thresh in thresholds:
        total_dets = 0
        total_raw = 0
        for img in images:
            res = victim.get_predictions_with_nms(img, conf_thresh=thresh, max_det=300, profile_latency=False)
            total_raw += res['num_raw_boxes']
            total_dets += res['num_final_boxes']
            
        avg_dets = total_dets / len(images)
        avg_raw = total_raw / len(images)
        
        results[thresh] = {
            "avg_final_detections": avg_dets,
            "avg_raw_candidates": avg_raw
        }
        print(f"Conf={thresh:.2f} | Avg Raw: {avg_raw:.1f} | Avg Detections: {avg_dets:.1f}")

    # Plot the Trade-off Curve
    conf_vals = list(results.keys())
    det_vals = [results[c]["avg_final_detections"] for c in conf_vals]
    
    # Normalize to 100% at conf=0.01
    baseline_dets = det_vals[0] + 1e-5
    recall_retention = [(d / baseline_dets) * 100 for d in det_vals]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Defense: Confidence Threshold')
    ax1.set_ylabel('Recall Retention (%)', color=color)
    ax1.plot(conf_vals, recall_retention, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the DoS prevention threshold
    ax1.axvline(x=0.25, color='red', linestyle='--', label='DoS Prevention Threshold (0.25)')
    
    plt.title('Cost of Defense: Detection Recall vs. Confidence Threshold')
    fig.tight_layout()
    
    os.makedirs('outputs/defense_tradeoff', exist_ok=True)
    plt.savefig('outputs/defense_tradeoff/tradeoff_curve.png', dpi=200)
    
    with open('outputs/defense_tradeoff/results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("[+] Trade-off curve saved to outputs/defense_tradeoff/")

if __name__ == "__main__":
    main()
