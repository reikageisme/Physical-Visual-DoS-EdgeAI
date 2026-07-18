"""
experiments/transferability.py
─────────────────────────────────────────────────────────────────
Evaluates the transferability of the YOLO-optimized Sponge Patch
against other architectures (Faster R-CNN, RetinaNet, DETR).
Proves the NMS bottleneck is a structural flaw across models.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from core.victim_model_zoo import VictimZoo

def load_image_tensor(path: str, size=320) -> torch.Tensor:
    if not os.path.exists(path):
        # Create dummy image if doesn't exist (for CI/CD or fresh run)
        print(f"[!] Warning: {path} not found. Creating random noise tensor.")
        return torch.rand((1, 3, size, size))
    
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size))
    tensor = TF.to_tensor(img).unsqueeze(0)
    return tensor

def apply_patch(base_tensor: torch.Tensor, patch_tensor: torch.Tensor, y: int, x: int) -> torch.Tensor:
    """Overlays patch onto base image at (y,x)"""
    # Resize patch to 64x64 if needed (assuming base is 320x320)
    patch_size = 64
    if patch_tensor.shape[-1] != patch_size:
        patch_tensor = torch.nn.functional.interpolate(patch_tensor, size=(patch_size, patch_size))
        
    patched = base_tensor.clone()
    patched[0, :, y:y+patch_size, x:x+patch_size] = patch_tensor[0]
    return patched

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Running Transferability Evaluation on {device}...")
    
    # Load images
    clean_tensor = torch.rand((1, 3, 320, 320)) # fallback clean
    
    patch_path = 'outputs/sponge_patch.png'
    if os.path.exists(patch_path):
        patch_tensor = load_image_tensor(patch_path, size=64)
    else:
        print("[!] Warning: outputs/sponge_patch.png not found. Using random patch.")
        patch_tensor = torch.rand((1, 3, 64, 64))
        
    patched_tensor = apply_patch(clean_tensor, patch_tensor, y=128, x=128)
    
    models_to_test = [
        "yolov8n.pt",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "retinanet_resnet50_fpn",
        "detr_resnet50"
    ]
    
    results = {}
    
    for m in models_to_test:
        print(f"\nEvaluating: {m}")
        try:
            zoo = VictimZoo(m, device=device)
            
            clean_res = zoo.evaluate_latency(clean_tensor)
            patched_res = zoo.evaluate_latency(patched_tensor)
            
            # Memory cleanup
            del zoo
            torch.cuda.empty_cache() if device == 'cuda' else None
            
            ratio = patched_res['avg_latency_ms'] / (clean_res['avg_latency_ms'] + 1e-5)
            
            print(f"  Clean latency   : {clean_res['avg_latency_ms']:.2f} ms")
            print(f"  Patched latency : {patched_res['avg_latency_ms']:.2f} ms")
            print(f"  Latency ratio   : {ratio:.2f}x")
            
            results[m] = {
                "has_nms": clean_res["has_nms"],
                "clean_latency_ms": clean_res["avg_latency_ms"],
                "patched_latency_ms": patched_res["avg_latency_ms"],
                "latency_ratio": ratio
            }
        except Exception as e:
            print(f"  [!] Failed to evaluate {m}: {e}")
            
    # Save results
    os.makedirs('outputs/transferability', exist_ok=True)
    out_path = 'outputs/transferability/cross_arch_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[+] Saved transferability results to {out_path}")

if __name__ == "__main__":
    main()
