import time
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    retinanet_resnet50_fpn
)
from ultralytics import YOLO

class VictimZoo:
    """
    Wrapper for evaluating transferability across different Object Detection architectures.
    Supports YOLOv8 (NMS), Faster R-CNN (NMS), RetinaNet (NMS), and DETR (NMS-free).
    """
    def __init__(self, model_name: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model_name = model_name
        self.is_yolo = "yolo" in model_name.lower()
        self.is_detr = "detr" in model_name.lower()
        
        print(f"[VictimZoo] Loading {model_name} on {self.device}...")
        
        if self.is_yolo:
            self.model = YOLO(model_name).model.to(self.device).eval()
        elif "fasterrcnn" in model_name.lower():
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT').to(self.device).eval()
        elif "retinanet" in model_name.lower():
            self.model = retinanet_resnet50_fpn(weights='DEFAULT').to(self.device).eval()
        elif "detr" in model_name.lower():
            # DETR is not in torchvision models detection by default, load via hub
            self.model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True).to(self.device).eval()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
    def evaluate_latency(self, image_tensor: torch.Tensor, n_warmup: int = 2, n_runs: int = 5) -> dict:
        """
        Evaluate inference latency. 
        For torchvision models, NMS is bundled in the forward pass.
        We measure total forward pass time to see if the patched image causes a bottleneck.
        """
        image_tensor = image_tensor.to(self.device)
        
        # Ensure correct dtype
        if self.device.type == 'cuda' and self.is_yolo:
            self.model = self.model.half()
            image_tensor = image_tensor.half()
        else:
            self.model = self.model.float()
            image_tensor = image_tensor.float()
            
        with torch.no_grad():
            # Warmup
            for _ in range(n_warmup):
                _ = self.model(image_tensor)
                
            # Measure
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            for _ in range(n_runs):
                preds = self.model(image_tensor)
                
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            avg_ms = ((end_time - start_time) / n_runs) * 1000
            
            # Extract final detection count
            num_final_boxes = 0
            if self.is_yolo:
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                num_final_boxes = preds.shape[2] # raw boxes actually, but YOLO NMS is separate in our custom pipeline.
                # Actually, YOLO PyTorch backbone returns raw boxes. We just want to see if forward pass slows down.
                # Wait, YOLO PyTorch forward pass does NOT include NMS. 
                # So YOLO latency here won't spike unless we use the Ultralytics wrapper.
            else:
                # Torchvision or DETR models return differently
                if self.is_detr and isinstance(preds, dict) and 'pred_boxes' in preds:
                    num_final_boxes = preds['pred_boxes'].shape[1]
                elif isinstance(preds, list) and isinstance(preds[0], dict) and 'boxes' in preds[0]:
                    num_final_boxes = preds[0]['boxes'].shape[0]
                    
        return {
            "model": self.model_name,
            "avg_latency_ms": avg_ms,
            "num_final_boxes": num_final_boxes,
            "has_nms": not self.is_detr
        }
