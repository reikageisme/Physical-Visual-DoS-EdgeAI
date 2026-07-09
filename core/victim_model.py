"""
victim_model.py — Victim Model Wrapper

Threat Model: Gray-box / White-box-at-output-head
- Attacker has access to pre-NMS confidence scores (via local model copy or API telemetry)
- NMS post-processing is NOT directly accessible (treated as black-box pipeline step)
- This is more accurately described as "score-based gray-box" rather than "strict black-box"

Note: fast_train.py uses gradient backprop (white-box) — separate threat model.
"""

import time
import torch
import torch.nn.functional as F
from ultralytics import YOLO


class VictimModel:
    """
    Wraps YOLOv8n for adversarial attack research.

    Provides two access modes:
      - get_raw_predictions()  : pre-NMS scores (gray-box mode)
      - get_predictions_with_nms() : full pipeline with NMS + latency profiling
    """

    def __init__(self, model_path: str = 'yolov8n.pt', device: str = None):
        # ── Device setup ─────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[VictimModel] Loading {model_path} on {self.device} ...")
        yolo = YOLO(model_path)

        # Half precision only on CUDA; CPU must use float32
        self.use_half = torch.cuda.is_available() and self.device.type == 'cuda'
        dtype = torch.float16 if self.use_half else torch.float32

        # Extract the raw PyTorch backbone (bypasses Ultralytics wrapper)
        self.model = yolo.model.to(self.device).eval()
        if self.use_half:
            self.model = self.model.half()
        else:
            self.model = self.model.float()

        # Keep float32 version for gradient-based attacks (fast_train.py)
        self.model_float = yolo.model.to(self.device).eval().float()

        print(f"[VictimModel] Ready. Precision: {'float16 (CUDA)' if self.use_half else 'float32 (CPU)'}")
        print(f"[VictimModel] Threat level: gray-box (pre-NMS scores accessible)")

    # ─────────────────────────────────────────────────────────────────────────
    # GRAY-BOX MODE: Returns pre-NMS raw confidence scores
    # Used by: GA optimization (main_train.py), fitness function
    # ─────────────────────────────────────────────────────────────────────────
    def get_raw_predictions(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        [GRAY-BOX] Run inference and return pre-NMS max confidence scores.

        Args:
            image_tensor: float32 tensor [B, 3, H, W], values in [0, 1]

        Returns:
            max_scores: float16 tensor [B, 8400] — max class confidence per anchor
        """
        with torch.no_grad():
            image_tensor = self._pad_to_stride32(image_tensor)
            # Use half only if CUDA available; CPU must stay float32
            if self.use_half:
                image_tensor = image_tensor.half()
            else:
                image_tensor = image_tensor.float()
            preds = self.model(image_tensor)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            # preds shape: [B, 84, 8400]  (4 bbox + 80 class probs)
            cls_probs = preds[:, 4:, :]            # [B, 80, 8400]
            max_scores, _ = torch.max(cls_probs, dim=1)   # [B, 8400]

        return max_scores  # float16, on self.device

    # ─────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE: Runs NMS and profiles each stage
    # Used by: test scripts, defense evaluation, latency breakdown
    # ─────────────────────────────────────────────────────────────────────────
    def get_predictions_with_nms(
        self,
        image_tensor: torch.Tensor,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_det: int = 300,
        profile_latency: bool = True,
    ) -> dict:
        """
        [FULL PIPELINE] Run inference with NMS and optional latency profiling.

        Args:
            image_tensor   : float32 tensor [1, 3, H, W], values in [0, 1]
            conf_thresh    : confidence threshold before NMS
            iou_thresh     : IoU threshold for NMS
            max_det        : maximum detections allowed (defense knob)
            profile_latency: whether to measure per-stage latency (ms)

        Returns:
            dict with keys:
              'boxes'            : [N, 4] final bounding boxes (xyxy)
              'scores'           : [N] final confidence scores
              'num_raw_boxes'    : int — candidates before NMS (attack metric)
              'num_final_boxes'  : int — detections after NMS
              'latency_ms'       : dict with per-stage ms (if profile_latency)
        """
        t = {}

        # ── 1. Preprocessing ─────────────────────────────────────────────────
        if profile_latency:
            t['preproc_start'] = time.perf_counter()

        img = self._pad_to_stride32(image_tensor)
        if self.use_half:
            img = img.half()
        else:
            img = img.float()

        if profile_latency:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t['preproc_end'] = time.perf_counter()

        # ── 2. Forward pass ───────────────────────────────────────────────────
        if profile_latency:
            t['forward_start'] = time.perf_counter()

        with torch.no_grad():
            preds = self.model(img)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]   # [B, 84, 8400]

        if profile_latency:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t['forward_end'] = time.perf_counter()

        # ── 3. Confidence filtering ───────────────────────────────────────────
        if profile_latency:
            t['conf_start'] = time.perf_counter()

        # preds: [1, 84, 8400] → take first batch item
        pred = preds[0]                          # [84, 8400]
        boxes_xywh = pred[:4, :].T               # [8400, 4]
        cls_probs  = pred[4:, :].T               # [8400, 80]
        max_scores, cls_ids = cls_probs.max(dim=1)  # [8400]

        keep_mask    = max_scores > conf_thresh
        num_raw_boxes = int(keep_mask.sum().item())

        filtered_scores = max_scores[keep_mask]
        filtered_boxes  = boxes_xywh[keep_mask]
        # filtered_cls    = cls_ids[keep_mask]

        if profile_latency:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t['conf_end'] = time.perf_counter()

        # ── 4. NMS ────────────────────────────────────────────────────────────
        if profile_latency:
            t['nms_start'] = time.perf_counter()

        final_boxes  = torch.zeros((0, 4), device=self.device)
        final_scores = torch.zeros(0, device=self.device)

        if num_raw_boxes > 0:
            # Convert xywh → xyxy for NMS
            boxes_xyxy = self._xywh_to_xyxy(filtered_boxes.float())
            scores_f32 = filtered_scores.float()

            nms_idx = self._nms(boxes_xyxy, scores_f32, iou_thresh)
            nms_idx = nms_idx[:max_det]   # defense: cap max detections

            final_boxes  = boxes_xyxy[nms_idx]
            final_scores = scores_f32[nms_idx]

        if profile_latency:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            t['nms_end'] = time.perf_counter()

        # ── Build result ──────────────────────────────────────────────────────
        result = {
            'boxes'          : final_boxes.cpu(),
            'scores'         : final_scores.cpu(),
            'num_raw_boxes'  : num_raw_boxes,
            'num_final_boxes': int(final_scores.shape[0]),
        }

        if profile_latency:
            result['latency_ms'] = {
                'preproc_ms'   : (t['preproc_end']  - t['preproc_start'])  * 1000,
                'forward_ms'   : (t['forward_end']  - t['forward_start'])  * 1000,
                'conf_filter_ms': (t['conf_end']    - t['conf_start'])     * 1000,
                'nms_ms'       : (t['nms_end']      - t['nms_start'])      * 1000,
                'total_ms'     : (t['nms_end']      - t['preproc_start'])  * 1000,
            }

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # GRADIENT MODE: Float32 model for white-box PGD (fast_train.py)
    # ─────────────────────────────────────────────────────────────────────────
    def get_model_float(self) -> torch.nn.Module:
        """Return float32 model for gradient-based optimization."""
        return self.model_float

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _pad_to_stride32(self, x: torch.Tensor) -> torch.Tensor:
        """Pad spatial dims to nearest multiple of 32 (YOLO stride requirement)."""
        _, _, H, W = x.shape
        new_H = ((H + 31) // 32) * 32
        new_W = ((W + 31) // 32) * 32
        if new_H != H or new_W != W:
            x = F.pad(x, (0, new_W - W, 0, new_H - H), value=0.0)
        return x

    @staticmethod
    def _nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
        """
        Pure-PyTorch NMS (no torchvision dependency).
        Returns indices of kept boxes, sorted by score descending.
        """
        if boxes_xyxy.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes_xyxy.device)

        # Sort by score descending
        order = scores.argsort(descending=True)
        keep  = []

        x1 = boxes_xyxy[:, 0]; y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]; y2 = boxes_xyxy[:, 3]
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]

            # Intersection
            ix1 = x1[rest].clamp(min=x1[i].item())
            iy1 = y1[rest].clamp(min=y1[i].item())
            ix2 = x2[rest].clamp(max=x2[i].item())
            iy2 = y2[rest].clamp(max=y2[i].item())
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

            iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
            order = rest[iou <= iou_thresh]

        return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)

    @staticmethod
    def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert [cx, cy, w, h] → [x1, y1, x2, y2]."""
        out = torch.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out