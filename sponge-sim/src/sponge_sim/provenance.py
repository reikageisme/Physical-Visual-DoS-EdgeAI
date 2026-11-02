"""Code COPY từ repo cũ (không sửa bản gốc). Mỗi hàm có header nguồn: file:dòng + SHA256 file
tại thời điểm copy (21/10/2026). tests/test_provenance.py kiểm SHA và so hành vi với bản gốc.

Khác biệt so với bản gốc (có chủ ý, đã ghi):
- decode/postprocess không nhận `victim`; dùng hàm copy `xywh_to_xyxy`/`nms_torch_pure`.
- postprocess bỏ phần đo thời gian (timing do timing.py đảm nhiệm) và nhận `nms_fn`.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SOURCES = {
    'core/victim_model.py':
        '1382c3256917cd3b5d9a435dcb23128ac7a25151077e14ef993c9e5ea7d68a17',
    'experiments/extended_evaluation.py':
        'a8a0fa033a563fba8779da5a2d47965ec34dbc8ba4b4f84c7ce1c434e92dfdf6',
    'experiments/local_q1_validation.py':
        '9739eb6675b0ac025309eb8874132d5f2c14594e27f18222c5aece8f9a34121a',
    'experiments/extended_validation.py':
        '5d551d5e3582b449c942a00279c18a026a30b8a5260d4abaf03e9e0dd48da69c',
}
IOU = .45      # experiments/extended_evaluation.py:43
MAX_OUT = 300  # experiments/extended_evaluation.py:44 (indices[:300])


# SOURCE: core/victim_model.py:233-266 (VictimModel._nms) sha256=1382c325...
def nms_torch_pure(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Pure-PyTorch NMS, class-agnostic. Tie-break: argsort(descending=True) của torch (không ổn định)."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes_xyxy.device)
    order = scores.argsort(descending=True)
    keep = []
    x1 = boxes_xyxy[:, 0]; y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]; y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = x1[rest].clamp(min=x1[i].item())
        iy1 = y1[rest].clamp(min=y1[i].item())
        ix2 = x2[rest].clamp(max=x2[i].item())
        iy2 = y2[rest].clamp(max=y2[i].item())
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyxy.device)


# SOURCE: core/victim_model.py:268-276 (VictimModel._xywh_to_xyxy) sha256=1382c325...
def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


# SOURCE: experiments/extended_evaluation.py:29-32 (decode) sha256=a8a0fa03...
def decode(raw: torch.Tensor) -> torch.Tensor:
    """raw [4+nc, A] → [A, 6] = x1,y1,x2,y2,score(max lớp),class. Class-agnostic score."""
    scores, classes = raw[4:].max(dim=0)
    boxes = xywh_to_xyxy(raw[:4].T.float())
    return torch.cat([boxes, scores[:, None], classes[:, None].float()], 1)


# SOURCE: experiments/extended_evaluation.py:35-48 (postprocess, bỏ timing) sha256=a8a0fa03...
def tau_filter(decoded: torch.Tensor, threshold: float = .01, cap: int | None = None) -> tuple[torch.Tensor, int]:
    active = decoded[decoded[:, 4] > threshold]
    count = len(active)
    if cap is not None and count > cap:
        active = active[active[:, 4].topk(cap).indices]
    return active, count


def postprocess(decoded, threshold=.01, cap=None, nms_fn=None):
    active, count = tau_filter(decoded, threshold, cap)
    nms_fn = nms_fn or nms_torch_pure
    indices = nms_fn(active[:, :4], active[:, 4], IOU)
    result = active[indices[:MAX_OUT]]
    return dict(active_candidates=count, nms_input_candidates=len(active),
                final_detections=len(result)), result


# SOURCE: experiments/local_q1_validation.py:64-67 (load_image) sha256=9739eb66...
def load_image(path) -> torch.Tensor:
    """Resize trực tiếp 320×320, RGB [0,1]; KHÔNG letterbox kiểu Ultralytics."""
    image = Image.open(path).convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    return (torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255).contiguous()


# SOURCE: experiments/extended_validation.py:84-89 (insert) sha256=5d551d5e...
def insert(images, patches, locations):
    result = images.unsqueeze(0).expand(len(patches), -1, -1, -1, -1).clone()
    size = patches.shape[-1]
    for index, (y, x) in enumerate(locations):
        result[:, index, :, y:y+size, x:x+size] = patches
    return result.flatten(0, 1).contiguous()
