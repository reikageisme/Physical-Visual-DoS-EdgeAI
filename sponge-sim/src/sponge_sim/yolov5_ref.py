"""Nạp YOLOv5s của kho tham chiếu Underload (chỉ đọc) để đo lại các nhiễu loạn Phantom/Overload đã lưu (E10).

SOURCE: experiments/extended_reference.py:35-82 (VENDOR/COMMIT/PACKAGE/WEIGHT, namespace, initialize)
        sha256=1d3d2b1236f5e103fdc47781f516bbded527b3bb851f3d0adde866b31af814bf (02/11/2026)
Khác bản gốc (có chủ ý): không import latency_attack (không chạy tối ưu tấn công), không gọi git kiểm checkout
(thay bằng kiểm SHA256 trọng số), đặt sys.dont_write_bytecode để không ghi __pycache__ vào outputs/.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch as mock_patch

from .cohorts import REPO

VENDOR = REPO / 'outputs/vendor_underload'
PACKAGE = VENDOR / 'robust_pkgs/ultralytics-yolov5-robust-0.0.1/yolov5'
WEIGHT = REPO / 'outputs/weights/reference/yolov5s.pt'
WEIGHT_SHA256 = '8b3b748c1e592ddd8868022e8732fde20025197328490623cc16c6f24d0782ee'  # reference/manifest.json:protocol.weights_sha256


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__file__ = str(path / '__init__.py')
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module


def load():
    """Trả về (model float32 eval, LoadImages). Fail-closed nếu trọng số đổi."""
    import torch
    sys.dont_write_bytecode = True
    if sha256(WEIGHT) != WEIGHT_SHA256:
        raise RuntimeError('Trọng số YOLOv5s tham chiếu đã đổi')
    os.environ['YOLOv5_AUTOINSTALL'] = 'false'
    os.environ['YOLOV5_CONFIG_DIR'] = str(REPO / 'outputs/runtime/reference_config')
    sys.path.insert(0, str(REPO / 'outputs/runtime/reference'))
    sys.path.insert(0, str(PACKAGE))
    _namespace('yolov5', PACKAGE)
    import models.yolo as yolo
    if not hasattr(yolo, 'DetectionModel'):
        yolo.DetectionModel = yolo.Model
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.dataloaders import LoadImages
    original_load = torch.load

    def fixed_checkpoint_load(path, *args, **kwargs):
        if Path(path).resolve() != WEIGHT.resolve():
            raise RuntimeError('Unexpected reference checkpoint path')
        kwargs['weights_only'] = False
        return original_load(path, *args, **kwargs)

    with mock_patch.object(torch, 'load', fixed_checkpoint_load):
        model = attempt_load(str(WEIGHT), torch.device('cpu'), False).float().eval()
    model.requires_grad_(False)
    return model, LoadImages


def decode(raw):
    """raw [N, 85] (xywh, obj, 80 lớp) -> [N, 6] = x1,y1,x2,y2, score = obj * max lớp, lớp. Như cách đếm của mã gốc (obj * cls)."""
    import torch
    xy, wh = raw[:, :2], raw[:, 2:4]
    cls_scores, cls = (raw[:, 4:5] * raw[:, 5:]).max(1)
    return torch.cat([xy - wh / 2, xy + wh / 2, cls_scores[:, None], cls[:, None].float()], 1)
