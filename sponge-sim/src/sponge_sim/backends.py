"""4 backend NMS cùng giao diện, cùng đầu vào (ứng viên SAU lọc tau).

run(boxes_xyxy[N,4], scores[N], classes[N], iou) -> chỉ số giữ lại (theo thứ tự score giảm, tối đa MAX_OUT).

Nhóm agnostic: torch_pure, tv_nms, ultra_agnostic. Nhóm class-aware: tv_batched, ultra_native.
Tie-break: torch_pure dùng argsort(descending=True) của torch; torchvision dùng sort nội bộ C++;
Ultralytics argsort khi n > max_nms rồi torchvision. Khớp trên mẫu hữu hạn KHÔNG có nghĩa tương đương mọi tie.
"""
from __future__ import annotations

import sys

import torch
import torchvision

from .provenance import MAX_OUT, nms_torch_pure, xywh_to_xyxy  # noqa: F401


class Backend:
    """prepare() chạy NGOÀI vùng đo (dựng đầu vào riêng của backend); run() là phần được đo.
    Gọi prepare() mới trước MỖI lần run() (Ultra ghi đè đầu vào tại chỗ)."""
    name = ''
    class_aware = False

    def prepare(self, boxes, scores, classes, nc=80):
        return (boxes, scores, classes)

    def run(self, prepared, iou):
        raise NotImplementedError

    def __call__(self, boxes, scores, classes, iou, nc=80):
        return self.run(self.prepare(boxes, scores, classes, nc), iou)


class TorchPure(Backend):
    name, class_aware = 'torch_pure', False  # vòng lặp Python (bản copy của repo cũ)
    def run(self, p, iou):
        return nms_torch_pure(p[0], p[1], iou)[:MAX_OUT]


class TvNms(Backend):
    name, class_aware = 'tv_nms', False
    def run(self, p, iou):
        return torchvision.ops.nms(p[0], p[1], iou)[:MAX_OUT]


class TvBatched(Backend):
    name, class_aware = 'tv_batched', True
    def prepare(self, boxes, scores, classes, nc=80):
        return (boxes, scores, classes.long())
    def run(self, p, iou):
        return torchvision.ops.batched_nms(p[0], p[1], p[2], iou)[:MAX_OUT]


class Ultra(Backend):
    """Dựng lại tensor head [1, 4+nc, N] (xywh, score one-hot theo lớp) rồi gọi
    ultralytics.utils.nms.non_max_suppression với mặc định Ultralytics
    (max_det=300, max_nms=30000, max_time_img=0.05), return_idxs=True để lấy chỉ số đầu vào.
    Lưu ý: vòng xywh->xyxy làm sai số float nhỏ ở toạ độ so với đầu vào xyxy.
    """
    def __init__(self, agnostic):
        self.agnostic = agnostic
        self.class_aware = not agnostic
        self.name = 'ultra_agnostic' if agnostic else 'ultra_native'

    def prepare(self, boxes, scores, classes, nc=80):
        n = len(scores)
        xywh = torch.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
        xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
        cls_scores = torch.zeros(nc, n)
        if n:
            cls_scores[classes.long(), torch.arange(n)] = scores
        return (torch.cat([xywh.T, cls_scores], 0)[None].contiguous(), nc)

    def run(self, p, iou):
        from ultralytics.utils.nms import non_max_suppression
        pred, nc = p
        if pred.shape[-1] == 0:
            return torch.zeros(0, dtype=torch.long)
        # non_max_suppression ghi đè `pred` tại chỗ -> harness PHẢI gọi prepare() mới trước mỗi lần run().
        _, idxs = non_max_suppression(pred, conf_thres=0.0, iou_thres=iou, agnostic=self.agnostic,
                                      max_det=MAX_OUT, nc=nc, return_idxs=True)
        return idxs[0].long()


BACKENDS = {b.name: b for b in (TorchPure(), TvNms(), TvBatched(), Ultra(agnostic=False))}
EXTRA = {'ultra_agnostic': Ultra(agnostic=True)}  # chỉ dùng để test khớp nhóm agnostic
CLASS_AWARE = {k: v.class_aware for k, v in {**BACKENDS, **EXTRA}.items()}


def metadata() -> dict:
    """Ghi vào manifest: phiên bản + đường cài đặt thực tế của Ultralytics."""
    import ultralytics
    return dict(torch=torch.__version__, torchvision=torchvision.__version__,
                ultralytics=ultralytics.__version__,
                ultra_impl='torchvision.ops.nms' if 'torchvision' in sys.modules else 'TorchNMS',
                ultra_defaults=dict(max_det=MAX_OUT, max_nms=30000, max_time_img=0.05,
                                    time_limit_s='2.0 + 0.05*batch'),
                class_aware=CLASS_AWARE, max_out=MAX_OUT)
