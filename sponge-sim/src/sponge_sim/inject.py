"""Tiêm ứng viên tổng hợp tại ranh giới decoded (sau decode, trước lọc tau). Không cần patch.

Đầu ra cùng định dạng provenance.decode: [A, 6] = x1, y1, x2, y2, score, class trên khung CANVAS×CANVAS.
A = số anchor của head: 2100 ứng với đầu vào 320×320 của pipeline (40²+20²+10²); 8400 ứng với 640×640.
N hàng có score > tau (ứng viên hoạt động), A−N hàng còn lại score < tau (nền của head).

Hình học:
- disjoint: lưới, hộp không chồng nhau (IoU = 0) -> NMS giữ hết (trường hợp xấu nhất cho vòng lặp greedy);
- dense_cluster: hộp rung quanh ít tâm, IoU với tâm cụm > .45 -> phần lớn bị loại sớm;
- mixed: nửa disjoint, nửa dense_cluster (trộn ngẫu nhiên thứ tự).
Tất định theo seed; trả kèm SHA256 tensor để ghi raw thay vì lưu toàn bộ hộp.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

CANVAS = 320
HEAD_ANCHORS = {320: 2100, 640: 8400}


def _disjoint(n, rng, canvas):
    if n == 0:
        return np.zeros((0, 4))
    g = math.ceil(math.sqrt(n))
    cell = canvas / g
    idx = rng.permutation(g * g)[:n]
    r, c = idx // g, idx % g
    side = cell * 0.8
    x1 = c * cell + cell * 0.1
    y1 = r * cell + cell * 0.1
    return np.stack([x1, y1, x1 + side, y1 + side], 1)


def _dense(n, rng, canvas, clusters=8, side=48.0, jitter=2.0):
    if n == 0:
        return np.zeros((0, 4))
    centers = rng.uniform(side, canvas - side, (clusters, 2))
    which = rng.integers(0, clusters, n)
    c = centers[which] + rng.uniform(-jitter, jitter, (n, 2))
    s = side + rng.uniform(-jitter, jitter, (n, 1))
    return np.concatenate([c - s / 2, c + s / 2], 1)


def make(n: int, geometry: str, seed: int, anchors: int = 2100, tau: float = .01, nc: int = 80,
         canvas: int = CANVAS) -> tuple[torch.Tensor, dict]:
    if not 0 <= n <= anchors:
        raise ValueError(f'N={n} vượt số anchor {anchors}')
    rng = np.random.default_rng(seed)
    if geometry == 'disjoint':
        boxes = _disjoint(n, rng, canvas)
    elif geometry == 'dense_cluster':
        boxes = _dense(n, rng, canvas)
    elif geometry == 'mixed':
        k = n // 2
        boxes = np.concatenate([_disjoint(k, rng, canvas), _dense(n - k, rng, canvas)])
        boxes = boxes[rng.permutation(n)] if n else boxes
    else:
        raise ValueError(geometry)
    scores = rng.uniform(max(tau, .01) + 1e-3, 1.0, n)
    classes = rng.integers(0, nc, n)
    bg = anchors - n
    bg_boxes = _dense(bg, rng, canvas) if bg else np.zeros((0, 4))
    bg_scores = rng.uniform(0, tau * 0.5, bg)
    bg_classes = rng.integers(0, nc, bg)
    rows = np.concatenate([
        np.concatenate([boxes, scores[:, None], classes[:, None]], 1),
        np.concatenate([bg_boxes, bg_scores[:, None], bg_classes[:, None]], 1),
    ])
    rows = rows[rng.permutation(anchors)]  # ứng viên rải khắp anchor như head thật
    t = torch.from_numpy(rows.astype(np.float32))
    meta = dict(n=n, geometry=geometry, seed=seed, anchors=anchors, tau=tau, nc=nc, canvas=canvas,
                sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
    return t, meta
