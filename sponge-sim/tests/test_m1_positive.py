"""Kiểm M1 KHÔNG cần trọng số seg: builder mask tất định, gọi NMS thật (ops.non_max_suppression suy nm=32 từ bề rộng),
logic K*/ngân sách. Phần forward thật kiểm bằng smoke khi có yolov8n-seg.pt."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / 'src'))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SIM / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load('m1_run', 'studies/m1_mask_positive/run.py')
an = _load('m1_an', 'studies/m1_mask_positive/analyze.py')


def test_make_masks_deterministic():
    a = run.make_masks(300, 160, 32, 640, 6100, 6101)
    b = run.make_masks(300, 160, 32, 640, 6100, 6101)
    assert a[3] == b[3] and tuple(a[0].shape) == (32, 160, 160) and tuple(a[1].shape) == (300, 32)


def test_real_nms_on_synthetic_preds_carries_32_coeffs():
    """preds [1, 4+80+32, A]: ops.non_max_suppression phải trả det có 6+32 cột (đã mang hệ số mask)."""
    from ultralytics.utils.nms import non_max_suppression
    nc, nm, A = 80, 32, 8400
    rng = np.random.default_rng(0)
    box = rng.uniform(10, 600, (1, 4, A)).astype('float32')
    cls = rng.uniform(0, 1, (1, nc, A)).astype('float32')
    mask = rng.standard_normal((1, nm, A)).astype('float32')
    preds = torch.from_numpy(np.concatenate([box, cls, mask], 1))
    dets = non_max_suppression(preds, 0.25, 0.45, nc=nc, max_det=300)
    assert len(dets) == 1
    assert dets[0].shape[1] == 6 + nm          # x1,y1,x2,y2,conf,cls + 32 hệ số
    assert dets[0].shape[0] <= 300             # tôn trọng max_det


def test_process_mask_both_retina_shapes():
    protos, coeffs, boxes, _ = run.make_masks(300, 160, 32, 640, 6100, 6101)
    for retina, exp in ((False, 160), (True, 640)):
        out = run.mask_fn(retina)(protos, coeffs[:100].contiguous(), boxes[:100].contiguous(), (640, 640))
        assert out.shape[0] == 100 and out.shape[-1] == exp


def test_analyze_selftest_passes():
    assert an.selftest()


def test_timing_sync_hook_called_twice_per_rep():
    """Hook sync (dùng cho torch.cuda.synchronize) phải được gọi 2 lần/lần lặp (trước t0 và trước t1)."""
    from sponge_sim import timing
    calls = {'n': 0}
    def sync():
        calls['n'] += 1
    out = timing.timed(lambda: 1, repeats=3, sync=sync)
    assert len(out) == 3 and calls['n'] == 6   # 2 × 3 rep
    # sync=None (mặc định): không gọi, hành vi CPU cũ không đổi
    out2 = timing.timed(lambda: 1, repeats=2)
    assert len(out2) == 2 and all(r['ns'] >= 0 for r in out2)
