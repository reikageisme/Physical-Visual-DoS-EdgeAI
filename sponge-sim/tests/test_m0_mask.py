"""Kiểm M0 KHÔNG cần trọng số seg: builder mask tất định, công đoạn process_mask chạy đúng hình dạng,
logic quyết định go/no-go. Phần forward (cần trọng số) kiểm bằng smoke khi có yolov8n-seg.pt."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / 'src'))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SIM / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load('m0_run', 'studies/m0_mask_microbench/run.py')
an = _load('m0_an', 'studies/m0_mask_microbench/analyze.py')


def test_make_masks_deterministic_and_shapes():
    a = run.make_masks(300, 160, 32, 640, 5100, 5101)
    b = run.make_masks(300, 160, 32, 640, 5100, 5101)
    assert a[3] == b[3]  # SHA trùng -> tất định
    protos, coeffs, boxes, _ = a
    assert tuple(protos.shape) == (32, 160, 160)
    assert tuple(coeffs.shape) == (300, 32)
    assert tuple(boxes.shape) == (300, 4)
    assert (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()


def test_make_masks_seed_changes_sha():
    assert run.make_masks(300, 160, 32, 640, 5100, 5101)[3] != run.make_masks(300, 160, 32, 640, 9999, 5101)[3]


def test_process_mask_runs_for_K_grid_both_retina():
    protos, coeffs, boxes, _ = run.make_masks(300, 160, 32, 640, 5100, 5101)
    for retina in (False, True):
        fn = run.mask_fn(retina)
        for k in (0, 10, 50, 100, 300):
            out = fn(protos, coeffs[:k].contiguous(), boxes[:k].contiguous(), (640, 640))
            assert out.shape[0] == k
            if k > 0:
                # retina (process_mask_native) phóng lên full-res; không retina giữ độ phân giải proto
                assert out.shape[-1] == (640 if retina else 160)


def test_analyze_selftest_passes():
    assert an.selftest()
