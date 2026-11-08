"""Kiểm M3 KHÔNG cần trọng số/GPU: hàm mục tiêu proxy khả vi và đơn điệu; logic quyết định.
Hiệu lực đẩy K thật chỉ xác nhận bằng smoke GPU khi có máy."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM / 'src'))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SIM / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = _load('m3_run', 'studies/m3_attack/run.py')
an = _load('m3_an', 'studies/m3_attack/analyze.py')


def _preds(cls_val, nc=80, nm=32, A=50):
    box = torch.zeros(1, 4, A)
    cls = torch.full((1, nc, A), float(cls_val))
    mask = torch.zeros(1, nm, A)
    return torch.cat([box, cls, mask], 1)


def test_soft_count_differentiable():
    cls = torch.zeros(1, 80, 50, requires_grad=True)
    preds = torch.cat([torch.zeros(1, 4, 50), torch.sigmoid(cls), torch.zeros(1, 32, 50)], 1)
    val = run.soft_candidate_count(preds, 80, 0.25, 0.02)
    val.backward()
    assert cls.grad is not None and bool(torch.isfinite(val))


def test_soft_count_monotone_in_confidence():
    lo = float(run.soft_candidate_count(_preds(0.1), 80, 0.25, 0.02))
    hi = float(run.soft_candidate_count(_preds(0.9), 80, 0.25, 0.02))
    assert hi > lo and hi > 40 and lo < 5   # 0.9 >> ngưỡng 0.25 (gần A=50); 0.1 << ngưỡng (gần 0)


def test_find_preds_picks_right_tensor():
    raw = ((torch.zeros(1, 116, 8400), torch.zeros(1, 32, 160, 160)), {'x': torch.zeros(1, 5)})
    t = run._find_preds(raw, 116)
    assert tuple(t.shape) == (1, 116, 8400)


def test_analyze_selftest():
    assert an.selftest()
