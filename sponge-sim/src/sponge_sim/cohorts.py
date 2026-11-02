"""Đọc raw cohort đã lưu (CHỈ ĐỌC). Tên cohort thống nhất theo CLAUDE.md.

OLD128 = outputs/extended_q1_20261002 ; X1-500 = outputs/x1_500img_20261018.
Bộ lọc mặc định: threshold == .01 và pre_nms_cap is None (không lấy record đầu vì đã shuffle).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
COHORTS = {'OLD128': REPO / 'outputs/extended_q1_20261002', 'X1-500': REPO / 'outputs/x1_500img_20261018'}


def case_path(cohort: str, name: str, seed: int, q: int) -> Path:
    return COHORTS[cohort] / 'evaluation' / f'{name}_seed{seed}_q{q}.json'


def read_case(cohort: str, name: str, seed: int, q: int) -> dict:
    return json.loads(case_path(cohort, name, seed, q).read_text(encoding='utf-8'))


def records(case: dict, tau: float = .01, cap=None) -> dict[int, dict]:
    return {r['image_id']: r for r in case['records'] if r['threshold'] == tau and r['pre_nms_cap'] == cap}


def mean_image_ratio(case: dict, clean: dict, key: str = 'active_candidates', ids=None) -> float:
    """Tỷ số theo ảnh rồi mới trung bình; mẫu số max(clean, 1)."""
    a, b = records(case), records(clean)
    keys = sorted(set(a) & set(b) if ids is None else set(ids) & set(a) & set(b))
    assert keys
    return float(np.mean([a[i][key] / max(b[i][key], 1) for i in keys]))
