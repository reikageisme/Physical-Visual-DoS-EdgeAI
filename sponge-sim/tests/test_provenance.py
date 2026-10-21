"""Bản copy trong provenance.py == bản gốc (import read-only), và SHA256 nguồn không đổi."""
import hashlib
import sys

import pytest
import torch

from conftest import REPO
from sponge_sim import provenance as pv


@pytest.mark.parametrize('rel,sha', list(pv.SOURCES.items()))
def test_source_sha_unchanged(rel, sha):
    assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha


def _original_victim_cls():
    sys.path.insert(0, str(REPO))
    from core.victim_model import VictimModel  # chỉ import để so sánh, không sửa
    return VictimModel


def _fixture(seed, n=400):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 280
    wh = torch.rand(n, 2, generator=g) * 40 + 4
    boxes = torch.cat([xy, xy + wh], 1)
    scores = torch.rand(n, generator=g)
    scores[::7] = 0.5  # ép tie điểm số
    return boxes, scores


@pytest.mark.parametrize('seed', range(5))
def test_nms_copy_matches_original(seed):
    V = _original_victim_cls()
    boxes, scores = _fixture(seed)
    assert torch.equal(pv.nms_torch_pure(boxes, scores, .45), V._nms(boxes, scores, .45))


def test_xywh_copy_matches_original():
    V = _original_victim_cls()
    b = torch.rand(50, 4) * 100
    assert torch.equal(pv.xywh_to_xyxy(b), V._xywh_to_xyxy(b))


def test_decode_shape_and_agnostic_score():
    raw = torch.rand(84, 2100)
    d = pv.decode(raw)
    assert d.shape == (2100, 6)
    assert torch.equal(d[:, 4], raw[4:].max(0).values)
