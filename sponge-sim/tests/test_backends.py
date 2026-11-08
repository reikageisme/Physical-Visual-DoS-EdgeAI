"""Khớp tập giữ lại trên MẪU HỮU HẠN trong từng nhóm. Không tuyên bố tương đương mọi tie.

Tie-break: torch_pure dùng argsort(descending) của torch; torchvision sort nội bộ C++.
Fixture dùng score phân biệt (kiểm bằng assert) để so tập; tie không được phủ bởi test này.
"""
import pytest
import torch

from sponge_sim import inject
from sponge_sim.backends import BACKENDS, EXTRA
from sponge_sim.provenance import IOU, tau_filter

AGNOSTIC = ['torch_pure', 'tv_nms', 'ultra_agnostic']
AWARE = ['tv_batched', 'ultra_native']
ALL = {**BACKENDS, **EXTRA}


def _cands(n, geom, seed, nc):
    t, _ = inject.make(n, geom, seed, anchors=2100, nc=nc)
    active, _ = tau_filter(t, .01)
    return active[:, :4].contiguous(), active[:, 4].contiguous(), active[:, 5].contiguous()


@pytest.mark.parametrize('geom', ['disjoint', 'dense_cluster', 'mixed'])
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('nc', [1, 80])
def test_groups_keep_same_set(geom, seed, nc):
    boxes, scores, classes = _cands(250, geom, seed, nc)
    assert len(torch.unique(scores)) == len(scores)
    for group in (AGNOSTIC, AWARE):
        sets = {name: set(ALL[name](boxes, scores, classes, IOU, nc=nc).tolist()) for name in group}
        ref = sets[group[0]]
        for name, s in sets.items():
            assert s == ref, (name, len(s), len(ref))


def test_disjoint_keeps_all_up_to_cap():
    boxes, scores, classes = _cands(250, 'disjoint', 0, 80)
    for name, b in ALL.items():
        assert len(b(boxes, scores, classes, IOU)) == 250, name


def test_empty_input():
    e = torch.zeros(0, 4), torch.zeros(0), torch.zeros(0)
    for name, b in ALL.items():
        assert len(b(*e, IOU)) == 0, name


def test_injection_counts_and_determinism():
    for n in (0, 10, 2100):
        t1, m1 = inject.make(n, 'mixed', 7)
        t2, m2 = inject.make(n, 'mixed', 7)
        assert m1['sha256'] == m2['sha256']
        assert int((t1[:, 4] > .01).sum()) == n


@pytest.mark.parametrize('n', [0, 1, 5, 6, 7])
def test_small_n_including_six_keeps_same_set(n):
    """N == 6 làm ultralytics nhầm tensor với đầu ra end-to-end; backend phải bù cột đệm và vẫn khớp nhóm."""
    g = torch.Generator().manual_seed(n)
    boxes = torch.rand(n, 4, generator=g) * 100
    boxes[:, 2:] += boxes[:, :2] + 5
    scores = torch.rand(n, generator=g) * 0.9 + 0.1
    classes = torch.zeros(n)
    ref = set(ALL['tv_batched'](boxes, scores, classes, IOU).tolist())
    assert set(ALL['ultra_native'](boxes, scores, classes, IOU).tolist()) == ref
