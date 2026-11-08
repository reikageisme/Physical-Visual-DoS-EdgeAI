"""Các điểm tổng quát hóa trong attacks_provenance KHÔNG đổi hành vi gốc ở cấu hình gốc (canvas 320, ngưỡng .01, CPU),
và đúng ở canvas 640."""
import sys

import numpy as np
import pytest
import torch

from conftest import REPO
from sponge_sim import attacks_provenance as ap
from sponge_sim import provenance as pv

IMG_DIR = REPO / 'outputs/datasets/coco128/images/train2017'
FILES = ['000000000009.jpg', '000000000030.jpg']


@pytest.fixture(scope='module')
def original():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / 'experiments'))
    import extended_validation as ev   # chỉ import để so sánh, không sửa
    return ev


def test_load_image_canvas_320_equals_original_loader():
    for f in FILES:
        assert torch.equal(ap.load_image_canvas(IMG_DIR / f, 320), pv.load_image(IMG_DIR / f))


@pytest.mark.parametrize('mode', ['center', 'random', 'saliency'])
@pytest.mark.parametrize('size', [64, 45, 128])
def test_find_locations_canvas_320_equals_original(original, mode, size):
    images = [pv.load_image(IMG_DIR / f) for f in FILES]
    assert ap.find_locations(images, size, mode, 7, canvas=320) == original.find_locations(images, size, mode, 7)


@pytest.mark.parametrize('mode,size', [('center', 128), ('random', 128), ('saliency', 128), ('saliency', 64)])
def test_find_locations_canvas_640_in_bounds(mode, size):
    images = [ap.load_image_canvas(IMG_DIR / f, 640) for f in FILES]
    locs = ap.find_locations(images, size, mode, 3, canvas=640)
    for y, x in locs:
        assert 0 <= y <= 640 - size and 0 <= x <= 640 - size
    if mode == 'center':
        assert locs[0] == [(640 - size) // 2] * 2


def test_objective_threshold_is_forwarded():
    scores = torch.tensor([[0.02, 0.3, 0.6, 0.0]])
    f01 = ap.objective(scores, None, 0.01)
    f25 = ap.objective(scores, None, 0.25)
    assert float(f01) != float(f25)
    assert float(f25) == pytest.approx(0.3 + 0.6 + 1.5 * 2)
