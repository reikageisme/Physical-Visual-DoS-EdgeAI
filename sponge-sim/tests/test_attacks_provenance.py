"""Hồi quy: attacks_provenance.optimize() chạy lại (CPU, budget 400, seed 0, canvas 320) phải khớp
patch đã lưu trong outputs/x1_500img_20261018 (cùng thuật toán, cùng seed, cùng ảnh train, cùng CPU
-> cùng RNG stream). Chạy chậm (~3-6 phút/case): đánh dấu slow, bỏ qua khi laptop đang đo thời gian."""
import json

import numpy as np
import pytest
import torch

from conftest import REPO
from sponge_sim import attacks_provenance as ap

DATA = REPO / 'outputs/datasets/coco128/images/train2017'
pytestmark = pytest.mark.slow


def _train_images(paths):
    from sponge_sim.provenance import load_image
    return [load_image(p) for p in paths]


@pytest.fixture(scope='module')
def victim():
    """Tái lập đúng môi trường lượt X1-500: manifest.protocol.torch_threads = 2
    (experiments/extended_validation.py:238-239) và tham số requires_grad False (:270-273).
    Kiểm 21/10/2026: gradient_tanh KHÔNG khớp patch đã lưu khi dùng số thread mặc định (12) hoặc 1
    (chênh tối đa 0,0549 = 14/255, do sign(grad) lật vì thứ tự cộng số thực theo thread); khớp tuyệt đối với 2."""
    import sys
    sys.path.insert(0, str(REPO))
    from core.victim_model import VictimModel
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    v = VictimModel(device='cpu')
    v.model.eval()
    for p in v.model.parameters():
        p.requires_grad_(False)
    yield v
    torch.set_num_threads(old)


@pytest.mark.parametrize('name,method,location', [('sg_ga', 'sg_ga', 'saliency'),
                                                   ('tanh_whitebox_patch', 'gradient_tanh', 'saliency')])
def test_reruns_match_saved_patch_q400_seed0(tmp_path, victim, name, method, location):
    manifest = json.loads((REPO / 'outputs/x1_500img_20261018/manifest.json').read_text(encoding='utf-8'))
    train_paths = [REPO / p['path'] for p in manifest['protocol']['training_images']]
    assert len(train_paths) == 4
    images = _train_images(train_paths)
    spec = dict(name=name, method=method, location=location, size=64, eot='full', draws=1, budgets=[400], canvas=320)
    result = ap.optimize(victim, images, spec, seed=0, run_dir=tmp_path / name)
    got = np.load(tmp_path / name / 'patch_q400.npy')
    want = np.load(REPO / f'outputs/x1_500img_20261018/optimization/{name}_seed0/patch_q400.npy')
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)
    assert result['queries'] == 400
