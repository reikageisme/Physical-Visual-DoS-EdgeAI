"""Tái sinh decoded cho vài ảnh X1-500 bằng forward mới -> khớp record và raw_cache đã lưu.
Fail-closed: lệch là lỗi; không nới tolerance để pass."""
import json

import numpy as np
import pytest
import torch

from conftest import REPO
from sponge_sim import cohorts, provenance as pv

DATA = REPO / 'outputs/datasets/coco_val2017_extended'
N_IMAGES = 6


@pytest.fixture(scope='module')
def model():
    from ultralytics import YOLO
    torch.set_num_threads(2)
    return YOLO(str(REPO / 'yolov8n.pt')).model.eval().float()


@pytest.fixture(scope='module')
def files():
    coco = json.loads((DATA / 'instances_subset.json').read_text(encoding='utf-8'))
    return {im['id']: im['file_name'] for im in coco['images']}


@pytest.mark.parametrize('name,seed,q', [('clean', 0, 0), ('sg_ga', 0, 2500)])
def test_regenerated_counts_match(model, files, name, seed, q):
    case = cohorts.read_case('X1-500', name, seed, q)
    recs = cohorts.records(case)
    patch = None
    if name != 'clean':
        patch = torch.from_numpy(np.load(cohorts.COHORTS['X1-500'] / f'optimization/{name}_seed{seed}/patch_q{q}.npy'))
    cache_dir = cohorts.COHORTS['X1-500'] / f'evaluation/raw_cache/{name}_seed{seed}_q{q}'
    for image_id in sorted(recs)[:N_IMAGES]:
        r = recs[image_id]
        image = pv.load_image(DATA / 'images' / files[image_id])
        if patch is not None:
            image = pv.insert(image, patch[None], [tuple(r['location'])])
        with torch.no_grad():
            raw = model(image)
        raw = raw[0] if isinstance(raw, (tuple, list)) else raw
        decoded = pv.decode(raw[0])
        cached = torch.from_numpy(np.load(cache_dir / f'{image_id}.npz')['decoded'])
        info, _ = pv.postprocess(decoded, .01, None)
        info_c, _ = pv.postprocess(cached, .01, None)
        assert info_c['active_candidates'] == r['active_candidates'], ('cache', image_id)
        assert info['active_candidates'] == r['active_candidates'], ('regen', image_id)
        assert info['final_detections'] == r['final_detections'], ('regen-final', image_id)
        assert torch.allclose(decoded, cached, atol=1e-4), image_id
