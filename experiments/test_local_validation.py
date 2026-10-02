"""Small regression checks. Run: python experiments/test_local_validation.py"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.eot_transforms import AUG_LIST, apply_eot
from core.victim_model import VictimModel
from local_coco128_validation import verify_metrics
from local_q1_statistics import paired_interval


class LocalValidationTests(unittest.TestCase):
    def test_noncontiguous_eot_and_gradient(self):
        torch.manual_seed(42)
        leaf = torch.rand(2, 32, 32, 3, requires_grad=True)
        image = leaf.permute(0, 3, 1, 2)
        self.assertFalse(image.is_contiguous())
        previous = AUG_LIST[2].p
        try:
            AUG_LIST[2].p = 1.0
            output = apply_eot(image)
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(torch.isfinite(output).all())
            output.mean().backward()
            self.assertTrue(torch.isfinite(leaf.grad).all())
        finally:
            AUG_LIST[2].p = previous

    def test_nms_matches_known_fixture(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.], [20., 20., 30., 30.]])
        scores = torch.tensor([.9, .8, .7])
        result = VictimModel._nms(boxes, scores, .45)
        self.assertEqual(result.tolist(), [0, 2])
        self.assertTrue(torch.equal(result, torchvision.ops.nms(boxes, scores, .45)))

    def test_detection_metric_fixtures(self):
        verify_metrics()

    def test_constant_paired_bootstrap(self):
        result = paired_interval([2, 3, 4], [1, 2, 3], np.random.default_rng(0), draws=1000)
        self.assertEqual(result["mean_difference"], 1.)
        self.assertEqual(result["percentile_ci95"], [1., 1.])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
