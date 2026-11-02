"""Formula, gradient, NMS and exact query-accounting fixtures; no real model.

These tests are not scientific performance measurements. Author formulas are
tested using hand-computable tensors, with no imported/copied vendor attack.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from extended_phantom_local import (raw_tensor, target_margin, classwise_detections,
                                   preservation_loss, phantom_loss, optimize)


def setUpModule():
    # Small deterministic fixtures, not a multithreaded timing experiment.
    torch.set_num_threads(1)


def prediction(boxes, classes, scores):
    raw = torch.zeros(1, 7, len(boxes))
    for index, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        raw[0, :4, index] = torch.tensor(box)
        raw[0, 4+cls, index] = score
    return raw


class PhantomFormulaFixtures(unittest.TestCase):
    def test_author_margin_uses_all_slots_and_inactive_mask(self):
        raw = prediction([[5, 5, 2, 2]]*4, [2, 2, 1, 2], [.05, .10, .60, .25]).requires_grad_(True)
        loss = target_margin(raw)
        # 0.25-0.05 + 0.25-0.10, divided by all four slots.
        self.assertAlmostEqual(float(loss.detach()), .35/4, places=7)
        gradient, = torch.autograd.grad(loss, raw)
        self.assertTrue(torch.allclose(gradient[0, 6], torch.tensor([-.25, -.25, 0., 0.])))
        self.assertEqual(float(gradient[:, :6].abs().sum()), 0.)

    def test_classwise_nms_does_not_suppress_other_class(self):
        raw = prediction([[20, 20, 10, 10]]*3, [0, 0, 2], [.9, .8, .7])
        found = classwise_detections(raw, .25, 1000)[0]
        self.assertEqual(len(found), 2)
        self.assertEqual(set(found[:, 5].tolist()), {0., 2.})
        self.assertTrue(torch.allclose(found[:, :4], torch.tensor([[15., 15., 25., 25.]]*2)))
        self.assertEqual(len(classwise_detections(raw, .25, 1)[0]), 1)

    def test_same_class_best_iou_and_missing_detection(self):
        clean = prediction([[20, 20, 10, 10], [50, 50, 10, 10]], [0, 1], [.9, .8])
        adv = prediction([[20, 20, 10, 10], [50, 50, 10, 10]], [0, 2], [.9, .8]).requires_grad_(True)
        # First clean object retained, second has no same-class detection.
        loss = preservation_loss(clean, adv)
        self.assertAlmostEqual(float(loss.detach()), .5, places=7)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, adv)[0]).all())

    def test_preservation_averages_detections_not_image_means(self):
        first = prediction([[20, 20, 10, 10], [50, 50, 10, 10]], [0, 1], [.9, .8])
        second = prediction([[20, 20, 10, 10], [50, 50, 10, 10]], [2, 1], [.9, .0])
        clean = torch.cat([first, second])
        adv = clean.clone()
        adv[0, 4:, :] = 0
        # Three original detections, exactly one preserved: loss=2/3.
        self.assertAlmostEqual(float(preservation_loss(clean, adv)), 2/3, places=7)

    def test_empty_clean_constant_and_default_loss_weights(self):
        clean = prediction([[20, 20, 10, 10]], [0], [0.])
        adv = prediction([[20, 20, 10, 10]], [2], [.1]).requires_grad_(True)
        loss, margin, preservation = phantom_loss(clean, adv)
        self.assertEqual(float(preservation.detach()), 1.)
        self.assertAlmostEqual(float(loss.detach()), .75*.15+.25, places=7)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, adv)[0]).all())

    def test_empty_adversarial_and_degenerate_union_are_finite(self):
        clean = prediction([[20, 20, 10, 10]], [0], [.9])
        adv = prediction([[20, 20, 10, 10]], [0], [0.]).requires_grad_(True)
        self.assertEqual(float(preservation_loss(clean, adv).detach()), 1.)
        invalid = prediction([[20, 20, 0, 0]], [0], [.9]).requires_grad_(True)
        loss = preservation_loss(invalid.detach(), invalid)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, invalid)[0]).all())

    def test_prediction_adapter_is_explicit_and_rejects_nonfinite(self):
        raw = prediction([[1, 1, 2, 2]], [2], [.1])
        self.assertIs(raw_tensor((raw, [])), raw)
        with self.assertRaises(ValueError):
            raw_tensor(torch.zeros(1, 84))
        raw[0, 4, 0] = float('nan')
        with self.assertRaises(ValueError):
            raw_tensor(raw)
        with self.assertRaises(ValueError):
            target_margin(torch.zeros(1, 7, 1), target_class=3)


class CountingVictim:
    def __init__(self):
        self.calls, self.image_forwards = [], 0

    def model(self, images):
        self.calls.append(images.detach().clone())
        self.image_forwards += len(images)
        # Low-confidence boxes: target-margin has a nonzero input gradient;
        # no clean detections means the preservation term is constant.
        scores = images.mean(dim=(1, 2, 3))*.1
        boxes = images.new_tensor([20., 20., 10., 10.])[None, :, None].expand(len(images), -1, 1)
        classes = torch.stack([scores*0, scores*0, scores], dim=1).unsqueeze(-1)
        return torch.cat([boxes, classes], dim=1)


class PhantomQueryFixtures(unittest.TestCase):
    def test_clean_patch_eot_draws_are_paired(self):
        images = [torch.full((1, 3, 320, 320), .2) for _ in range(4)]
        victim = CountingVictim()
        with tempfile.TemporaryDirectory(prefix='phantom-paired-fixture-') as directory:
            folder = Path(directory)
            with patch('extended_validation.make_eot', return_value=lambda image:image+torch.randn_like(image)*.001):
                result = optimize(victim, images, 2, folder, 'paired-fixture', budget=8)
            mask = torch.ones(4, 3, 320, 320, dtype=torch.bool)
            for index, (y, x) in enumerate(result['locations']):
                mask[index, :, y:y+64, x:x+64] = False
            self.assertTrue(torch.equal(victim.calls[0][mask], victim.calls[1][mask]))
            self.assertEqual(result['history'][0]['eot_draw_seed'], 272013)

    def test_interrupted_resume_preserves_count_and_evaluated_iterate(self):
        images = [torch.full((1, 3, 320, 320), .2) for _ in range(4)]
        first, resumed, uninterrupted = CountingVictim(), CountingVictim(), CountingVictim()
        original_model = first.model
        def fail_after_first_step(image):
            if first.image_forwards == 8:
                raise RuntimeError('fixture-interruption')
            return original_model(image)
        first.model = fail_after_first_step
        with tempfile.TemporaryDirectory(prefix='phantom-resume-fixture-') as directory:
            folder = Path(directory)
            with patch('extended_validation.make_eot', return_value=lambda image:image):
                with self.assertRaisesRegex(RuntimeError, 'fixture-interruption'):
                    optimize(first, images, 3, folder/'resumed', 'resume-fixture', budget=16)
                result = optimize(resumed, images, 3, folder/'resumed', 'resume-fixture', budget=16)
                full = optimize(uninterrupted, images, 3, folder/'full', 'resume-fixture', budget=16)
            self.assertEqual(first.image_forwards+resumed.image_forwards, 16)
            self.assertEqual(result['history'], full['history'])
            self.assertEqual(result['patch_sha256'], full['patch_sha256'])

    def test_clean_plus_patch_forwards_and_saved_last_measured_iterate(self):
        images = [torch.full((1, 3, 320, 320), .2) for _ in range(4)]
        victim = CountingVictim()
        with tempfile.TemporaryDirectory(prefix='phantom-local-fixture-') as directory:
            folder = Path(directory)
            with patch('extended_validation.make_eot', return_value=lambda image:image):
                result = optimize(victim, images, 1, folder, 'fixture-protocol', budget=16)
            self.assertEqual(victim.image_forwards, 16)
            self.assertEqual([len(call) for call in victim.calls], [4]*4)
            self.assertEqual(result['clean_image_forwards'], 8)
            self.assertEqual(result['patched_image_forwards'], 8)
            self.assertEqual(result['backward_calls'], 2)
            self.assertEqual([h['image_forwards'] for h in result['history']], [8, 16])
            saved = torch.from_numpy(np.load(folder/'patch_q16.npy'))
            y, x = result['locations'][0]
            self.assertTrue(torch.equal(saved, victim.calls[-1][0, :, y:y+64, x:x+64]))
            # The first sign step increases this fixture's target confidence.
            self.assertTrue(torch.all(saved >= victim.calls[1][0, :, y:y+64, x:x+64]))
            state = torch.load(folder/'resume.pt', weights_only=False)
            self.assertFalse(torch.equal(saved, state['patch']))
            # Completed/resumed calls do not perform additional forwards.
            repeated = optimize(victim, images, 1, folder, 'fixture-protocol', budget=16)
            self.assertEqual(repeated['patch_sha256'], result['patch_sha256'])
            self.assertEqual(victim.image_forwards, 16)
            with self.assertRaises(RuntimeError):
                optimize(victim, images, 1, folder, 'changed-protocol', budget=16)

    def test_nondivisible_budget_rejected_before_any_forward(self):
        victim = CountingVictim()
        with tempfile.TemporaryDirectory(prefix='phantom-budget-fixture-') as directory:
            with self.assertRaises(ValueError):
                optimize(victim, [torch.zeros(1, 3, 320, 320)]*4, 0,
                         Path(directory), 'fixture', budget=12)
        self.assertEqual(victim.image_forwards, 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
