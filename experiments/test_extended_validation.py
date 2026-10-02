"""Deterministic fixtures for the new protocol; these are not attack results."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from extended_validation import find_locations, insert, make_eot, objective, optimize, specifications


class ToyVictim:
    def get_raw_predictions(self, x):
        return x.mean(dim=(1, 2, 3))[:, None].expand(-1, 8)


class ExtendedFixtures(unittest.TestCase):
    def test_insertion_preserves_outside_and_gradient(self):
        images = torch.zeros(2, 3, 320, 320)
        patches = torch.ones(3, 3, 32, 32, requires_grad=True)
        x = insert(images, patches, [[0, 0], [288, 288]])
        self.assertEqual(x.shape, (6, 3, 320, 320))
        self.assertEqual(float(x.sum()), 6*3*32*32)
        x.sum().backward()
        self.assertTrue(torch.equal(patches.grad, torch.full_like(patches, 2)))

    def test_locations_bounded_and_deterministic(self):
        images = [torch.zeros(1, 3, 320, 320)]*4
        loc = find_locations(images, 128, 'random', 8)
        self.assertEqual(loc, find_locations(images, 128, 'random', 8))
        self.assertTrue(all(0 <= a <= 192 for pair in loc for a in pair))

    def test_budget_counts_actual_image_forwards(self):
        for spec in specifications():
            self.assertTrue(all(b % (4*spec['draws']) == 0 for b in spec['budgets']))
        torch.set_num_threads(2)
        images = [torch.zeros(1, 3, 320, 320)]*4
        spec = dict(name='fixture', method='sg_ga', location='center', size=32, eot='none', draws=1, budgets=[16])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            result = optimize(ToyVictim(), images, spec, 0, path)
            self.assertEqual(result['queries'], 16)
            self.assertEqual(len(result['history']), 4)
            patch = torch.from_numpy(np.load(path/'patch_q16.npy'))
            actual = float(objective(ToyVictim().get_raw_predictions(insert(torch.cat(images), patch[None], [[144, 144]]*4))).mean())
            self.assertAlmostEqual(actual, result['checkpoints'][0]['best_training_fitness'], places=5)

    def test_eot_finite_and_gradient_noncontiguous(self):
        for mode in ['full', 'none', 'no_rotation', 'no_color', 'no_blur', 'no_noise']:
            x = torch.rand(1, 3, 16, 16).transpose(2, 3).detach().requires_grad_(True)
            out = make_eot(mode)(x)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(((out >= 0) & (out <= 1)).all())
            out.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
