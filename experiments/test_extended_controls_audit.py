"""Small mutated fixtures for control query/checkpoint and geometry auditing."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from audit_extended_controls import audit_integrity
from audit_extended_evaluation import audit_case
from test_extended_evaluation_audit import fixture


def gradient_fixture():
    history = [dict(step=i, image_forwards=i*4, smooth_confidence_sum=i*.1,
                    sponge_fitness=i*.3) for i in range(1, 101)]
    record = dict(seed=2, queries=400, backward_calls=100, backward_image_examples=400,
                  history=history, locations=[[0, 0], [128, 128], [256, 256], [0, 256]],
                  final_iterate_not_stochastic_best=True, not_exact_published_reproduction=True)
    measured = np.full((3, 64, 64), .25, dtype=np.float32)
    state = dict(seed=2, step=100, history=copy.deepcopy(history), evaluated=torch.from_numpy(measured.copy()))
    return record, measured, state


class ControlAuditFixtures(unittest.TestCase):
    def audit(self, record, measured, state):
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        with patch('torch.load', return_value=state):
            audit_integrity(record, 2, measured, Path('fixture-unused-folder'), check)
        return errors

    def test_complete_exact_query_and_checkpoint_history_passes(self):
        self.assertEqual(self.audit(*gradient_fixture()), [])

    def test_query_history_and_backward_count_mutations_rejected(self):
        record, measured, state = gradient_fixture()
        record['history'][0]['image_forwards'] = 8
        record['backward_calls'] = 99
        errors = self.audit(record, measured, state)
        self.assertIn('actual image-forward history mismatch', errors)
        self.assertIn('backward count mismatch', errors)
        self.assertIn('integrity checkpoint history mismatch', errors)

    def test_unevaluated_next_patch_rejected(self):
        record, measured, state = gradient_fixture()
        self.assertIn('saved patch is not last measured candidate', self.audit(record, measured+.001, state))

    def test_control_case_uses_geometry_from_frozen_protocol(self):
        record, infos = fixture()
        configs = {(0.01, None), (0.25, None), (0.01, 100)}
        record['case'] = dict(name='gaussian_noise', seed=0, budget=0, spec=dict(location='saliency', budgets=[0]))
        record['records'] = [r for r in record['records'] if (r['threshold'], r['pre_nms_cap']) in configs]
        record['metrics'] = [r for r in record['metrics'] if (r['threshold'], r['pre_nms_cap']) in configs]
        record['predictions'] = {k:v for k, v in record['predictions'].items() if k in {str(c) for c in configs}}
        for row in record['records']:
            row['location'] = [256, 256]
        expected = ('gaussian_noise', 0, 0, dict(location='saliency', budgets=[0]))
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        audit_case(record, expected, infos, {1}, check, patch_size=64)
        self.assertEqual(errors, [])
        record['records'][0]['location'] = [257, 256]
        audit_case(record, expected, infos, {1}, check, patch_size=64)
        self.assertIn('invalid patch placement', errors)


if __name__ == '__main__':
    unittest.main()
