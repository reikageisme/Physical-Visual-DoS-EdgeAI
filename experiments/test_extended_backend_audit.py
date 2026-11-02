"""Mutated fixtures for timing scope, CPU normalization and offline recurrence."""
import copy
import unittest
from audit_extended_backend import CONDITIONS, audit_timing, audit_queue


def timing_fixture():
    ids = [139, 285]
    rows = []
    for name in CONDITIONS:
        for image_id in ids:
            for repeat in range(3):
                rows.append(dict(condition=name, image_id=image_id, repeat=repeat,
                    location=None if name=='clean' else [128,128],
                    forward_ms=10., decode_ms=1., filter_ms=2., nms_ms=3., postprocess_ms=5.,
                    total_prepared_tensor_ms=16.1, late_processing_budget=False,
                    active_candidates=100, nms_input_candidates=100, final_detections=50,
                    condition_block_cpu_percent_one_core_scale=144.,
                    condition_block_cpu_percent_machine_normalized=12., rss_mib_at_condition_end=512.))
    return dict(seed=0, cpu_threads=2, records=rows, temperature_available=False,
                temperature_not_substituted_with_zero=True), ids


def queue_fixture():
    # Three offers at 0/100/200 ms; FIFO processing lasts 150/50/25 ms.
    rows = [dict(frame=0,arrival_ms=0,start_ms=0.,finish_ms=150.,modeled_age_ms=150.,modeled_wait_ms=0.,late=True),
            dict(frame=1,arrival_ms=100,start_ms=150.,finish_ms=200.,modeled_age_ms=100.,modeled_wait_ms=50.,late=False),
            dict(frame=2,arrival_ms=200,start_ms=200.,finish_ms=225.,modeled_age_ms=25.,modeled_wait_ms=0.,late=False)]
    return dict(policy='fifo',arrival_period_ms=100,generated=3,processed=3,dropped=0,
                modeled_late_fraction=1/3,max_modeled_age_ms=150.,max_modeled_wait_ms=50.,
                records=rows,actual_camera_measurement=False,queue_model_assumes_deterministic_arrivals=True)


class BackendAuditFixtures(unittest.TestCase):
    def errors_for_timing(self, block, ids):
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        audit_timing(block,0,2,ids,check)
        return errors

    def errors_for_queue(self, record, policy='fifo'):
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        audit_queue(record,[150.,50.,25.],100,policy,check)
        return errors

    def test_full_timing_coverage_and_resource_boundary_passes(self):
        block, ids = timing_fixture()
        self.assertEqual(self.errors_for_timing(block,ids), [])

    def test_duplicate_missing_repeats_and_fake_cpu_normalization_rejected(self):
        block, ids = timing_fixture()
        block['records'][-1] = copy.deepcopy(block['records'][0])
        block['records'][0]['condition_block_cpu_percent_machine_normalized'] = 144.
        errors = self.errors_for_timing(block,ids)
        self.assertIn('timing coverage/uniqueness mismatch',errors)
        self.assertIn('CPU normalization mismatch',errors)

    def test_wrong_whole_clock_and_deadline_flag_rejected(self):
        block, ids = timing_fixture()
        block['records'][0]['total_prepared_tensor_ms'] = 15.
        block['records'][0]['late_processing_budget'] = True
        errors = self.errors_for_timing(block,ids)
        self.assertIn('whole-tensor duration smaller than timed components',errors)
        self.assertIn('300ms budget flag mismatch',errors)

    def test_fifo_wait_and_per_period_lateness_not_300ms_passes(self):
        self.assertEqual(self.errors_for_queue(queue_fixture()),[])

    def test_mutated_queue_wait_and_conservation_rejected(self):
        record = queue_fixture()
        record['records'][1]['modeled_wait_ms'] = 0.
        record['dropped'] = 1
        errors = self.errors_for_queue(record)
        self.assertIn('queue recurrence mismatch: modeled_wait_ms',errors)
        self.assertIn('queue frame conservation mismatch',errors)

    def test_busy_drop_does_not_become_actual_camera_experiment(self):
        record = queue_fixture()
        record.update(policy='drop_if_busy',processed=2,dropped=1,modeled_late_fraction=.5,
                      max_modeled_wait_ms=0.,records=[record['records'][0],record['records'][2]])
        self.assertEqual(self.errors_for_queue(record,'drop_if_busy'),[])
        record['actual_camera_measurement'] = True
        self.assertIn('offline queue scope changed',self.errors_for_queue(record,'drop_if_busy'))


if __name__ == '__main__':
    unittest.main()
