"""Small scheduling fixtures, not model/latency research results."""
import unittest
from pathlib import Path
from extended_stream_qos import Admission, execute_stream, checkpoint_path


class StreamFixtures(unittest.TestCase):
    def test_patch_paths_match_real_optimizer_case_names(self):
        root = Path('fixtures')
        self.assertEqual(checkpoint_path(root,'sg_ga_q2500',3),root/'optimization/sg_ga_seed3/patch_q2500.npy')
        self.assertEqual(checkpoint_path(root,'tanh_whitebox_patch_q400',3),root/'optimization/tanh_whitebox_patch_seed3/patch_q400.npy')
        self.assertIsNone(checkpoint_path(root,'clean',3))
        self.assertRaises(ValueError,checkpoint_path,root,'gradient_tanh_q400',3)

    def test_ewma_is_causal_and_bounded(self):
        controller = Admission(100)
        self.assertEqual(controller.interval_ms,100)
        controller.observe(200)
        self.assertEqual(controller.interval_ms,240)
        controller.observe(100)
        self.assertEqual(controller.ewma_ms,175)
        self.assertEqual(controller.interval_ms,210)
        controller.observe(10000)
        self.assertEqual(controller.interval_ms,600)

    def test_admission_uses_busy_and_past_feedback(self):
        controller = Admission(100)
        self.assertEqual(controller.admit(0,False),(True,None))
        self.assertEqual(controller.admit(100,True),(False,'busy'))
        controller.observe(200)
        self.assertEqual(controller.admit(200,False),(False,'adaptive_interval'))
        self.assertEqual(controller.admit(240,False),(True,None))

    def test_invalid_measurements_rejected(self):
        controller = Admission(100)
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                controller.observe(value)
        with self.assertRaises(ValueError):
            Admission(700)

    def test_actual_fifo_preserves_all_frames_and_measured_timestamps(self):
        result = execute_stream([1,2,3],lambda frame: {'value':frame},1,'fifo',30000)
        self.assertTrue(result['actual_scheduled_execution'])
        self.assertFalse(result['actual_camera_measurement'])
        self.assertEqual(result['processed_frames'],3)
        self.assertEqual(result['discarded_frames'],0)
        self.assertEqual([r['result']['value'] for r in result['records']],[1,2,3])
        for row in result['records']:
            self.assertGreaterEqual(row['finish_ms'],row['start_ms'])
            self.assertGreaterEqual(row['start_ms'],row['actual_arrival_ms'])
            self.assertNotIn('arrival_ns',row)
        self.assertGreaterEqual(result['throughput_denominator_ms'],3)

    def test_worker_failure_propagates_not_fabricated_as_result(self):
        def fail(frame):
            raise ValueError('fixture failure')
        with self.assertRaisesRegex(RuntimeError,'Inference worker failed'):
            execute_stream([1,2],fail,1,'fifo')

    def test_policy_validation(self):
        for policy in ['queue_simulation','unknown']:
            with self.assertRaises(ValueError):
                execute_stream([1],lambda x:x,100,policy)
        with self.assertRaises(ValueError):
            execute_stream([],lambda x:x,100,'fifo')


if __name__ == '__main__':
    unittest.main()
