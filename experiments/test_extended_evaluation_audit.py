"""Mutated fixtures prove that the read-only artifact audit rejects bad data."""
import copy
import unittest
from audit_extended_evaluation import AP_KEYS, FIELDS, audit_case, audit_eot, classify_files, summary


def fixture():
    infos = [dict(id=i, width=100, height=80) for i in [139, 285]]
    configs = [(tau, cap) for tau in [.01, .05, .1, .25, .5] for cap in [None, 50, 100, 300]]
    rows, predictions, metrics = [], {}, []
    for tau, cap in configs:
        found = []
        for info in infos:
            samples = [dict(active_candidates=1, nms_input_candidates=1, final_detections=1,
                            filter_ms=i, nms_ms=i+3, postprocess_ms=2*i+3) for i in [1., 2., 3.]]
            rows.append(dict(image_id=info['id'], threshold=tau, pre_nms_cap=cap, location=None,
                             forward_ms=2., forward_ms_samples=[1., 2., 3.],
                             active_candidates=1, nms_input_candidates=1, final_detections=1,
                             filter_ms=2., nms_ms=5., postprocess_ms=7., postprocess_samples=samples))
            found.append(dict(image_id=info['id'], category_id=1, score=.9, bbox=[10., 10., 20., 20.]))
        predictions[str((tau, cap))] = found
        metric = dict(threshold=tau, pre_nms_cap=cap, coco_subset_bbox={key:.5 for key in AP_KEYS})
        selected = rows[-2:]
        metric.update({field:summary([r[field] for r in selected]) for field in FIELDS})
        metrics.append(metric)
    case = dict(case=dict(name='clean', seed=0, budget=0, spec=None),
                images=2, image_ids=[i['id'] for i in infos], records=rows, predictions=predictions,
                metrics=metrics, total_latency_not_sum_of_medians=True,
                evaluator='pycocotools.COCOeval bbox, maxDets 1/10/100, original crowd/ignore')
    return case, infos


class EvaluationAuditFixtures(unittest.TestCase):
    def test_eot_method_names_are_not_mistaken_for_paired_eot_files(self):
        method = 'eot_no_blur_seed0_q400'
        expected = {method:('eot_no_blur', 0, 400, dict(budgets=[400]))}
        cases, paired, names, unexpected = classify_files({method:'case', 'eot_'+method:'paired'}, expected)
        self.assertEqual(cases, {method:'case'})
        self.assertEqual(paired, {'eot_'+method:'paired'})
        self.assertEqual(names, {'eot_'+method})
        self.assertEqual(unexpected, [])

    def errors(self, record, infos):
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        audit_case(record, ('clean', 0, 0, None), infos, {1}, check)
        return errors

    def test_exact_coverage_and_arithmetic_passes(self):
        case, infos = fixture()
        self.assertEqual(self.errors(case, infos), [])

    def test_missing_and_duplicate_record_rejected(self):
        case, infos = fixture()
        case['records'][-1] = copy.deepcopy(case['records'][0])
        self.assertIn('record coverage/uniqueness mismatch', self.errors(case, infos))

    def test_timing_sample_boundary_and_saved_median_rejected(self):
        case, infos = fixture()
        case['records'][0]['postprocess_samples'][0]['postprocess_ms'] += .2
        case['records'][0]['forward_ms'] += .4
        errors = self.errors(case, infos)
        self.assertIn('sample timing boundary mismatch', errors)
        self.assertIn('forward median mismatch', errors)

    def test_summary_arithmetic_and_candidate_count_rejected(self):
        case, infos = fixture()
        case['metrics'][0]['nms_ms']['mean'] += .5
        case['records'][0]['nms_input_candidates'] = 2
        errors = self.errors(case, infos)
        self.assertIn('summary arithmetic mismatch: nms_ms/mean', errors)
        self.assertIn('candidate/cap/final count limit mismatch', errors)

    def test_wrong_category_box_extent_and_threshold_rejected(self):
        case, infos = fixture()
        prediction = case['predictions']['(0.01, None)'][0]
        prediction.update(category_id=2, bbox=[90., 0., 20., 10.], score=.001)
        errors = self.errors(case, infos)
        self.assertIn('prediction image/category mismatch', errors)
        self.assertIn('COCO box exceeds image extent', errors)
        self.assertIn('prediction confidence outside filter threshold', errors)

    def test_eot_coverage_pair_difference_and_nonfinite_rejected(self):
        infos = [dict(id=i) for i in range(8)]
        record = dict(case='sg_ga', seed=0, queries=400, images=8, independent_draws=16,
                      paired_transform_parameters=True, records=[dict(image_id=i, samples=[
                          dict(draw=d, clean_fitness=1., patch_fitness=3., difference=2.) for d in range(16)]) for i in range(8)])
        errors = []
        def check(condition, message):
            if not condition:
                errors.append(message)
        expected = ('sg_ga', 0, 400, dict(size=64))
        self.assertEqual(audit_eot(record, expected, infos, check), 128)
        self.assertEqual(errors, [])
        record['records'][0]['samples'][0].update(draw=1, difference=float('nan'))
        audit_eot(record, expected, infos, check)
        self.assertIn('EOT draw coverage mismatch', errors)
        self.assertIn('EOT difference mismatch', errors)
        self.assertIn('nonfinite EOT sample', errors)


if __name__ == '__main__':
    unittest.main()
