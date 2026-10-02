"""Offline fixtures for category/geometry/COCO/NMS measurement semantics."""
import contextlib
import io
import unittest

from extended_evaluation import COCO, VictimModel, coco_metrics, official_predictions, postprocess
import torch


def ground_truth():
    coco = COCO()
    coco.dataset = dict(images=[dict(id=7, width=640, height=160)],
                        categories=[dict(id=13, name='fixture')],
                        annotations=[dict(id=1, image_id=7, category_id=13,
                                          bbox=[0, 0, 20, 5], area=100, iscrowd=0)])
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


class EvaluationFixtures(unittest.TestCase):
    def test_exact_and_empty_official_coco(self):
        coco = ground_truth()
        pred = [dict(image_id=7, category_id=13, bbox=[0, 0, 20, 5], score=.9)]
        self.assertAlmostEqual(coco_metrics(coco, pred, [7])['AP'], 1)
        self.assertEqual(coco_metrics(coco, [], [7])['AP'], 0)

    def test_wrong_category(self):
        coco = ground_truth()
        coco.dataset['categories'].append(dict(id=18, name='wrong'))
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        pred = [dict(image_id=7, category_id=18, bbox=[0, 0, 20, 5], score=.9)]
        self.assertEqual(coco_metrics(coco, pred, [7])['AP'], 0)

    def test_inverse_resize_and_category_mapping(self):
        found = torch.tensor([[-3., -4., 10., 10., .9, 0.]])
        pred = official_predictions(found, dict(id=7, width=640, height=160), [13])
        self.assertEqual(pred[0]['category_id'], 13)
        self.assertEqual(pred[0]['bbox'], [0., 0., 20., 5.])

    def test_crowd_ignored_not_false_positive(self):
        coco = ground_truth()
        coco.dataset['annotations'].append(dict(id=2, image_id=7, category_id=13,
                                                bbox=[100, 20, 40, 40], area=1600, iscrowd=1))
        with contextlib.redirect_stdout(io.StringIO()):
            coco.createIndex()
        pred = [dict(image_id=7, category_id=13, bbox=[100, 20, 40, 40], score=.99),
                dict(image_id=7, category_id=13, bbox=[0, 0, 20, 5], score=.9)]
        self.assertAlmostEqual(coco_metrics(coco, pred, [7])['AP'], 1)

    def test_pre_cap_and_backend_equivalence_distinct_scores(self):
        decoded = torch.tensor([[0,0,10,10,.9,0], [0,0,10,10,.8,1],
                                [20,20,30,30,.7,0], [40,40,50,50,.005,0]], dtype=torch.float)
        a, keep_a = postprocess(VictimModel, decoded, backend='custom')
        b, keep_b = postprocess(VictimModel, decoded, backend='torchvision')
        self.assertTrue(torch.equal(keep_a, keep_b))
        self.assertEqual(a['active_candidates'], 3)
        self.assertEqual(b['final_detections'], 2)
        capped, keep = postprocess(VictimModel, decoded, cap=1)
        self.assertEqual(capped['active_candidates'], 3)
        self.assertEqual(capped['nms_input_candidates'], 1)
        self.assertEqual(len(keep), 1)


if __name__ == '__main__':
    unittest.main()
