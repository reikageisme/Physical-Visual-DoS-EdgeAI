"""Fixtures for controls, paired inference, queue semantics and transfer mapping."""
import unittest
import numpy as np
import torch

from extended_analysis import paired,holm
from extended_backend_qos import queue_model
from extended_controls import pattern
from extended_transfer import predictions,predict
import torchvision


class AdditionalFixtures(unittest.TestCase):
    def test_controls_bounded_and_reproducible(self):
        names = ['uniform_noise','gaussian_noise','solid_gray','checkerboard','binary_modules']
        for name in names:
            a = pattern(name,3)
            self.assertEqual(tuple(a.shape),(3,64,64))
            self.assertTrue(torch.equal(a,pattern(name,3)))
            self.assertTrue(torch.isfinite(a).all())
            self.assertGreaterEqual(float(a.min()),0)
            self.assertLessEqual(float(a.max()),1)
        for name in ['checkerboard','binary_modules']:
            self.assertEqual(set(pattern(name,3).unique().tolist()),{0.,1.})

    def test_queue_is_model_not_capture(self):
        fifo = queue_model([150,150,150],100)
        drop = queue_model([150,150,150],100,'drop_if_busy')
        self.assertEqual([r['modeled_age_ms'] for r in fifo['records']],[150,200,250])
        self.assertEqual(fifo['max_modeled_wait_ms'],100)
        self.assertEqual(drop['dropped'],1)
        self.assertEqual(drop['processed'],2)
        self.assertFalse(drop['actual_camera_measurement'])

    def test_exact_sign_flip_and_zero_effect(self):
        positive = paired(np.ones(10))
        zero = paired(np.zeros(10))
        self.assertEqual(positive['exact_two_sided_sign_flip_p'],2/1024)
        self.assertEqual(zero['exact_two_sided_sign_flip_p'],1)
        self.assertEqual(zero['conditional_paired_seed_bootstrap_95ci'],[0.,0.])
        self.assertRaises(ValueError,paired,[1.,2.])

    def test_holm_family(self):
        rows = [dict(statistics=dict(exact_two_sided_sign_flip_p=p)) for p in [.1,.001,.02]]
        adjusted = holm(rows)
        self.assertAlmostEqual(adjusted[0]['holm_p_within_declared_secondary_family'],.1)
        self.assertAlmostEqual(adjusted[1]['holm_p_within_declared_secondary_family'],.003)
        self.assertAlmostEqual(adjusted[2]['holm_p_within_declared_secondary_family'],.04)

    def test_category_mapping_matches_native_ids(self):
        boxes = torch.tensor([[0.,0.,10.,10.]])
        info = dict(id=7,width=640,height=160)
        yolo = predictions((boxes,torch.tensor([.9]),torch.tensor([0])),info,'yolov8s',[13])
        native = predictions((boxes,torch.tensor([.9]),torch.tensor([13])),info,'ssdlite',[13])
        self.assertEqual(yolo,native)
        self.assertEqual(native[0]['bbox'],[0.,0.,20.,5.])

    def test_native_nms_call_instrumented_and_restored(self):
        class Fixture(torch.nn.Module):
            def forward(self, images):
                boxes = torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.]])
                scores,classes = torch.tensor([.9,.8]),torch.tensor([1,1])
                keep = torchvision.ops.boxes.batched_nms(boxes,scores,classes,.45)
                return [dict(boxes=boxes[keep],scores=scores[keep],labels=classes[keep])]
        original = torchvision.ops.boxes.batched_nms
        stats, found = predict('ssdlite',Fixture(),torch.zeros(1,3,320,320))
        self.assertEqual(stats['nms_calls'][0]['input_candidates'],2)
        self.assertEqual(stats['final_detections'],1)
        self.assertIs(torchvision.ops.boxes.batched_nms,original)
        self.assertEqual(len(found[0]),1)


if __name__ == '__main__':
    unittest.main()
