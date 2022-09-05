'''
use this methods for assertion
self.assertEqual(a, b)          a == b
self.assertTrue(x)              bool(x) is True
self.assertFalse(x)             bool(x) is False
self.assertIs(a, b)             a is b
self.assertIsNone(x)            x is None
self.assertIn(a, b)             a in b
self.assertIsInstance(a, b)     isinstance(a, b)
'''

import unittest
import torch
from speech_shifts.datasets.mlsr_dataset import MLSRDataset
from speech_shifts.common.metrics.sv_metrics import EqualErrorRate, DCF

class MLSRDatasetTest(unittest.TestCase):
    root_dir = "/home/tsargsyan/davit/voxsrc22/dg-sr/cv-corpus-wav"
    
    def test_speaker_intersections(self):
        d = MLSRDataset(self.root_dir)
        train = d.get_subset("train")
        val = d.get_subset("val")
        id_val = d.get_subset("id_val")
        test = d.get_subset("test")

        train_speakers = set(train.y_array)
        val_speakers = set(val.y_array)
        id_val_speakers = set(id_val.y_array)
        test_speakers = set(test.y_array)

        for i, split_i in enumerate([train_speakers, val_speakers, id_val_speakers, test_speakers]):
            for j, split_j in enumerate([train_speakers, val_speakers, id_val_speakers, test_speakers]):
                if i != j:
                    self.assertEqual(len(split_i & split_j), 0)
        
class MetricTest(unittest.TestCase):
    def test_eer(self):
        metric = EqualErrorRate()
        scores = torch.tensor([0.567, 0.578, 0.660])
        labels = torch.tensor([1., 1., 0.])
        eer = metric.compute(scores, labels, return_dict=False)
        self.assertEqual(eer, 1.0)
    
    def test_compute_group_wise_eer(self):
        metric = EqualErrorRate()
        scores = torch.tensor([0.567, 0.578, 0.660, 0.4, 0.4])
        labels = torch.tensor([1., 1., 0., 0., 0.])
        g = torch.tensor([0, 0, 0, 1, 1])
        result = metric.compute_group_wise(scores, labels, g, 2, return_dict=True)
        
        self.assertEqual(result["EER_group:0"], 1.0)
        self.assertEqual(result["EER_wg"], 1.0)
        self.assertEqual(result["count_group:0"], 3)
        self.assertEqual(result["count_group:1"], 2)


    def test_dcf(self):
        metric = DCF()
        scores = torch.tensor([0.567, 0.578, 0.660])
        labels = torch.tensor([1., 1., 0.])
        dcf = metric.compute(scores, labels, return_dict=False)
        self.assertEqual(dcf, 1.0)

if __name__ == '__main__':
    unittest.main()