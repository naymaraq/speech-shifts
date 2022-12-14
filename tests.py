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
from speech_shifts.common.get_loaders import get_train_loader, get_eval_loader
from speech_shifts.common.grouper import CombinatorialGrouper

class MLSRDatasetTest(unittest.TestCase):
    root_dir = "/data/mlsr-data/cv-corpus-wav"
    
    def test_group_loader(self):
        d = MLSRDataset(self.root_dir)
        train = d.get_mixed_train_subset(["train", "rw-train"], loader_kwargs={"n_views": 2})
        
        grouper = CombinatorialGrouper(
            meta_fields=train._metadata_fields, 
            meta_map=train._metadata_map,
            meta_array=train.metadata_array, 
            groupby_fields=['lang']
        )
        n_groups_per_batch = 6
        train_loader  = get_train_loader("group",
                                         train,
                                         batch_size=60,
                                         uniform_over_groups=True,
                                         grouper=grouper,
                                         distinct_groups=True,
                                         n_groups_per_batch=n_groups_per_batch,
                                        )
        audio_signal, audio_lengths, labels, metadata, indices = next(iter(train_loader))
        self.assertEqual(len(set(metadata[:, 0].tolist())), n_groups_per_batch)


    def test_other(self):
        d = MLSRDataset(self.root_dir)
        zh_CN_train = d.get_subset("zh-CN-train", loader_kwargs={"n_views": 1})
        test = d.get_subset("test", {"n_views": 1})

        zh_CN_train_speakers = set(zh_CN_train.y_array)
        test_speakers = set(test.y_array)
        
        for i, split_i in enumerate([zh_CN_train_speakers, test_speakers]):
            for j, split_j in enumerate([zh_CN_train_speakers, test_speakers]):
                if i != j:
                    self.assertEqual(len(split_i & split_j), 0)


    def test_dataloader(self):
        d = MLSRDataset(self.root_dir)
        val = d.get_subset("val", loader_kwargs={"n_views": 1})
        val_loader  = get_eval_loader("standard",
                                      val,
                                      batch_size=20)
        batch = next(iter(val_loader))
        audio_signal, audio_lengths, labels, metadata, indices = batch
        self.assertEqual(audio_signal.shape[0], 20)
        self.assertEqual(audio_lengths.shape[0], 20)
        self.assertEqual(labels.shape[0], 20)
        self.assertEqual(metadata.shape[0], 20)
        self.assertEqual(indices.shape[0], 20)

        train = d.get_subset("train", loader_kwargs={"n_views": 3})
        train_loader  = get_train_loader("standard",
                                         train,
                                         batch_size=64)
        batch = next(iter(train_loader))
        audio_signal, audio_lengths, labels, metadata, indices = batch
        self.assertEqual(audio_signal.shape[0], 3*64)
        self.assertEqual(audio_lengths.shape[0], 3*64)
        self.assertEqual(labels.shape[0], 3*64)
        self.assertEqual(metadata.shape[0], 3*64)
        self.assertEqual(indices.shape[0], 3*64)


    def test_mlsr(self):
        d = MLSRDataset(self.root_dir)
        val = d.get_subset("val", {"n_views": 1})
        id_val = d.get_subset("id_val", {"n_views": 1})
        test = d.get_subset("test", {"n_views": 1})
        train = d.get_subset("train", {"n_views": 1})

        v = set([i[2] for i in val.input_trial_array] + [i[1] for i in val.input_trial_array])
        v1 = set([val.dataset._input_array[ix] for ix in val.indices])

        i = set([i[2] for i in id_val.input_trial_array] + [i[1] for i in id_val.input_trial_array])
        i1 = set([id_val.dataset._input_array[ix] for ix in id_val.indices])
        
        t = set([i[2] for i in test.input_trial_array] + [i[1] for i in test.input_trial_array])
        t1 = set([test.dataset._input_array[ix] for ix in test.indices])
        
        self.assertEqual(v, v1)
        self.assertEqual(i, i1)
        self.assertEqual(t, t1)

        print("Number of files in val {}".format(len(v)))
        print("Number of files in id_val {}".format(len(i)))
        print("Number of files in test {}".format(len(t)))

        train_speakers = set(train.y_array)
        val_speakers = set(val.y_array)
        id_val_speakers = set(id_val.y_array)
        test_speakers = set(test.y_array)

        for i, split_i in enumerate([train_speakers, val_speakers, id_val_speakers, test_speakers]):
            for j, split_j in enumerate([train_speakers, val_speakers, id_val_speakers, test_speakers]):
                if i != j:
                    self.assertEqual(len(split_i & split_j), 0)

        # crush test
        for subset in [id_val, test, val]:
            y_true = subset.trial_y_array
            metadata = subset.trial_metadata_array
            y_pred = torch.rand(size=y_true.shape)

            _,s1, _, s2 = subset.eval(
                y_pred, 
                y_true, 
                metadata
            )
            
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
