import pandas as pd
import numpy as np
import os
import torch
import warnings

from speech_shifts.datasets.speech_shifts_dataset import (SpeechShiftsDataset, 
                                                          SpeechShiftsSubsetWithTrials,
                                                          SpeechShiftsSubset,
                                                          _fixed_seq_collate_fn)

from speech_shifts.common.grouper import CombinatorialGrouper
from speech_shifts.common.audio.waveform_featurizer import WaveformFeaturizer
from speech_shifts.common.metrics.sv_metrics import EqualErrorRate, DCF
EPSILION = 1e-15


class MLSRDataset(SpeechShiftsDataset):
    """Multilingual speaker recognition dataset"""
    
    _dataset_name = 'MLSR'

    def read_trials(self, root_dir):
        trials = []
        with open(os.path.join(root_dir, "trials.txt")) as f:
            for line in f.readlines():
                line = line.strip()
                y, file1, file2, lang, split = line.split(" ")
                y = int(y)
                trials.append((y, file1, file2, lang, split))
        
        trial_df = pd.DataFrame(
            data=trials, 
            columns=["y", "file1", "file2", "lang", "split"]
        )
        trial_df["file1"] = trial_df["file1"].apply(lambda p: os.path.join(root_dir, p))
        trial_df["file2"] = trial_df["file2"].apply(lambda p: os.path.join(root_dir, p))
        return trial_df
    
    def get_subset(self, 
                   split, 
                   loader_kwargs, 
                   frac=1.0, 
                   min_dur=None, 
                   max_dur=None,
                   augmentor=None):
        
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")
        
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        if split != "train":
            if frac < 1.0:
                warnings.warn("Only for train split frac can be lower than 1.0")
            trial_split_mask = self._trial_split_array == self.split_dict[split]
            trial_split_idx = np.where(trial_split_mask)[0]
            subset_dataset = SpeechShiftsSubsetWithTrials(self, split_idx, trial_split_idx, loader_kwargs, augmentor)
        else:
            if frac < 1.0:
                # Randomly sample a fraction of the split
                num_to_retain = int(np.round(float(len(split_idx)) * frac))
                split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])
                
            if min_dur:
                min_duration_mask = self._duration_array >= min_dur
                min_dur_idx = np.where(min_duration_mask)[0]
                split_idx = np.intersect1d(split_idx, min_dur_idx)
        
            if max_dur:
                max_duration_mask = self._duration_array <= max_dur
                max_dur_idx = np.where(max_duration_mask)[0]
                split_idx = np.intersect1d(split_idx, max_dur_idx)
            
            subset_dataset = SpeechShiftsSubset(self, split_idx, loader_kwargs, augmentor)

        return subset_dataset
    
    def __init__(self, root_dir):
        
        # read metadata
        metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        trial_df = self.read_trials(root_dir)
        
        self._input_array = [os.path.join(root_dir, p) for p in metadata["audio_filepath"].tolist()]
        self._index2path = dict(enumerate(self._input_array))

        self._input_trial_array = list(zip(trial_df["y"].tolist(), trial_df["file1"].tolist(), trial_df["file2"].tolist()))
        self._duration_array = metadata["duration"].values

        # init _split_dict, _split_names, _split_array
        self._split_dict = {
            'train': 0,
            'id_val': 1,
            'test': 2,
            'val': 3
        }
        self._split_names = {
            'train': 'Train',
            'id_val': 'Validation (ID)',
            'test': 'Test',
            'val': 'Validation (OOD)',
        }
        for split in self.split_dict:
            metadata.loc[(metadata["split"] == split), "split"] = self.split_dict[split]
            trial_df.loc[(trial_df["split"] == split), "split"] = self.split_dict[split]
        self._split_array = metadata['split'].values

        # init _y_array, _y_size, _n_classes
        self._y_array = metadata["speaker"].tolist()
        sp2id = {speaker: index for index, speaker in enumerate(set(self._y_array))}
        self._y_array = torch.LongTensor([sp2id[sp] for sp in self._y_array])
        self._y_size = 1
        self._n_classes = None

        
        # init _metadata_array, _metadata_fields, _metadata_map
        langs = list(set(metadata["lang"].tolist()))
        langs = sorted(langs)
        lang2id = {lang: i for i, lang in enumerate(langs)}
        metadata["lang"] = metadata["lang"].apply(lambda x: lang2id[x])
        trial_df["lang"] = trial_df["lang"].apply(lambda x: lang2id[x])

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata['lang'].values),
             self._y_array),
            dim=1
        )
        self._metadata_fields = ['lang', 'y']
        self._metadata_map = {"lang": langs}

        
        # init _featurizer
        # defualt _featurizer without augmentor
        self._featurizer = WaveformFeaturizer(sample_rate=16000,
                                              int_values=False)
        
        self._collate = _fixed_seq_collate_fn
            
        self._trial_split_array = trial_df['split'].values
        self._trial_y_array = torch.LongTensor(trial_df["y"].values)
        self._trial_y_size = 1
        self._trial_n_classes = 2
        self._trial_metadata_array = torch.stack(
            (torch.LongTensor(trial_df['lang'].values),
             self._trial_y_array),
            dim=1
        )
        self._trial_metadata_fields = ['lang', 'y']
        self._trial_metadata_map = {"lang": langs}

        self._trial_eval_grouper = CombinatorialGrouper(
            meta_fields=self._trial_metadata_fields, 
            meta_map=self._trial_metadata_map,
            meta_array=self._trial_metadata_array, 
            groupby_fields=['lang']
        )
        super().__init__(root_dir)
    
    
    def get_input(self, idx, norm_energy=True, norm_value=0.061, augmentor=None):
        audio_filepath = self._input_array[idx]
        duration = self._duration_array[idx]

        features = self._featurizer.process(audio_filepath, 
                        offset=0, 
                        duration=duration, 
                        trim=False,
                        augmentor=augmentor
        )
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        
        if norm_energy:
            features *= norm_value / (torch.sqrt(torch.mean(features ** 2)) + EPSILION)
        f, fl = features, torch.tensor(features.shape[0]).long()
        return f, fl, idx
    
    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        eer_metric = EqualErrorRate(prediction_fn=prediction_fn)
        min_dcf_metric = DCF(prediction_fn=prediction_fn)
        
        eer_results, eer_results_str = self.standard_group_eval(
            eer_metric,
            self._trial_eval_grouper,
            y_pred, y_true, metadata
        )
        dcf_results, dcf_results_str = self.standard_group_eval(
            min_dcf_metric,
            self._trial_eval_grouper,
            y_pred, y_true, metadata
        )
        return eer_results, eer_results_str, dcf_results, dcf_results_str


