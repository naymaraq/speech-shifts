import pandas as pd
import os
import torch

from speech_shifts.datasets.speech_shifts_dataset import SpeechShiftsDataset
from speech_shifts.common.grouper import CombinatorialGrouper
from speech_shifts.common.waveform_featurizer import WaveformFeaturizer

EPSILION = 1e-15


class MLSRDataset(SpeechShiftsDataset):
    """Multilingual speaker recognition dataset"""
    
    _dataset_name = 'MLSR'
    def __init__(self, root_dir):
        
        metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        self._input_array = [os.path.join(root_dir, p) for p in metadata["audio_filepath"].tolist()]
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

        
        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata['lang'].values),
             self._y_array),
            dim=1
        )

        self._metadata_fields = ['lang', 'y']
        self._metadata_map = {
            "lang": langs
        }

        # init _eval_grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['lang']
        )
        
        # init _featurizer
        # defualt _featurizer without augmentor
        self._featurizer = WaveformFeaturizer(sample_rate=16000,
                                              int_values=False,
                                              augmentor=None)
        
        self._collate = _fixed_seq_collate_fn
        super().__init__(root_dir)
    
    def set_waveform_featurizer(self, featurizer):
        self._featurizer = featurizer

    def get_input(self, idx, norm_energy=True, norm_value=0.061):
        audio_filepath = self._input_array[idx]
        duration = self._duration_array[idx]

        features = self._featurizer.process(audio_filepath, offset=0, duration=duration, trim=False)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        
        if norm_energy:
            features *= norm_value / (torch.sqrt(torch.mean(features ** 2)) + EPSILION)
        f, fl = features, torch.tensor(features.shape[0]).long()
        return f, fl
            

def _fixed_seq_collate_fn(batch):
    """collate batch of audio sig, audio len, label, metadata"""
    sig_and_length, _, _ = zip(*batch)
    audio_lengths = [length for _, length in sig_and_length]
    fixed_length = int(max(audio_lengths))

    audio_signal, labels, new_audio_lengths, metadata = [], [], [], []
    for (sig, sig_len), labels_i, meta in batch:
        sig_len = sig_len.item()
        if sig_len < fixed_length:
            repeat = fixed_length // sig_len
            rem = fixed_length % sig_len
            sub = sig[-rem:] if rem > 0 else torch.tensor([])
            rep_sig = torch.cat(repeat * [sig])
            sig = torch.cat((rep_sig, sub))
        
        new_audio_lengths.append(torch.tensor(fixed_length))    
        audio_signal.append(sig)
        labels.append(labels_i)
        metadata.append(meta)

    audio_signal = torch.stack(audio_signal)
    audio_lengths = torch.stack(new_audio_lengths)
    labels = torch.stack(labels)
    metadata = torch.stack(metadata)
    return audio_signal, audio_lengths, labels, metadata

