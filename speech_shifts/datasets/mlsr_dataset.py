import pandas as pd
import os
import torch

from speech_shifts.datasets.speech_shifts_dataset import SpeechShiftsDataset
from speech_shifts.common.grouper import CombinatorialGrouper

class MLSRDataset(SpeechShiftsDataset):
    """Multilingual speaker recognition dataset"""
    
    _dataset_name = 'MLSR'
    def __init__(self, root_dir):
        
        metadata = pd.read_csv(os.path.join(root_dir, "metadata.csv"))
        self._input_array = [os.path.join(root_dir, p) for p in metadata["audio_filepath"].tolist()]

        self._y_array = metadata["speaker"].tolist()
        sp2id = {speaker: index for index, speaker in enumerate(set(self._y_array))}
        self._y_array = torch.LongTensor([sp2id[sp] for sp in self._y_array])
        
        self._y_size = 1
        self._n_classes = None

        metadata.loc[(metadata["split"] == "train"), "split"] = self.split_dict["train"]
        metadata.loc[(metadata["split"] == "val"), "split"] = self.split_dict["val"]
        metadata.loc[(metadata["split"] == "test"), "split"] = self.split_dict["test"]

        self._split_array = metadata['split'].values
        
        langs = list(set(metadata["lang"].tolist()))
        langs = sorted(langs)
        lang2id = {lang: i for i, lang in enumerate(langs)}
        metadata["lang"] = metadata["lang"].apply(lambda x: lang2id[x])

        #torch.LongTensor(metadata["duration"].values*1000),
        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata['lang'].values),
             self._y_array),
            dim=1
        )

        self._metadata_fields = ['lang', 'y']
        self._metadata_map = {
            "lang": langs
        }

        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=['lang']
        )

        super().__init__(root_dir)

    
    def get_input(self, idx):
        pass


if __name__ == "__main__":

    d = MLSRDataset("/home/tsargsyan/davit/voxsrc22/dg-sr/cv-corpus-wav")