from telnetlib import X3PAD
import torch
import torch.nn as nn

from examples.models.jasper_block.utils import init_weights
from examples.models.conv_asr_featurizer import SpeakerDecoder
from examples.models.tdnn.tdnn_module import TDNNModule, TDNNSEModule
from examples.models.base_featurizer import BaseFeaturizer

class ECAPAEncoder(BaseFeaturizer):
    """
    Modified ECAPA Encoder layer without Res2Net module for faster training and inference which achieves
    better numbers on speaker diarization tasks
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    input:
        feat_in: input feature shape (mel spec feature shape)
        filters: list of filter shapes for SE_TDNN modules
        kernel_sizes: list of kernel shapes for SE_TDNN modules
        dilations: list of dilations for group conv se layer
        scale: scale value to group wider conv channels (deafult:8)
    output:
        outputs : encoded output
        output_length: masked output lengths
    """

    def __init__(
        self,
        feat_in: int,
        filters: list,
        decoder: dict,
        kernel_sizes: list,
        dilations: list,
        scale: int = 8,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()

        self._feat_in = feat_in
        self.layers = nn.ModuleList()
        self.layers.append(TDNNModule(feat_in, filters[0], kernel_size=kernel_sizes[0], dilation=dilations[0]))

        for i in range(len(filters) - 2):
            self.layers.append(
                TDNNSEModule(
                    filters[i],
                    filters[i + 1],
                    group_scale=scale,
                    se_channels=128,
                    kernel_size=kernel_sizes[i + 1],
                    dilation=dilations[i + 1],
                )
            )
        self.feature_agg = TDNNModule(filters[-1], filters[-1], kernel_sizes[-1], dilations[-1])
        self.apply(lambda x: init_weights(x, mode=init_mode))

        self.decoder = SpeakerDecoder(**decoder)

    def emb_dim(self):
        return self.decoder.emb_dim
    
    def get_feat_in(self):
        return self._feat_in
    
    def features(self, audio_signal, length=None):
        x = audio_signal
        outputs = []

        for layer in self.layers:
            x = layer(x, length=length)
            outputs.append(x)

        x = torch.cat(outputs[1:], dim=1)
        x = self.feature_agg(x)
        pool, emb = self.decoder(x, length)
        return pool, emb
