import torch
import torch.nn as nn

from examples.models.titanet.utils import init_weights, get_activation
from examples.models.titanet.jasper.jasper import JasperBlock
from examples.models.titanet.pooling_layers import StatsPoolLayer, AttentivePoolLayer
from examples.models.base_featurizer import BaseFeaturizer


class ConvASREncoder(BaseFeaturizer):
    def __init__(
        self,
        jasper,
        decoder,
        activation: str,
        feat_in: int,
        normalization_mode: str = "batch",
        residual_mode: str = "add",
        norm_groups: int = -1,
        conv_mask: bool = True,
        frame_splicing: int = 1,
        init_mode: str = 'xavier_uniform'
    ):
        super().__init__()
        activation = get_activation(activation)
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        feat_in = feat_in * frame_splicing

        self._feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_size', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            future_context = lcfg.get('future_context', -1)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation(),
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                    future_context=future_context,
                )
            )
            feat_in = lcfg['filters']

        self._feat_out = feat_in

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))
        self.max_audio_length = 0

        self.decoder = SpeakerDecoder(**decoder)

    def get_feat_in(self):
        return self._feat_in

    def emb_dim(self):
        return self.decoder.emb_dim

    def features(self, audio_signal, length):
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]
        
        encoder_output = s_input[-1]
        pool, emb = self.decoder(encoder_output, length)
        return pool, emb

    def forward(self, audio_signal, length):
        pool, emb = self.features(audio_signal, length)
        return pool, emb


class SpeakerDecoder(nn.Module):
    def __init__(
        self,
        feat_in: int,
        emb_sizes: int = 256,
        pool_mode: str = 'xvector',
        attention_channels: int = 128,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self.emb_id = 2
        emb_sizes = [emb_sizes] if type(emb_sizes) is int else emb_sizes

        self.pool_mode = pool_mode.lower()
        if self.pool_mode == 'xvector' or self.pool_mode == 'tap':
            self._pooling = StatsPoolLayer(feat_in=feat_in, pool_mode=self.pool_mode)
            affine_type = 'linear'
        elif self.pool_mode == 'attention':
            self._pooling = AttentivePoolLayer(inp_filters=feat_in, attention_channels=attention_channels)
            affine_type = 'conv'

        shapes = [self._pooling.feat_in]
        for size in emb_sizes:
            shapes.append(int(size))

        emb_layers = []
        i = 0 
        for shape_in, shape_out in zip(shapes[:-1], shapes[1:]):
            layer = self.affine_layer(shape_in, shape_out, learn_mean=False, affine_type=affine_type)
            emb_layers.append(layer)
            if i<=self.emb_id-1:
                self.emb_dim = shape_out

        self.emb_layers = nn.ModuleList(emb_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def affine_layer(
        self, inp_shape, out_shape, learn_mean=True, affine_type='conv',
    ):
        if affine_type == 'conv':
            layer = nn.Sequential(
                nn.BatchNorm1d(inp_shape, affine=True, track_running_stats=True),
                nn.Conv1d(inp_shape, out_shape, kernel_size=1),
            )
        else:
            layer = nn.Sequential(
                nn.Linear(inp_shape, out_shape),
                nn.BatchNorm1d(out_shape, affine=learn_mean, track_running_stats=True),
                nn.ReLU(),
            )

        return layer
 
    def forward(self, encoder_output, length=None):
        pool = self._pooling(encoder_output, length)

        for layer in self.emb_layers:
            pool, emb = layer(pool), layer[: self.emb_id](pool)

        pool = pool.squeeze(-1)
        emb = emb.squeeze(-1)
        return pool, emb

# if __name__ == "__main__":
#     import yaml
#     cfg_path = "models/titanet/config/titanet_medium.yaml"
#     with open(cfg_path) as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
    
#     titanet_featurizer = ConvASREncoder(**cfg["featurizer"])
#     print(titanet_featurizer)