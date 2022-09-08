import torch
from torch import nn as nn
from torch.nn import functional as F
from numpy import inf

from examples.models.titanet.utils import get_same_padding, lens_to_mask, get_statistics_with_mask

class StatsPoolLayer(nn.Module):
    """
    Statistics and time average pooling (TAP) layer
    This computes mean and variance statistics across time dimension (dim=-1)
    input:
        feat_in: input channel feature length
        pool_mode: type of pool mode
        supported modes are xvector (mean and variance),
        tap (mean)
    output:
        pooled: statistics of feature input
    """

    def __init__(self, feat_in: int, pool_mode: str = 'xvector'):
        super().__init__()
        self.pool_mode = pool_mode
        self.feat_in = feat_in
        if self.pool_mode == 'xvector':
            self.feat_in += feat_in
        elif self.pool_mode == 'tap':
            self.feat_in = feat_in
        else:
            raise ValueError("pool mode for stats must be either tap or xvector based")

    def forward(self, encoder_output, length=None):
        mean = encoder_output.mean(dim=-1)  # Time Axis
        if self.pool_mode == 'xvector':
            std = encoder_output.std(dim=-1)
            pooled = torch.cat([mean, std], dim=-1)
        else:
            pooled = mean
        return pooled


class TDNNModule(nn.Module):
    """
    Time Delayed Neural Module (TDNN) - 1D
    input:
        inp_filters: input filter channels for conv layer
        out_filters: output filter channels for conv layer
        kernel_size: kernel weight size for conv layer
        dilation: dilation for conv layer
        stride: stride for conv layer
        padding: padding for conv layer (default None: chooses padding value such that input and output feature shape matches)
    output:
        tdnn layer output
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


class AttentivePoolLayer(nn.Module):
    """
    Attention pooling layer for pooling speaker embeddings
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input feature channel length from encoder
        attention_channels: intermediate attention channel size
        kernel_size: kernel_size for TDNN and attention conv1d layers (default: 1)
        dilation: dilation size for TDNN and attention conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
    ):
        super().__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TDNNModule(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels, out_channels=inp_filters, kernel_size=kernel_size, dilation=dilation,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)