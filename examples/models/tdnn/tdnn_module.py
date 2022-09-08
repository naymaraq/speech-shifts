import torch
from torch import nn as nn

from examples.models.jasper_block.utils import get_same_padding, init_weights, lens_to_mask


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


class MaskedSEModule(nn.Module):
    """
    Squeeze and Excite module implementation with conv1d layers
    input:
        inp_filters: input filter channel size 
        se_filters: intermediate squeeze and excite channel output and input size
        out_filters: output filter channel size
        kernel_size: kernel_size for both conv1d layers
        dilation: dilation size for both conv1d layers

    output:
        squeeze and excite layer output
    """

    def __init__(self, inp_filters: int, se_filters: int, out_filters: int, kernel_size: int = 1, dilation: int = 1):
        super().__init__()
        self.se_layer = nn.Sequential(
            nn.Conv1d(inp_filters, se_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.ReLU(),
            nn.BatchNorm1d(se_filters),
            nn.Conv1d(se_filters, out_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.Sigmoid(),
        )

    def forward(self, input, length=None):
        if length is None:
            x = torch.mean(input, dim=2, keep_dim=True)
        else:
            max_len = input.size(2)
            mask, num_values = lens_to_mask(length, max_len=max_len, device=input.device)
            x = torch.sum((input * mask), dim=2, keepdim=True) / (num_values)

        out = self.se_layer(x)
        return out * input


class TDNNSEModule(nn.Module):
    """
    Modified building SE_TDNN group module block from ECAPA implementation for faster training and inference
    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf)
    inputs:
        inp_filters: input filter channel size 
        out_filters: output filter channel size
        group_scale: scale value to group wider conv channels (deafult:8)
        se_channels: squeeze and excite output channel size (deafult: 1024/8= 128)
        kernel_size: kernel_size for group conv1d layers (default: 1)
        dilation: dilation size for group conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        group_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        init_mode: str = 'xavier_uniform',
    ):
        super().__init__()
        self.out_filters = out_filters
        padding_val = get_same_padding(kernel_size=kernel_size, dilation=dilation, stride=1)

        group_conv = nn.Conv1d(
            out_filters,
            out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding_val,
            groups=group_scale,
        )
        self.group_tdnn_block = nn.Sequential(
            TDNNModule(inp_filters, out_filters, kernel_size=1, dilation=1),
            group_conv,
            nn.ReLU(),
            nn.BatchNorm1d(out_filters),
            TDNNModule(out_filters, out_filters, kernel_size=1, dilation=1),
        )

        self.se_layer = MaskedSEModule(out_filters, se_channels, out_filters)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, input, length=None):
        x = self.group_tdnn_block(input)
        x = self.se_layer(x, length)
        return x + input