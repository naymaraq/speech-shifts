import torch
import torch.nn as nn


@torch.jit.script
def _masked_conv_init_lens(lens: torch.Tensor, current_maxlen: int, original_maxlen: torch.Tensor):
    if current_maxlen > original_maxlen:
        new_lens = torch.arange(current_maxlen)
        new_max_lens = torch.tensor(current_maxlen)
    else:
        new_lens = lens
        new_max_lens = original_maxlen
    return new_lens, new_max_lens


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (tuple, list):
            self.pad_layer = nn.ConstantPad1d(padding, value=0.0)
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0] == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding) == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = torch.tensor(0)
            self.lens = torch.tensor(0)

    def get_seq_len(self, lens):
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                torch.div(
                    lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1,
                    self.conv.stride[0],
                    rounding_mode='trunc',
                )
                + 1
            )
        else:
            return (
                torch.div(
                    lens + sum(self._padding) - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1,
                    self.conv.stride[0],
                    rounding_mode='trunc',
                )
                + 1
            )

    def forward(self, x, lens):
        if self.use_mask:
            # Generally will be called by ConvASREncoder, but kept as single gpu backup.
            if x.size(2) > self.max_len:
                self.update_masked_length(x.size(2), device=lens.device)
            x = self.mask_input(x, lens)

        # Update lengths
        lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens

    def update_masked_length(self, max_len, seq_range=None, device=None):
        if seq_range is None:
            self.lens, self.max_len = _masked_conv_init_lens(self.lens, max_len, self.max_len)
            self.lens = self.lens.to(device)
        else:
            self.lens = seq_range
            self.max_len = max_len

    def mask_input(self, x, lens):
        max_len = x.size(2)
        mask = self.lens[:max_len].unsqueeze(0).to(lens.device) < lens.unsqueeze(1)
        x = x * mask.unsqueeze(1).to(device=x.device)
        return x


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x