import torch
import torch.nn as nn

from examples.models.titanet.se_utils import _se_pool_step_script_train, _se_pool_step_script_infer


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
        activation=None,
        quantize: bool = False,
    ):
        """
        Squeeze-and-Excitation sub-module.
        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(SqueezeExcite, self).__init__()
        self.interpolation_mode = interpolation_mode
        self._quantize = quantize

        self.pool = None  # prepare a placeholder which will be updated

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Set default context window
        self.change_context_window(context_window=context_window)

        # Set default max sequence length
        self.max_len = 16

    def forward(self, x, lengths):
        return self.forward_for_export(x, lengths)

    def forward_for_export(self, x, lengths):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        max_len = x.shape[-1]
        self.max_len = max(max_len, self.max_len)
        seq_range = torch.arange(0, self.max_len, device=x.device)
        
        # Computes in float32 to avoid instabilities during training with AMP.
        with torch.cuda.amp.autocast(enabled=False):
            # Create sample mask - 1 represents value, 0 represents pad

            mask = seq_range[:max_len].expand(lengths.size(0), -1) < lengths.unsqueeze(-1)  # [B, T]; bool
            mask = mask.unsqueeze(1)
            mask = ~mask  # 0 represents value, 1 represents pad
            x = x.float()  # For stable AMP, SE must be computed at fp32.
            x.masked_fill_(mask, 0.0)  # mask padded values explicitly to 0
            y = self._se_pool_step(x, mask)  # [B, C, 1]
            y = y.transpose(1, -1)  # [B, 1, C]
            y = self.fc(y)  # [B, 1, C]
            y = y.transpose(1, -1)  # [B, C, 1]

            y = torch.sigmoid(y)
            y = x * y
        return y, lengths

    def _se_pool_step(self, x, mask):
        # Negate mask back to represent 1 for signal and 0 for padded timestep.
        mask = ~mask

        if self.context_window < 0:
            # [B, C, 1] - Masked Average over value + padding.
            y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).type(x.dtype)
        else:
            # [B, C, 1] - Masked Average over value + padding with limited context.
            # During training randomly subsegments a context_window chunk of timesteps.
            # During inference selects only the first context_window chunk of timesteps.
            if self.training:
                y = _se_pool_step_script_train(x, self.context_window, mask)
            else:
                y = _se_pool_step_script_infer(x, self.context_window, mask)
        return y


    def change_context_window(self, context_window: int):
        """
        Update the context window of the SqueezeExcitation module, in-place if possible.
        Will update the pooling layer to either nn.AdaptiveAvgPool1d() (for global SE) or nn.AvgPool1d()
        (for limited context SE).
        If only the context window is changing but still a limited SE context block - then
        the earlier instance of nn.AvgPool1d() will be updated.
        Args:
            context_window: An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.
                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
        """
        if hasattr(self, 'context_window'):
            print(f"Changing Squeeze-Excitation context window from {self.context_window} to {context_window}")

        self.context_window = context_window