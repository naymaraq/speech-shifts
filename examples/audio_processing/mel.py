import math

import torchaudio
from examples.audio_processing.base import AudioPreprocessor


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):

    def __init__(
            self,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            window="hann",
            n_fft=400,
            lowfreq=0,
            highfreq=None,
            n_mels=128,
            **kwargs):

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        super().__init__(n_window_size, n_window_stride)

        mel_kwargs = {"sample_rate": self._sample_rate}
        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))
        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride
        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels
        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )
        mel_kwargs['window_fn'] = window_fn

        self.featurizer = torchaudio.transforms.MelSpectrogram(**mel_kwargs)

    def get_features(self, input_signal, length):

        features = self.featurizer(input_signal)
        seq_len = torch.ceil(length.to(torch.float32) / self.hop_length).to(dtype=torch.long)
        return features, seq_len


from torch_stft import STFT
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from librosa.util import tiny


class STFTPatch(STFT):
    def forward(self, input_data):
        return super().transform(input_data)[0]


# Create helper class for STFT that yields num_frames = num_samples // hop_length
class STFTExactPad(STFTPatch):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, *params, **kw_params):
        super().__init__(*params, **kw_params)
        self.pad_amount = (self.filter_length - self.hop_length) // 2

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = librosa.filters.window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= self.filter_length / self.hop_length

        inverse_transform = inverse_transform[:, :, self.pad_amount:]
        inverse_transform = inverse_transform[:, :, : -self.pad_amount:]

        return inverse_transform


def normalize_batch(x, seq_len, normalize_type):
    if normalize_type == "per_feature":
        x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            if x[i, :, : seq_len[i]].shape[1] == 1:
                raise ValueError(
                    "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                    "in torch.std() returning nan"
                )
            x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
            x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2)
    else:
        return x


def splice_frames(x, frame_splicing):
    """ Stacks frames together across feature dim
    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames
    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


from typing import Optional

import torch
from packaging import version


# Library version globals
TORCH_VERSION = None
TORCH_VERSION_MIN = version.Version('1.7')


def stft_patch(
        input: torch.Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        pad_mode: str = 'reflect',
        normalized: bool = False,
        onesided: Optional[bool] = None,
        return_complex: Optional[bool] = None,
):
    """
    Patch over torch.stft for PyTorch <= 1.6.
    Arguments are same as torch.stft().
    # TODO: Remove once PyTorch 1.7+ is a requirement.
    """
    global TORCH_VERSION
    if TORCH_VERSION is None:
        TORCH_VERSION = version.parse(torch.__version__)

        # logging.warning(
        #     "torch.stft() signature has been updated for PyTorch 1.7+\n"
        #     "Please update PyTorch to remain compatible with later versions of NeMo."
        # )

    if TORCH_VERSION < TORCH_VERSION_MIN:
        return torch.stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=True,
        )
    else:
        return torch.stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )


CONSTANT = 1e-5
from torch import nn
import librosa


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
            self,
            sample_rate=16000,
            n_window_size=320,
            n_window_stride=160,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            nfilt=64,
            lowfreq=0,
            highfreq=None,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2 ** -24,
            dither=CONSTANT,
            pad_to=16,
            max_duration=16.7,
            frame_splicing=1,
            stft_exact_pad=False,
            stft_conv=False,
            pad_value=0,
            mag_power=2.0,
    ):
        super().__init__()
        self.log_zero_guard_value = log_zero_guard_value
        if (
                n_window_size is None
                or n_window_stride is None
                or not isinstance(n_window_size, int)
                or not isinstance(n_window_stride, int)
                or n_window_size <= 0
                or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        #         logging.info(f"PADDING: {pad_to}")

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_exact_pad = stft_exact_pad
        self.stft_conv = stft_conv

        if stft_conv:
            # logging.info("STFT using conv")
            if stft_exact_pad:
                #                 logging.info("STFT using exact pad")
                self.stft = STFTExactPad(self.n_fft, self.hop_length, self.win_length, window)
            else:
                self.stft = STFTPatch(self.n_fft, self.hop_length, self.win_length, window)
        else:
            #             logging.info("STFT using torch")
            torch_windows = {
                'hann': torch.hann_window,
                'hamming': torch.hamming_window,
                'blackman': torch.blackman_window,
                'bartlett': torch.bartlett_window,
                'none': None,
            }
            window_fn = torch_windows.get(window, None)
            window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
            self.register_buffer("window", window_tensor)
            self.stft = lambda x: stft_patch(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=False if stft_exact_pad else True,
                window=self.window.to(dtype=torch.float),
                return_complex=False,
            )

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )
        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type

    #         logging.debug(f"sr: {sample_rate}")
    #         logging.debug(f"n_fft: {self.n_fft}")
    #         logging.debug(f"win_length: {self.win_length}")
    #         logging.debug(f"hop_length: {self.hop_length}")
    #         logging.debug(f"n_mels: {nfilt}")
    #         logging.debug(f"fmin: {lowfreq}")
    #         logging.debug(f"fmax: {highfreq}")

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        return torch.ceil(seq_len / self.hop_length).to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    @torch.no_grad()
    def forward(self, x, seq_len):
        seq_len = self.get_seq_len(seq_len.float())

        if self.stft_exact_pad and not self.stft_conv:
            p = (self.n_fft - self.hop_length) // 2
            x = torch.nn.functional.pad(x.unsqueeze(1), (p, p), "reflect").squeeze(1)

        # dither
        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # disable autocast to get full range of stft values
        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        # torch returns real, imag; so convert to magnitude
        if not self.stft_conv:
            x = torch.sqrt(x.pow(2).sum(-1))

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of
        # `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(x.device)
        mask = mask.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)

        return x, seq_len


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):
    """Featurizer module that converts wavs to mel spectrograms.
        We don't use torchaudio's implementation here because the original
        implementation is not the same, so for the sake of backwards-compatibility
        this will use the old FilterbankFeatures for now.
        Args:
            sample_rate (int): Sample rate of the input audio data.
                Defaults to 16000
            window_size (float): Size of window for fft in seconds
                Defaults to 0.02
            window_stride (float): Stride of window for fft in seconds
                Defaults to 0.01
            n_window_size (int): Size of window for fft in samples
                Defaults to None. Use one of window_size or n_window_size.
            n_window_stride (int): Stride of window for fft in samples
                Defaults to None. Use one of window_stride or n_window_stride.
            window (str): Windowing function for fft. can be one of ['hann',
                'hamming', 'blackman', 'bartlett']
                Defaults to "hann"
            normalize (str): Can be one of ['per_feature', 'all_features']; all
                other options disable feature normalization. 'all_features'
                normalizes the entire spectrogram to be mean 0 with std 1.
                'pre_features' normalizes per channel / freq instead.
                Defaults to "per_feature"
            n_fft (int): Length of FT window. If None, it uses the smallest power
                of 2 that is larger than n_window_size.
                Defaults to None
            preemph (float): Amount of pre emphasis to add to audio. Can be
                disabled by passing None.
                Defaults to 0.97
            features (int): Number of mel spectrogram freq bins to output.
                Defaults to 64
            lowfreq (int): Lower bound on mel basis in Hz.
                Defaults to 0
            highfreq  (int): Lower bound on mel basis in Hz.
                Defaults to None
            log (bool): Log features.
                Defaults to True
            log_zero_guard_type(str): Need to avoid taking the log of zero. There
                are two options: "add" or "clamp".
                Defaults to "add".
            log_zero_guard_value(float, or str): Add or clamp requires the number
                to add with or clamp to. log_zero_guard_value can either be a float
                or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
                passed.
                Defaults to 2**-24.
            dither (float): Amount of white-noise dithering.
                Defaults to 1e-5
            pad_to (int): Ensures that the output size of the time dimension is
                a multiple of pad_to.
                Defaults to 16
            frame_splicing (int): Defaults to 1
            stft_exact_pad (bool): If True, uses pytorch_stft and convolutions with
                padding such that num_frames = num_samples / hop_length. If False,
                stft_conv will be used to determine how stft will be performed.
                Defaults to False
            stft_conv (bool): If True, uses pytorch_stft and convolutions. If
                False, uses torch.stft.
                Defaults to False
            pad_value (float): The value that shorter mels are padded with.
                Defaults to 0
            mag_power (float): The power that the linear spectrogram is raised to
                prior to multiplication with mel basis.
                Defaults to 2 for a power spec
        """

    def __init__(
            self,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_window_size=None,
            n_window_stride=None,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            features=64,
            lowfreq=0,
            highfreq=None,
            log=True,
            log_zero_guard_type="add",
            log_zero_guard_value=2 ** -24,
            dither=1e-5,
            pad_to=16,
            frame_splicing=1,
            stft_exact_pad=False,
            stft_conv=False,
            pad_value=0,
            mag_power=2.0,
            **kwargs
    ):
        super().__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        self.featurizer = FilterbankFeatures(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            stft_exact_pad=stft_exact_pad,
            stft_conv=stft_conv,
            pad_value=pad_value,
            mag_power=mag_power,
        )

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks