import random

import torch
import torch.nn as nn

from examples.audio_processing.spec_augment.spec_augment import SpecAugment

class MaskedPatchAugmentation(nn.Module):
    """
        Zeroes out fixed size time patches of the spectrogram.
        All samples in batch are guaranteed to have the same amount of masked time steps.
        Optionally also performs frequency masking in the same way as SpecAugment.
        Args:
            patch_size (int): up to how many time steps does one patch consist of.
                Defaults to 48.
            mask_patches (float): how many patches should be masked in each sample.
                if >= 1., interpreted as number of patches (after converting to int)
                if <1.,   interpreted as fraction of total tokens to be masked (number of patches is rounded up)
                Defaults to 10.
            freq_masks (int): how many frequency segments should be cut.
                Defaults to 0.
            freq_width (int): maximum number of frequencies to be cut in a segment.
                Defaults to 0.
    """

    def __init__(
        self, patch_size: int = 48, mask_patches: float = 10.0, freq_masks: int = 0, freq_width: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        if mask_patches >= 1:
            self.mask_patches = int(mask_patches)
        elif mask_patches >= 0:
            self._mask_fraction = mask_patches
            self.mask_patches = None
        else:
            raise ValueError('mask_patches cannot be negative')

        if freq_masks > 0:
            self.spec_augment = SpecAugment(
                    freq_masks=freq_masks, 
                    time_masks=0, 
                    freq_width=freq_width, 
                    time_width=0
                )
        else:
            self.spec_augment = None

    @torch.no_grad()
    def forward(self, input_spec, length):
        augmented_spec = input_spec
        min_len = torch.min(length)

        if self.mask_patches is None:
            # masking specified as fraction
            len_fraction = int(min_len * self._mask_fraction)
            mask_patches = len_fraction // self.patch_size + int(len_fraction % self.patch_size != 0)
        else:
            mask_patches = self.mask_patches

        if min_len < self.patch_size * mask_patches:
            mask_patches = min_len // self.patch_size

        for idx in range(input_spec.shape[0]):
            cur_len = length[idx]
            patches = range(cur_len // self.patch_size - 1)
            masked_patches = random.sample(patches, mask_patches)

            for mp in masked_patches:
                augmented_spec[idx, :, mp * self.patch_size : (mp + 1) * self.patch_size] = 0.0

        if self.spec_augment is not None:
            augmented_spec, length  = self.spec_augment(input_spec=augmented_spec, length=length)

        return augmented_spec, length






