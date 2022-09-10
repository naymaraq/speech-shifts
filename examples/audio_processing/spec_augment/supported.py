from examples.audio_processing.spec_augment.spec_augment import SpecAugment
from examples.audio_processing.spec_augment.spec_cutout import SpecCutout
from examples.audio_processing.spec_augment.spec_patch_augment import SpecMaskedPatchAugmentation

supported = {
    "spec_augment": SpecAugment,
    "spec_cutout": SpecCutout,
    "masked_patch": SpecMaskedPatchAugmentation
}

