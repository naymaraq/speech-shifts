from examples.augmentations.speed_augment import SpeedPerturbation
from examples.augmentations.codec_augment import TranscodePerturbation
from examples.augmentations.noise_augment import NoisePerturbation
from examples.augmentations.random_crop_augment import RandomCropPerturbation
from examples.augmentations.impulse_augment import ImpulsePerturbation
from examples.augmentations.rir_augment import RirAndNoisePerturbation

supported_perturbations = {
    "noise": NoisePerturbation,
    "codec": TranscodePerturbation,
    "speed": SpeedPerturbation,
    "crop": RandomCropPerturbation,
    "impulse": ImpulsePerturbation,
    "rir": RirAndNoisePerturbation
} 