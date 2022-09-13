import random
import numpy as np
from scipy import signal

from speech_shifts.common.audio.perturb import Perturbation
from examples.augmentations.noise_augment import read_one_audiosegment
from examples.augmentations.collections import SpeechLabelManifestProcessor
from examples.augmentations.manifest import parse_noise_item

class ImpulsePerturbation(Perturbation):
    """
    Convolves audio with a Room Impulse Response.
    Args:
        manifest_path (list): Manifest file for RIRs
        audio_tar_filepaths (list): Tar files, if RIR audio files are tarred
        shuffle_n (int): Shuffle parameter for shuffling buffered files from the tar files
        shift_impulse (bool): Shift impulse response to adjust for delay at the beginning
    """

    def __init__(self, 
            manifest_path=None, 
            shift_impulse=False,
            rng=None):

        self._manifest = SpeechLabelManifestProcessor(manifest_path,
                                                      parse_func=parse_noise_item,
                                                      index_by_file_id=True)
        self._shift_impulse = shift_impulse
        self._rng = random.Random() if rng is None else rng

    def __call__(self, data):

        impulse = read_one_audiosegment(
            self._manifest,
            data.sample_rate,
            self._rng
        )
        if not self._shift_impulse:
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            data._samples = signal.fftconvolve(data._samples, impulse_norm, "same")
        else:
            # Find peak and shift peak to left
            impulse_norm = (impulse.samples - min(impulse.samples)) / (max(impulse.samples) - min(impulse.samples))
            max_ind = np.argmax(np.abs(impulse_norm))

            impulse_resp = impulse_norm[max_ind:]
            delay_after = len(impulse_resp)
            data._samples = signal.fftconvolve(data._samples, impulse_resp, "full")[:-delay_after]