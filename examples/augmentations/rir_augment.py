import random
from examples.augmentations.noise_augment import NoisePerturbation
from examples.augmentations.impulse_augment import ImpulsePerturbation
from speech_shifts.common.audio.perturb import Perturbation

class RirAndNoisePerturbation(Perturbation):
    """
        RIR augmentation with additive foreground and background noise.
        In this implementation audio data is augmented by first convolving the audio with a Room Impulse Response
        and then adding foreground noise and background noise at various SNRs. RIR, foreground and background noises
        should either be supplied with a manifest file or as tarred audio files (faster).
        Different sets of noise audio files based on the original sampling rate of the noise. This is useful while
        training a mixed sample rate model. For example, when training a mixed model with 8 kHz and 16 kHz audio with a
        target sampling rate of 16 kHz, one would want to augment 8 kHz data with 8 kHz noise rather than 16 kHz noise.
        Args:
            rir_manifest_path: Manifest file for RIRs
            rir_tar_filepaths: Tar files, if RIR audio files are tarred
            rir_prob: Probability of applying a RIR
            noise_manifest_paths: Foreground noise manifest path
            min_snr_db: Min SNR for foreground noise
            max_snr_db: Max SNR for background noise,
            noise_tar_filepaths: Tar files, if noise files are tarred
            apply_noise_rir: Whether to convolve foreground noise with a a random RIR
            orig_sample_rate: Original sampling rate of foreground noise audio
            max_additions: Max number of times foreground noise is added to an utterance,
            max_duration: Max duration of foreground noise
            bg_noise_manifest_paths: Background noise manifest path
            bg_min_snr_db: Min SNR for background noise
            bg_max_snr_db: Max SNR for background noise
            bg_noise_tar_filepaths: Tar files, if noise files are tarred
            bg_orig_sample_rate: Original sampling rate of background noise audio
    """

    def __init__(
        self,
        rir_manifest_path=None,
        rir_prob=0.5,
        noise_manifest_paths=None,
        min_snr_db=0,
        max_snr_db=50,
        apply_noise_rir=False,
        orig_sample_rate=16000,
        max_additions=5,
        max_duration=2.0,
        bg_noise_manifest_paths=None,
        bg_min_snr_db=10,
        bg_max_snr_db=50,
        bg_orig_sample_rate=16000,
        shift_impulse=False
    ):

        self._rir_prob = rir_prob
        self._rng = random.Random()

        self._rir_perturber = ImpulsePerturbation(
            manifest_path=rir_manifest_path,
            shift_impulse=shift_impulse,
        )
        self.fg_perturber = None
        self.bg_perturber = None
        if noise_manifest_paths:
            self.fg_perturber = NoisePerturbation(
                manifest_path=noise_manifest_paths,
                min_snr_db=min_snr_db,
                max_snr_db=max_snr_db,
                orig_sr=orig_sample_rate,
            )
        self._max_additions = max_additions
        self._max_duration = max_duration
        if bg_noise_manifest_paths:
            self.bg_perturber = NoisePerturbation(
                manifest_path=bg_noise_manifest_paths,
                min_snr_db=bg_min_snr_db,
                max_snr_db=bg_max_snr_db,
                orig_sr=bg_orig_sample_rate,
            )

        self._apply_noise_rir = apply_noise_rir

    def __call__(self, data):
        prob = self._rng.uniform(0.0, 1.0)

        if prob < self._rir_prob:
            self._rir_perturber(data)

        data_rms = data.rms_db
        if self.fg_perturber:
            noise = self.fg_perturber.get_one_noise_sample(data.sample_rate)
            if self._apply_noise_rir:
                self._rir_perturber(noise)
            
            self.fg_perturber.perturb_with_foreground_noise(
                data, noise, data_rms=data_rms, max_noise_dur=self._max_duration, max_additions=self._max_additions
            )

        if self.bg_perturber:
            noise = self.bg_perturber.get_one_noise_sample(data.sample_rate)
            self.bg_perturber.perturb_with_input_noise(data, noise, data_rms=data_rms)