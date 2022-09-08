import random
import numpy as np

from speech_shifts.common.audio.perturb import Perturbation
from speech_shifts.common.audio.segments import AudioSegment

from examples.augmentations.collections import SpeechLabelManifestProcessor
from examples.augmentations.manifest import parse_noise_item


def read_one_audiosegment(manifest, target_sr, rng):
    audio_record = rng.sample(manifest.data, 1)[0]
    audio_file = audio_record.audio_file
    offset = 0 if audio_record.offset is None else audio_record.offset
    duration = 0 if audio_record.duration is None else audio_record.duration

    return AudioSegment.from_file(audio_file, target_sr=target_sr, offset=offset, duration=duration)


class NoisePerturbation(Perturbation):
    def __init__(
            self,
            manifest_path=None,
            min_snr_db=10,
            max_snr_db=50,
            max_gain_db=300.0,
            rng=None,
            orig_sr=16000,
    ):
        self._manifest = SpeechLabelManifestProcessor(manifest_path,
                                                      parse_func=parse_noise_item,
                                                      index_by_file_id=True)
        self._audiodataset = None
        self._orig_sr = orig_sr

        self._rng = random.Random() if rng is None else rng
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._max_gain_db = max_gain_db

    @property
    def orig_sr(self):
        return self._orig_sr

    def __call__(self, data):
        noise = read_one_audiosegment(
            self._manifest,
            data.sample_rate,
            self._rng,
        )
        self.perturb_with_input_noise(data, noise)

    
    def get_one_noise_sample(self, target_sr):
        noise = read_one_audiosegment(
            self._manifest,
            target_sr,
            self._rng,
        )
        return noise

    def perturb_with_input_noise(self, data, noise, data_rms=None):
        
        if data.rms != 0.0 and noise.rms != 0.0:

            snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
            if data_rms is None:
                data_rms = data.rms_db
            
            # noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
            noise_gain_db = data_rms - noise.rms_db - snr_db
            # logging.debug("noise: %s %s %s", snr_db, noise_gain_db, noise_record.audio_file)

            # calculate noise segment to use
            start_time = self._rng.uniform(0.0, noise.duration - data.duration)
            if noise.duration > (start_time + data.duration):
                noise.subsegment(start_time=start_time, end_time=start_time + data.duration)

            # adjust gain for snr purposes and superimpose
            noise.gain_db(noise_gain_db)

            if noise._samples.shape[0] < data._samples.shape[0]:
                noise_idx = self._rng.randint(0, data._samples.shape[0] - noise._samples.shape[0])
                data._samples[noise_idx: noise_idx + noise._samples.shape[0]] += noise._samples

            else:
                data._samples += noise._samples


    def perturb_with_foreground_noise(self, data, noise, data_rms=None, max_noise_dur=2, max_additions=1,):
        
        if data.rms != 0.0 and noise.rms != 0.0:
            snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
            if data_rms is None:
                data_rms = data.rms_db

            noise_gain_db = min(data_rms - noise.rms_db - snr_db, self._max_gain_db)
            n_additions = self._rng.randint(1, max_additions)

            for i in range(n_additions):
                noise_dur = self._rng.uniform(0.0, max_noise_dur)
                start_time = self._rng.uniform(0.0, noise.duration)
                start_sample = int(round(start_time * noise.sample_rate))
                end_sample = int(round(min(noise.duration, (start_time + noise_dur)) * noise.sample_rate))
                noise_samples = np.copy(noise._samples[start_sample:end_sample])
                # adjust gain for snr purposes and superimpose
                noise_samples *= 10.0 ** (noise_gain_db / 20.0)

                if noise_samples.shape[0] > data._samples.shape[0]:
                    noise_samples = noise_samples[0 : data._samples.shape[0]]

                noise_idx = self._rng.randint(0, data._samples.shape[0] - noise_samples.shape[0])
                data._samples[noise_idx : noise_idx + noise_samples.shape[0]] += noise_samples