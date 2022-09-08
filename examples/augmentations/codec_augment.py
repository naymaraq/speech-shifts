import random
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf

from speech_shifts.common.audio.perturb import Perturbation
from speech_shifts.common.audio.segments import AudioSegment


class TranscodePerturbation(Perturbation):
    """
        Audio codec augmentation. This implementation uses sox to transcode audio with low rate audio codecs,
        so users need to make sure that the installed sox version supports the codecs used here (G711 and amr-nb).
        Args:
            rng: Random number generator
    """

    def __init__(self, codecs=None, rng=None):
        self._rng = np.random.RandomState() if rng is None else rng
        self._codecs = codecs if codecs is not None else ["g711", "amr-nb", "ogg"]
        self.att_factor = 0.8  # to avoid saturation while writing to wav
        if codecs is not None:
            for codec in codecs:
                if codec not in ["g711", "amr-nb", "ogg"]:
                    raise ValueError(
                        f"TranscodePerturbation with {codec} isnot supported. Only {codecs} are supported"
                    )

    def __call__(self, data):
        max_level = np.max(np.abs(data._samples))
        if max_level > 0.8:
            norm_factor = self.att_factor / max_level
            norm_samples = norm_factor * data._samples
        else:
            norm_samples = data._samples
        orig_f = NamedTemporaryFile(suffix=".wav")
        sf.write(orig_f.name, norm_samples.transpose(), 16000)

        codec_ind = random.randint(0, len(self._codecs) - 1)
        if self._codecs[codec_ind] == "amr-nb":
            transcoded_f = NamedTemporaryFile(suffix="_amr.wav")
            rates = list(range(0, 4))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "ogg":
            transcoded_f = NamedTemporaryFile(suffix="_ogg.wav")
            rates = list(range(-1, 8))
            rate = rates[random.randint(0, len(rates) - 1)]
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0 -C {rate} -t ogg - | sox -t ogg - -V0 -b 16 -r 16000 {transcoded_f.name}",
                shell=True,
            )
        elif self._codecs[codec_ind] == "g711":
            transcoded_f = NamedTemporaryFile(suffix="_g711.wav")
            _ = subprocess.check_output(
                f"sox {orig_f.name} -V0  -r 8000 -c 1 -e a-law {transcoded_f.name} lowpass 3400 highpass 300",
                shell=True,
            )

        new_data = AudioSegment.from_file(transcoded_f.name, target_sr=16000)
        data._samples = new_data._samples[0: data._samples.shape[0]]