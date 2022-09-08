import random

from speech_shifts.common.audio.perturb import Perturbation

class RandomCropPerturbation(Perturbation):

    def __init__(
            self,
            min_dur,
            max_dur,
            rng=None,
    ):
        self.min_dur = min_dur
        self.max_dur = max_dur
        self._rng = random.Random() if rng is None else rng
            
    def random_crop(self, data):
        dur = self._rng.uniform(self.min_dur, self.max_dur)
        dur = min(dur, data.duration)
        start = self._rng.uniform(0, data.duration - dur)
        data.subsegment(start_time=start, end_time=start + dur)

    def __call__(self, data):
        self.random_crop(data)
