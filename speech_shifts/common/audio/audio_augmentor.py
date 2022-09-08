import random

class AudioAugmentor:
    def __init__(self, perturbations=None, mutex_perturbations=None, rng=None):
        self._rng = random.Random() if rng is None else rng
        self._pipeline = perturbations if perturbations is not None else []
        self._mutex_perturbations = mutex_perturbations if mutex_perturbations is not None else []

    def isinstance(self, perturbation, perturbation_list):
        for p_type in perturbation_list:
            if isinstance(perturbation, p_type):
                return True
        return False
    
    def __call__(self, segment):
        applied=False
        for (prob, p) in self._pipeline:
            if self._rng.random() <= prob:
                if self.isinstance(p, self._mutex_perturbations):
                    if not applied:
                        p(segment)
                        applied = True
                    else:
                        continue
                else:
                    p(segment)