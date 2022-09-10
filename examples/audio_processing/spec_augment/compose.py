import random

class Compose:
    def __init__(self, perturbations, rng=None):
        self.perturbations = perturbations
        self._rng = random.Random() if rng is None else rng

    def __call__(self, arg1, arg2):
        for prob, trans in self.perturbations:
            if self._rng.random() <= prob:
                arg1, arg2 = trans(arg1, arg2)
        return arg1, arg2