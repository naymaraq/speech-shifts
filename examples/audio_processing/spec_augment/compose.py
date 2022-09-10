import random

class Compose:
    def __init__(self, *args):
        self.transformations = args
        self._rng = random.Random()

    def __call__(self, arg1, arg2):
        for prob, trans in self.transformations:
            if self._rng.random() <= prob:
                arg1, arg2 = trans(arg1, arg2)
        return arg1, arg2