class Perturbation:

    def __call__(self, data):
        """
        Args:
            - data (AudioSegment): audio segment of type AudioSegment.
            data should be changed inplace.
        """
        raise NotImplementedError