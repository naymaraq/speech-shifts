import torch.nn as nn
import torch.nn.functional as F

from  examples.models.jasper_block.utils import init_weights

class SpeakerClassifier(nn.Module):

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        angular: bool = False,
        init_mode: str = "xavier_uniform",
    ):
        super().__init__()
        self.angular = angular
        bias = False if self.angular else True
        self._num_classes = num_classes
        self.final = nn.Linear(feat_in, self._num_classes, bias=bias)
        self.apply(lambda x: init_weights(x, mode=init_mode))


    def forward(self, pool, emb):
        if self.angular:
            self.final.weight.data = F.normalize(self.final.weight.data, p=2, dim=1)
            pool = F.normalize(pool, p=2, dim=1)

        out = self.final(pool)
        return out