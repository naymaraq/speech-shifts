from torch import nn
import torch

class CrossEntropyLoss(nn.CrossEntropyLoss):
    __loss_name__ = "CrossEntropyLoss"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        return super().forward(input=logits, target=labels)