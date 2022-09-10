import torch
from torch.nn.modules.loss import _Loss

class AngularSoftmaxLoss(_Loss):
    def __init__(self, scale=20.0, margin=1.35):
        super().__init__()

        self.eps = 1e-7
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        numerator = self.scale * torch.cos(
            torch.acos(torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            + self.margin
        )
        excl = torch.cat(
            [torch.cat((logits[i, :y], logits[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.scale * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)