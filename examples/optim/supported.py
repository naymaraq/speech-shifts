from torch.optim import AdamW, Adam, SGD
from torch.optim import SGD

from examples.optim.cosine_annealing import CosineAnnealing

supported_optimizers = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD
}

supported_schedulars = {
    "CosineAnnealing": CosineAnnealing
}