from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging

from examples.callbacks.model_checkpoint import CustomModelCheckpoint

supported_callbacks = {
    "StochasticWeightAveraging": StochasticWeightAveraging,
    "LearningRateMonitor": LearningRateMonitor,
    "CustomModelCheckpoint": CustomModelCheckpoint,
    "ModelCheckpoint": ModelCheckpoint
}

