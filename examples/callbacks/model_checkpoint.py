import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def on_keyboard_interrupt(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.save_checkpoint(trainer)