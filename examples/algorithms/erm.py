import torch
from examples.optim.supported import supported_optimizers, supported_schedulars
from examples.losses.supported import supported_losses
from examples.algorithms.base import BaseSpeakerEmbeddingModel

def is_multiview(loss):
    for k in  ["proto", "ge2e", "angular_proto"]:
        if isinstance(loss, supported_losses[k]):
            return True
    return False

class ERM(BaseSpeakerEmbeddingModel):

    def __init__(self, 
                 preprocessor,
                 train_spec_augmentor,
                 featurizer,
                 loss,
                 optim_config,
                 classifier=None
                 ):
        super(ERM, self).__init__()

        self.preprocessor = preprocessor
        self.train_spec_augmentor = train_spec_augmentor
        self.featurizer = featurizer
        self.classifier = classifier
        self.loss = loss
        self.multiview = is_multiview(self.loss)
        self.optim_config = optim_config

        self._train_dls = None
        self._val_dls = None
    
    def train_dataloader(self):
        return self._train_dls

    def setup_train_dataloaders(self, train_dataset, train_dataloader):        
        if self.multiview:
            assert train_dataset.n_views > 1
        self.n_views = train_dataset.n_views
        self.train_dataset = train_dataset
        self._train_dls = train_dataloader
        self.train_y_map = {y:ix for ix, y in enumerate(set(self.train_dataset.y_array.tolist()))}

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_length = self.preprocessor(input_signal, input_signal_length)
        processed_signal, processed_signal_length = self.train_spec_augmentor(processed_signal, processed_signal_length)
        pool, emb = self.featurizer(audio_signal=processed_signal, length=processed_signal_length)
        
        logits = None
        if self.classifier is not None:
            logits = self.classifier(pool, emb)
        return logits, pool, emb
    
    def training_step(self, batch, batch_idx):
        audio_signal, audio_lengths, labels, metadata, indices = batch
        logits, pool, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_lengths)
        if (isinstance(self.loss, supported_losses["aam"]) or 
            isinstance(self.loss, supported_losses["cross_entropy"])):
            labels = torch.stack([torch.tensor(self.train_y_map[label.item()]).long() for label in labels]).to(labels.device)
            loss = self.loss(logits=logits, labels=labels)
        else:
            N = audio_signal.shape[0] // self.n_views
            emb = emb.view(N, self.n_views, -1)
            loss = self.loss(x=emb, labels=None)
        
        self.log(f"{type(self.loss).__name__}", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx): 
        audio_signal, audio_signal_len, labels, metadata, indices = batch
        processed_signal, processed_signal_length = self.preprocessor(audio_signal, audio_signal_len) 
        _, embs = self.featurizer(processed_signal, processed_signal_length)
        if dataloader_idx == 0:
            self.id_val_scorer(embs, indices)
        elif dataloader_idx == 1:
            self.val_scorer(embs, indices)
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        audio_signal, audio_signal_len, labels, metadata, indices = batch
        processed_signal, processed_signal_length = self.preprocessor(audio_signal, audio_signal_len) 
        _, embs = self.featurizer(processed_signal, processed_signal_length)
        if dataloader_idx == 0:
            self.id_val_scorer(embs, indices)
        elif dataloader_idx == 1:
            self.val_scorer(embs, indices)
        elif dataloader_idx == 2:
            self.test_scorer(embs, indices)

    def configure_optimizers(self):
        optimizer_class = supported_optimizers[self.optim_config['type']]
        optimizer = optimizer_class(self.parameters(), **self.optim_config['params'])

        sched_config = self.optim_config.get('sched', None)
        if sched_config:
            schedule_class = supported_schedulars[sched_config['type']]
            schedule = schedule_class(optimizer, **sched_config['params'])
            schedule_dict = {
                'scheduler': schedule,
                'interval': 'step',
                'frequency': 1,
                'monitor': 'loss',
            }
            return [optimizer], [schedule_dict]
        else:
            return [optimizer]