import torch
import torch.nn.functional as F

import numpy as np
from examples.optim.supported import supported_optimizers
from examples.losses.supported import supported_losses
from examples.algorithms.base import BaseSpeakerEmbeddingModel
from examples.utils.grl import GRL

def num_params(model):
    n = 0 
    if isinstance(model, torch.nn.Module):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n

def is_multiview(loss):
    for k in  ["proto", "ge2e", "angular_proto"]:
        if isinstance(loss, supported_losses[k]):
            return True
    return False

class DANN(BaseSpeakerEmbeddingModel):

    def __init__(self, 
                 preprocessor,
                 train_spec_augmentor,
                 featurizer,
                 discriminator,
                 loss,
                 optim_config,
                 dann_params,
                 classifier=None
                 ):
        super(DANN, self).__init__()

        self.preprocessor = preprocessor
        self.train_spec_augmentor = train_spec_augmentor
        self.featurizer = featurizer
        self.classifier = classifier
        self.discriminator = discriminator

        self.loss = loss
        self.multiview = is_multiview(self.loss)

        self.optim_config = optim_config
        self.dann_params = dann_params

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
        self._train_dls_len = len(self._train_dls)
        self.train_y_map = {y:ix for ix, y in enumerate(set(self.train_dataset.y_array.tolist()))}
        self.train_domain_map = {dom:ix for ix, dom in enumerate(set(self.train_dataset.metadata_array[:, 0].tolist()))}

    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_length = self.preprocessor(input_signal, input_signal_length)
        processed_signal, processed_signal_length = self.train_spec_augmentor(processed_signal, processed_signal_length)
        pool, emb = self.featurizer(audio_signal=processed_signal, length=processed_signal_length)
        
        logits = None
        if self.classifier is not None:
            logits = self.classifier(pool, emb)
        return logits, pool, emb
    
    def get_p(self):
        p = float(self.global_step / (self.dann_params["epoch"]*self._train_dls_len))
        return p

    def get_lambda_p(self, p):
        lambda_p = 2. / (1. + np.exp(-self.dann_params["gamma"] * p)) - 1
        return lambda_p
    
    def lr_schedule_step(self, p):
        alpha = self.dann_params["alpha"]
        beta = self.dann_params["beta"]
        lr = self.dann_params["lr"]
        wd = self.dann_params["wd"]
        for param_group in self.optimizers().param_groups:
            param_group["lr"] = param_group["lr_mult"] * lr / (1 + alpha * p) ** beta
            param_group["weight_decay"] = wd * param_group["decay_mult"]

    def training_step(self, batch, batch_idx):

        p = self.get_p()
        lambda_p = self.get_lambda_p(p)
        self.lr_schedule_step(p)
        
        audio_signal, audio_lengths, labels, metadata, indices = batch
        logits, pool, emb = self.forward(input_signal=audio_signal, input_signal_length=audio_lengths)
        pool_rev = GRL.apply(pool, lambda_p)
        domain_logits = self.discriminator(pool_rev, None)
        
        domain_labels = torch.stack([torch.tensor(self.train_domain_map[label.item()]).long() for label in metadata[:, 0]]).to(metadata.device)


        dann_loss = F.cross_entropy(domain_logits, domain_labels)

        if (isinstance(self.loss, supported_losses["aam"]) or 
            isinstance(self.loss, supported_losses["cross_entropy"])):
            labels = torch.stack([torch.tensor(self.train_y_map[label.item()]).long() for label in labels]).to(labels.device)
            se_loss = self.loss(logits=logits, labels=labels)
        else:
            N = audio_signal.shape[0] // self.n_views
            emb = emb.view(N, self.n_views, -1)
            se_loss = self.loss(x=emb, labels=None)
        
        loss = se_loss + dann_loss
        self.log(f"{type(self.loss).__name__}", se_loss)
        self.log("DANN", dann_loss)
        self.log("Loss", loss)
        self.log("p", p)
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
        model_parameter = [
            {
                "params": self.featurizer.parameters(),
                "lr_mult": 0.05,
                'decay_mult': 2,
            },
            {
                "params": self.discriminator.parameters(),
                "lr_mult":  0.01,
                'decay_mult': 2,
            }
        ]

        if self.classifier is not None:
            model_parameter.append({
                "params": self.classifier.parameters(),
                "lr_mult": 0.1,
                'decay_mult': 2,
                }
            )
        if num_params(self.loss) > 0:
            model_parameter.append({
                "params": self.loss.parameters(),
                "lr_mult": 0.01,
                'decay_mult': 2,
                }
            )

        optimizer_class = supported_optimizers[self.optim_config['type']]
        optimizer = optimizer_class(
            model_parameter, 
            **self.optim_config['params']
        )
        return optimizer