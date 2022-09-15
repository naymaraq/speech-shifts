import pytorch_lightning as pl
import torch
from examples.optim.supported import supported_optimizers, supported_schedulars
from examples.algorithms.cosine_scorer import CosineScorer
from examples.exp_manager.logger import sr_logger
from examples.losses.supported import supported_losses

def is_multiview(loss):
    for k in  ["proto", "ge2e", "angular_proto"]:
        if isinstance(loss, supported_losses[k]):
            return True
    return False

class SpeakerEmbeddingModel(pl.LightningModule):

    def __init__(self, 
                 preprocessor,
                 train_spec_augmentor,
                 featurizer,
                 loss,
                 optim_config,
                 classifier=None
                 ):
        super(SpeakerEmbeddingModel, self).__init__()

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

    def val_dataloader(self):
        return self._val_dls

    def setup_dataloaders(self, 
        train_dataset, 
        train_dataloader, 
        val_dataset, 
        val_dataloader, 
        id_val_dataset, 
        id_val_dataloader
    ):        
        
        if self.multiview:
            assert train_dataset.n_views > 1
            self.n_views = train_dataset.n_views

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.id_val_dataset = id_val_dataset
        self._train_dls = train_dataloader
        self._val_dls = [id_val_dataloader, val_dataloader]
        self.id_val_scorer = CosineScorer(input_trial_array=self.id_val_dataset.input_trial_array, 
                                          index2path=self.id_val_dataset._index2path)
        self.id_val_scorer.disable_compute()

        self.val_scorer = CosineScorer(input_trial_array=self.val_dataset.input_trial_array, 
                                          index2path=self.val_dataset._index2path)
        self.val_scorer.disable_compute()

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
            #emb = emb.reshape(self.n_views, -1, emb.size()[-1]).transpose(1, 0).squeeze(1)
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

    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            self.id_val_scorer.enable_compute()
            self.val_scorer.enable_compute()

            id_val_y_pred, id_val_y_true = self.id_val_scorer.compute()
            val_y_pred, val_y_true = self.val_scorer.compute()

            self.id_val_scorer.disable_compute()
            self.val_scorer.disable_compute()
            self.id_val_scorer.reset()
            self.val_scorer.reset()

            (val_eer_results, val_eer_results_str, 
             val_dcf_results, val_dcf_results_str) = self.val_dataset.eval(
                val_y_pred, 
                val_y_true, 
                self.val_dataset.trial_metadata_array
            )
            (id_val_eer_results, id_val_eer_results_str, 
             id_val_dcf_results, id_val_dcf_results_str) = self.id_val_dataset.eval(
                id_val_y_pred, 
                id_val_y_true, 
                self.id_val_dataset.trial_metadata_array
            )

            sr_logger.info("\nvalidation_eer (OOD)\n\n{}".format(val_eer_results_str))
            sr_logger.info("\nvalidation_dcf (OOD)\n\n{}".format(val_dcf_results_str))

            sr_logger.info("\nvalidation_eer (ID)\n\n{}".format(id_val_eer_results_str))
            sr_logger.info("\nvalidation_dcf (ID)\n\n{}".format(id_val_dcf_results_str))


            for res, name in [(val_eer_results,"EER"), (id_val_eer_results, "EER"),
                              (val_dcf_results,"DCF"), (id_val_dcf_results, "DCF")]:
                zero_count_keys = [key for key in res if ("count" in key) and res[key]==0]
                count_keys = [key for key in res if "count" in key]
                other_keys = [key for key in res if "_wg" in key]
                zero_eer_keys = [f"{name}_lang:"+key.split(":")[-1] for key in zero_count_keys]
                for k in count_keys + zero_eer_keys + other_keys:
                    del res[k]

            self.log("validation_eer (OOD)", val_eer_results, prog_bar=False, logger=True, sync_dist=True)
            self.log("validation_dcf (OOD)", val_dcf_results, prog_bar=False, logger=True, sync_dist=True)

            self.log("validation_eer (ID)", id_val_eer_results, prog_bar=False, logger=True, sync_dist=True)
            self.log("validation_dcf (ID)", id_val_dcf_results, prog_bar=False, logger=True, sync_dist=True)
            
            self.log("avg", {
                "avg_val_eer (OOD)": val_eer_results["EER_all"],
                "avg_val_dcf (OOD)": val_dcf_results["DCF_all"],
                "avg_val_eer (ID)": id_val_eer_results["EER_all"],
                "avg_val_dcf (ID)": id_val_dcf_results["DCF_all"],
                }, sync_dist=True)
            self.log("val_eer", val_eer_results["EER_all"], logger=False, sync_dist=True)
            self.log("val_dcf", val_dcf_results["DCF_all"], logger=False, sync_dist=True)
        else:
            sr_logger.info("Skipping EER calculation for sanity checking")
        

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