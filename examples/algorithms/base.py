import pytorch_lightning as pl

from examples.algorithms.cosine_scorer import CosineScorer
from examples.exp_manager.logger import sr_logger

class BaseSpeakerEmbeddingModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def val_dataloader(self):
        return self._val_dls
    
    def test_dataloader(self):
        return self._test_dls
    
    def setup_test_dataloaders(self, 
                              val_dataset, 
                              val_dataloader, 
                              id_val_dataset, 
                              id_val_dataloader,
                              test_dataset,
                              test_dataloader):

        self.val_dataset = val_dataset
        self.id_val_dataset = id_val_dataset
        self.test_dataset = test_dataset
        self._test_dls = [id_val_dataloader, val_dataloader, test_dataloader]
        self.id_val_scorer = CosineScorer(input_trial_array=self.id_val_dataset.input_trial_array, 
                                          index2path=self.id_val_dataset._index2path)
        self.id_val_scorer.disable_compute()
        self.val_scorer = CosineScorer(input_trial_array=self.val_dataset.input_trial_array, 
                                          index2path=self.val_dataset._index2path)
        self.val_scorer.disable_compute()
        self.test_scorer = CosineScorer(input_trial_array=self.test_dataset.input_trial_array, 
                                        index2path=self.test_dataset._index2path)
        self.test_scorer.disable_compute()

    def setup_val_dataloaders(self, 
                              val_dataset, 
                              val_dataloader, 
                              id_val_dataset, 
                              id_val_dataloader):

        self.val_dataset = val_dataset
        self.id_val_dataset = id_val_dataset
        self._val_dls = [id_val_dataloader, val_dataloader]
        self.id_val_scorer = CosineScorer(input_trial_array=self.id_val_dataset.input_trial_array, 
                                          index2path=self.id_val_dataset._index2path)
        self.id_val_scorer.disable_compute()
        self.val_scorer = CosineScorer(input_trial_array=self.val_dataset.input_trial_array, 
                                          index2path=self.val_dataset._index2path)
        self.val_scorer.disable_compute()


    def test_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            self.id_val_scorer.enable_compute()
            self.val_scorer.enable_compute()
            self.test_scorer.enable_compute()

            id_val_y_pred, id_val_y_true = self.id_val_scorer.compute()
            val_y_pred, val_y_true = self.val_scorer.compute()
            test_y_pred, test_y_true = self.test_scorer.compute()
            
            self.id_val_scorer.disable_compute()
            self.val_scorer.disable_compute()
            self.test_scorer.disable_compute()

            self.id_val_scorer.reset()
            self.val_scorer.reset()
            self.test_scorer.reset()

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

            (test_eer_results, test_eer_results_str, 
             test_dcf_results, test_dcf_results_str) = self.test_dataset.eval(
                test_y_pred, 
                test_y_true, 
                self.test_dataset.trial_metadata_array
            )

            log_text = "\n"+"-"*50
            log_text += "\nValidation EER (OOD)\n{}\nValidation DCF (OOD)\n{}".format(val_eer_results_str, val_dcf_results_str)
            log_text += "\nValidation EER (ID) \n{}\nValidation DCF (ID)\n{}".format(id_val_eer_results_str, id_val_dcf_results_str)
            log_text += "\nTest EER \n{}\nTest DCF \n{}".format(test_eer_results_str, test_dcf_results_str)
            log_text += "\n"+"-"*50
            sr_logger.info(log_text)


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
            
            log_text = "\n"+"-"*50
            log_text += "\nValidation results at {} step".format(self.trainer.global_step)
            log_text += "\nValidation EER (OOD)\n{}\nValidation DCF (OOD)\n{}".format(val_eer_results_str, val_dcf_results_str)
            log_text += "\nValidation EER (ID) \n{}\nValidation DCF (ID)\n{}".format(id_val_eer_results_str, id_val_dcf_results_str)
            log_text += "\n"+"-"*50
            sr_logger.info(log_text)


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

        
    
