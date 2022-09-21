import argparse
import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from examples.exp_manager.exp_manager import exp_manager
from examples.exp_manager.logger import sr_logger
from examples.utils.factory import (
    build_augmentor,
    build_callbacks,
    build_loss, 
    build_model,
    build_preprocessor,
    build_spec_augmentor, 
    read_yaml
)
from examples.algorithms.erm import ERM
from examples.algorithms.dann import DANN
from speech_shifts.datasets.mlsr_dataset import MLSRDataset
from speech_shifts.common.get_loaders import (
    get_train_loader, 
    get_eval_loader
)


def setup_env_vars():
    os.environ['ECCL_DEBUG'] = 'INFO'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'

def build_trainer(cfg, cfg_path):
    logger_path, checkpoints_path = exp_manager(cfg_path)
    callbacks = build_callbacks(read_yaml(cfg.get("callbacks", None)), checkpoints_path)
    #logger = WandbLogger(save_dir=logger_path, project="Speech-Shifts", name='', version='')
    logger = TensorBoardLogger(save_dir=logger_path, name='', version='')

    trainer = Trainer(**cfg["trainer"],
                        logger=logger,
                        callbacks=callbacks,
    )
    explicit_checkpoint_path = cfg["explicit_checkpoint_path"]
    if explicit_checkpoint_path and os.path.exists(explicit_checkpoint_path):
        sr_logger.info(f"Loading pytorch checkpoint from {explicit_checkpoint_path}")
        loaded_checkpoint = torch.load(explicit_checkpoint_path)
        trainer.fit_loop.global_step = loaded_checkpoint['global_step']
        trainer.fit_loop.current_epoch = loaded_checkpoint['epoch']
    
    sr_logger.info("\N{heavy check mark}  Logger of type {} is CREATED.".format(type(logger).__name__))
    sr_logger.info("\N{heavy check mark}  Trainer of type {} is CREATED.".format(type(trainer).__name__))
    return trainer

def build_dataloaders(data_config, augmentor):
    if data_config is None:
        raise ValueError("Data is missing.")
    
    dataset = MLSRDataset(data_config["root_dir"])
    loader_kwargs = {"n_views": 1}
    val_dataset = dataset.get_subset("val", loader_kwargs=loader_kwargs, augmentor=None)
    id_val_dataset = dataset.get_subset("id_val", loader_kwargs=loader_kwargs, augmentor=None)
    test_dataset = dataset.get_subset("test", loader_kwargs=loader_kwargs, augmentor=None)

    params = data_config["params"]
    loader_kwargs = {"n_views": params["n_views"]}
    train_dataset = dataset.get_subset("train", loader_kwargs=loader_kwargs, augmentor=augmentor)

    eval_bs = params["eval_bs"]
    train_bs = params["train_bs"]
    num_workers = params["num_workers"]
    val_dataloader = get_eval_loader("standard", val_dataset, batch_size=eval_bs, num_workers=num_workers, pin_memory=True)
    #test_dataloader = get_eval_loader("standard", test_dataset, batch_size=eval_bs, num_workers=num_workers, pin_memory=True)
    id_val_dataloader = get_eval_loader("standard", id_val_dataset, batch_size=eval_bs, num_workers=num_workers, pin_memory=True)
    train_dataloader = get_train_loader("standard", train_dataset, batch_size=train_bs, num_workers=num_workers, pin_memory=True)

    sr_logger.info("\N{heavy check mark}  Train Dataset of type {} is CREATED.".format(type(train_dataset).__name__))
    sr_logger.info("\N{heavy check mark}  Val Dataset of type {} is CREATED.".format(type(val_dataset).__name__))
    sr_logger.info("\N{heavy check mark}  ID Val Dataset of type {} is CREATED.".format(type(id_val_dataset).__name__))

    return ((train_dataset, train_dataloader), 
            (val_dataset, val_dataloader),
            (id_val_dataset, id_val_dataloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training speech_shifts models')
    parser.add_argument('--cfg', type=str, help='Config Path', required=True)
    parser.add_argument('--algo', type=str, help='Algorithm Name', required=True)

    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    setup_env_vars()

    augmentor = build_augmentor(read_yaml(cfg.get("audio_augmentations", None)))
    ((train_dataset, train_dataloader), 
     (val_dataset, val_dataloader),
     (id_val_dataset, id_val_dataloader)) = build_dataloaders(cfg.get("data", None), augmentor)
    
    loss = build_loss(cfg.get("loss", None))
    num_classes = len(set(train_dataset.y_array.tolist()))
    num_domains = len(set(train_dataset.metadata_array[:,0].numpy().tolist()))
    featurizer, classifier, discriminator = build_model(cfg.get("model", None), num_classes, num_domains)

    spec_augmentor = build_spec_augmentor(read_yaml(cfg.get("spec_augmentations", None)))
    preprocessor = build_preprocessor(cfg.get("preprocessor", None))
    trainer = build_trainer(cfg, args.cfg)

    if args.algo == "erm":
        model = ERM(
            preprocessor=preprocessor,
            train_spec_augmentor=spec_augmentor,
            featurizer=featurizer,
            loss=loss,
            optim_config=cfg["optimizer"],
            classifier=classifier,
        )
    elif args.algo == "dann":
        model = DANN(
            preprocessor=preprocessor,
            train_spec_augmentor=spec_augmentor,
            featurizer=featurizer,
            discriminator=discriminator,
            loss=loss,
            optim_config=cfg["optimizer"],
            dann_params=cfg["dann_params"],
            classifier=classifier
        )
    
    model.setup_train_dataloaders(
        train_dataset, 
        train_dataloader
    )
    model.setup_val_dataloaders(
        val_dataset, 
        val_dataloader, 
        id_val_dataset, 
        id_val_dataloader
    )

    trainer.fit(model)


