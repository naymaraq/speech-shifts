import yaml


from examples.exp_manager.logger import sr_logger
from examples.audio_processing.spec_augment.supported import supported_spec_perturbations
from examples.audio_processing.spec_augment.compose import Compose
from examples.augmentations.supported import supported_perturbations
from speech_shifts.common.audio.audio_augmentor import AudioAugmentor
from examples.audio_processing.supported import supported_preprocessors
from examples.losses.supported import supported_losses
from examples.callbacks.supported import supported_callbacks
from examples.models.supported import supported_featurizers, supported_classifiers

def read_yaml(cfg_path):
    if cfg_path:
        with open(cfg_path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cfg

def build_spec_augmentor(spec_augmentations_config):
    perturbations = []
    if spec_augmentations_config:
        for aug_type, aug_config in spec_augmentations_config.items():
            if aug_type not in supported_spec_perturbations:
                sr_logger.info("\N{cross mark} Spectogram augmentation of type {} does not exist.".format(aug_type))
            else:
                if aug_config["do_augment"]:
                    aug_class = supported_spec_perturbations[aug_type]
                    aug_obj = aug_class(**aug_config["params"])
                    p = aug_config["p"]
                    perturbations.append((p, aug_obj))
                    sr_logger.info("\N{heavy check mark}  Spectogram augmentation of type {:^20} is turned ON.".format(aug_type))
                else:
                    sr_logger.info("\N{cross mark} Spectogram augmentation of type {:^20} is turned OFF.".format(aug_type))
    composed = Compose(perturbations)
    return composed

def build_augmentor(augmentations_config):
    perturbations = []
    if augmentations_config:
        for aug_type, aug_config in augmentations_config.items():
            if aug_type not in supported_perturbations:
                sr_logger.info(" \N{cross mark}Audio augmentation type of {} does not exist.".format(aug_type))
            else:
                if aug_config["do_augment"]:
                    aug_class = supported_perturbations[aug_type]
                    aug_obj = aug_class(**aug_config["params"])
                    p = aug_config["p"]
                    perturbations.append((p, aug_obj))
                    sr_logger.info("\N{heavy check mark}  Audio augmentation of type {:^10} is turned ON with probability {}.".format(aug_type, p))
                else:
                    sr_logger.info("\N{cross mark} Audio augmentation type of {:^10} is turned OFF.".format(aug_type))
    
    mutex_perturbations = [supported_perturbations[a] for a in ["impulse", "rir", "noise"]]
    augmentor = AudioAugmentor(perturbations=perturbations, mutex_perturbations=mutex_perturbations)
    return augmentor

def build_preprocessor(preprocessor_config, eval=False):
    if preprocessor_config is None:
        raise ValueError("Preprocessor is missing.")
    else:
        spec_type = preprocessor_config["type"]
        spec_params = preprocessor_config["params"]
        spec_class = supported_preprocessors[spec_type]
        if eval:
            spec_params["dither"] = 0.0
        preprocessor = spec_class(**spec_params)
        sr_logger.info("\N{heavy check mark}  Preprocessor of type {} is CREATED.".format(type(preprocessor).__name__))
        return preprocessor

def build_loss(loss_config):
    if loss_config is None:
        raise ValueError("Loss is missing.")
    else:
        loss_type = loss_config["type"]
        loss_params = loss_config["params"]
        loss_class = supported_losses[loss_type]
        loss = loss_class(**loss_params)
        sr_logger.info("\N{heavy check mark}  Loss function of type {} is CREATED.".format(type(loss).__name__))
        return loss

def build_callbacks(callbacks_config, checkpoints_save_dir):
    callbacks = []
    if callbacks_config:
        for call_type, call_config in callbacks_config.items():
            if call_type not in supported_callbacks:
                sr_logger.info("Callback type {} does not exist.".format(call_type))
            else:
                if call_config["do_callback"]:
                    call_class = supported_callbacks[call_type]
                    if call_type == 'ModelCheckpoint':
                        call_obj = call_class(dirpath=checkpoints_save_dir, **call_config["params"])
                    else:
                        call_obj = call_class(**call_config["params"])
                    callbacks.append(call_obj)
                    sr_logger.info("\N{heavy check mark}  Callback of type {:^20} is CREATED.".format(type(call_obj).__name__))
                else:
                    sr_logger.info("\N{cross mark} Callback type of {:^20} is turned OFF.".format(call_type))

    return callbacks

def build_model(model_config, num_classess, num_domains):
    if model_config is None:
        raise ValueError("Model is missing.")
    else:
        f_type = model_config["featurizer_type"]
        classifier_p = model_config.get("classifier", None)
        discriminator_p = model_config.get("discriminator", None)

        cfg = read_yaml(model_config["cfg_path"])
        featurizer = supported_featurizers[f_type](**cfg["featurizer"])
        sr_logger.info("\N{heavy check mark}  Featurizer of type {} is CREATED.".format(type(featurizer).__name__))

        classifier = None
        if classifier_p is not None:
            c_type = classifier_p["classifier_type"]
            classifier_params = cfg["classifier"]
            classifier_params.update(classifier_p["params"])
            classifier_params["num_classes"] = num_classess
            classifier = supported_classifiers[c_type](**classifier_params)
            sr_logger.info("\N{heavy check mark}  Classifier (num_classes={}) of type {} is CREATED.".format(num_classess, type(classifier).__name__))
        
        discriminator = None
        if discriminator_p is not None:
            d_type = discriminator_p["discriminator_type"]
            discriminator_params = cfg["discriminator"]
            discriminator_params.update(discriminator_p["params"])
            discriminator_params["num_classes"] = num_domains
            discriminator = supported_classifiers[d_type](**discriminator_params)
            sr_logger.info("\N{heavy check mark}  Discriminator (num_domains={}) of type {} is CREATED.".format(num_domains, type(discriminator).__name__))

        return featurizer, classifier, discriminator