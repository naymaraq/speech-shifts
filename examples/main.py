import argparse
import yaml

from examples.augmentations.supported import supported as supported_perturbations
from examples.audio_processing.spec_augment.supported import supported as supported_spec_perturbations
from examples.audio_processing.supported import supported as supported_preprocessors
from examples.losses.supported import supported as supported_losses
from examples.models.supported import supported_featurizers, supported_classifiers

from examples.audio_processing.spec_augment.compose import Compose
from speech_shifts.common.audio.audio_augmentor import AudioAugmentor

def read_yaml(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def build_spec_augmentor(spec_augmentations_config):
    perturbations = []
    if spec_augmentations_config:
        for aug_type, aug_config in spec_augmentations_config.items():
            if aug_type not in supported_spec_perturbations:
                print("\N{cross mark} Spectogram augmentation of type {} does not exist.".format(aug_type))
            else:
                if aug_config["do_augment"]:
                    aug_class = supported_spec_perturbations[aug_type]
                    aug_obj = aug_class(**aug_config["params"])
                    p = aug_config["p"]
                    perturbations.append((p, aug_obj))
                    print("\N{heavy check mark}  Spectogram augmentation of type {} is turned ON.".format(aug_type))
                else:
                    print("\N{cross mark} Spectogram augmentation of type {} is turned OFF.".format(aug_type))
    composed = Compose(perturbations)
    return composed
    
def build_augmentor(augmentations_config):
    perturbations = []
    if augmentations_config:
        for aug_type, aug_config in augmentations_config.items():
            if aug_type not in supported_perturbations:
                print(" \N{cross mark}Audio augmentation type of {} does not exist.".format(aug_type))
            else:
                if aug_config["do_augment"]:
                    aug_class = supported_perturbations[aug_type]
                    aug_obj = aug_class(**aug_config["params"])
                    p = aug_config["p"]
                    perturbations.append((p, aug_obj))
                    print("\N{heavy check mark}  Audio augmentation of type {} is turned ON.".format(aug_type))
                else:
                    print("\N{cross mark} Audio augmentation type of {} is turned OFF.".format(aug_type))
    augmentor = AudioAugmentor(perturbations=perturbations, mutex_perturbations=None)
    return augmentor

def build_preprocessor(preprocessor_config, eval=False):
    if preprocessor_config is None:
        raise ValueError("Preprocessor is missing")
    else:
        spec_type = preprocessor_config["type"]
        spec_params = preprocessor_config["params"]
        spec_class = supported_preprocessors[spec_type]
        if eval:
            spec_params["dither"] = 0.0
        preprocessor = spec_class(**spec_params)
        print("\N{heavy check mark} Preprocessor of type {} is CREATED.".format(type(preprocessor).__name__))
        return preprocessor

def build_loss(loss_config):
    if loss_config is None:
        raise ValueError("Loss is missing")
    else:
        loss_type = loss_config["type"]
        loss_params = loss_config["params"]
        loss_class = supported_losses[loss_type]
        loss = loss_class(**loss_params)
        print("\N{heavy check mark} Loss function of type {} is CREATED.".format(type(loss).__name__))
        return loss

def build_model(model_config):
    if model_config is None:
        raise ValueError("Model is missing")
    else:
        f_type = model_config["featurizer_type"]
        c_type = model_config.get("classifier_type", None)
        params = model_config["params"]
        cfg = read_yaml(params["cfg_path"])
        featurizer = supported_featurizers[f_type](**cfg["featurizer"])
        print("\N{heavy check mark} Featurizer of type {} is CREATED.".format(type(featurizer).__name__))

        classifier = None
        if c_type is not None:
            classifier_params = cfg["classifier"]
            classifier_params.update(params)
            classifier = supported_classifiers[c_type](**classifier_params)
            print("\N{heavy check mark} Classifier of type {} is CREATED.".format(type(classifier).__name__))
        return featurizer, classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training speech_shifts models')
    parser.add_argument('--cfg', type=str, help='Config Path', required=False)
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    augmentor = build_augmentor(read_yaml(cfg.get("audio_augmentations", None)))
    spec_augmentor = build_spec_augmentor(read_yaml(cfg.get("spec_augmentations", None)))
    preprocessor = build_preprocessor(cfg.get("preprocessor", None))
    loss = build_loss(cfg.get("loss", None))
    featurizer, classifier = build_model(cfg.get("model", None))


