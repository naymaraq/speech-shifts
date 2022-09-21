import os
from shutil import copyfile

from examples.exp_manager.exp_utils import (is_global_rank_zero, 
                get_version_folder, 
                read_yaml, 
                get_commit_hash)


def exp_manager(config_path):
    config = read_yaml(config_path)
    save_dir = config['experiment']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if is_global_rank_zero():
        version_folder = get_version_folder(save_dir)
        os.makedirs(version_folder)
    else:
        version_folder = get_version_folder(save_dir, new=False)

    logger_path = os.path.join(version_folder, 'logs')
    checkpoints_path = os.path.join(version_folder, 'checkpoints')
    configs_versioning_path = os.path.join(version_folder, 'configs')

    if is_global_rank_zero():
        os.makedirs(logger_path)
        os.makedirs(checkpoints_path)
        os.makedirs(configs_versioning_path)

    config_name = os.path.split(config_path)[-1]
    model_config_name = os.path.split(config['model']['cfg_path'])[-1]
    
    if config.get("audio_augmentations", None):
        aug_config_name = os.path.split(config['audio_augmentations'])[-1]
        copyfile(config['audio_augmentations'], os.path.join(configs_versioning_path, aug_config_name))

    if config.get("spec_augmentations", None):
        spec_aug_config_name = os.path.split(config['spec_augmentations'])[-1]
        copyfile(config['spec_augmentations'], os.path.join(configs_versioning_path, spec_aug_config_name))

    if config.get("callbacks", None):
        callbacks_config_name = os.path.split(config['callbacks'])[-1]
        copyfile(config['callbacks'], os.path.join(configs_versioning_path, callbacks_config_name))

    copyfile(config_path, os.path.join(configs_versioning_path, config_name))
    copyfile(config['model']['cfg_path'], os.path.join(configs_versioning_path, model_config_name))

    with open(os.path.join(configs_versioning_path, "meta-info.txt"), "w") as f:
        commit_hash = get_commit_hash()
        f.write(f"commit-hash: {commit_hash}\n")

    return logger_path, checkpoints_path