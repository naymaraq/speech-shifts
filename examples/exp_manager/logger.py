import yaml
import logging
import logging.config

from examples.exp_manager.logging_conf import conf


def get_logger(logger_name):
    config = yaml.safe_load(conf)
    logging.config.dictConfig(config)
    logger = logging.getLogger(logger_name)
    return logger

sr_logger = get_logger("speech-shifts")
