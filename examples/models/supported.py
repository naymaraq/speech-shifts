import yaml
from examples.models.conv_asr_featurizer import ConvASREncoder
from examples.models.ecapa_tdnn_featurizer import ECAPAEncoder
from examples.models.jasper_block.linear_classifier import SpeakerClassifier

supported_featurizers = {
    "speakernet_medium": ConvASREncoder,
    "speakernet_small": ConvASREncoder,
    "titanet_medium": ConvASREncoder,
    "titanet_large": ConvASREncoder,
    "ecapa_tdnn": ECAPAEncoder
}

supported_classifiers = {
    "linear": SpeakerClassifier
}
    


    
