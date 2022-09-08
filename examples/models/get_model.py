import yaml
from examples.models.conv_asr_featurizer import ConvASREncoder

def get_featurizer(model, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if model in ["titanet", "speakernet"]:
        featurizer = ConvASREncoder(**cfg["featurizer"])
    
    return featurizer


# if __name__ == "__main__":
#     t_large = "models/config/titanet_large.yaml"
#     t_med = "models/config/titanet_medium.yaml"
#     s_small = "models/config/speakernet_3x2x256.yaml"
#     s_med = "models/config/speakernet_3x2x512.yaml"
#     f = get_featurizer("titanet", t_large)
#     f = get_featurizer("titanet", t_med)
#     f = get_featurizer("speakernet", s_small)
#     f = get_featurizer("speakernet", s_med)


    
