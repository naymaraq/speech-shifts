import yaml
from examples.models.conv_asr_featurizer import ConvASREncoder
from examples.models.ecapa_tdnn_featurizer import ECAPAEncoder

def get_featurizer(model, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if model in ["speakernet_medium", "speakernet_small", "titanet_large", "titanet_medium"]:
        featurizer = ConvASREncoder(**cfg["featurizer"])
    elif model == "ecapa_tdnn":
        featurizer = ECAPAEncoder(**cfg["featurizer"])

    return featurizer


def num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

if __name__ == "__main__":
    t_large = "models/config/titanet_large.yaml"
    t_med = "models/config/titanet_medium.yaml"
    s_small = "models/config/speakernet_3x2x256.yaml"
    s_med = "models/config/speakernet_3x2x512.yaml"
    ecapa_tdnn = "models/config/ecapa_tdnn.yaml"

    for name, cfg in zip(["titanet_large", "titanet_medium", 
                    "speakernet_small", "speakernet_medium", "ecapa_tdnn"],
                    [t_large, t_med, s_small, s_med, ecapa_tdnn]):
        f = get_featurizer(name, cfg)
        n_params = num_params(f)
        print(f"Model {name}: Params {round(n_params/10**6, 2)}M")
    


    
