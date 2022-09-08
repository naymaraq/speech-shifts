import torch.nn as nn

class BaseFeaturizer(nn.Module):

    def features(self, x):
        raise NotImplementedError

    def emb_dim(self):
        raise NotImplementedError
    
    def get_feat_in(self):
        raise NotImplementedError
    
    def forward(self, audio_signal, length):
        pool, emb = self.features(audio_signal, length)
        return pool, emb
