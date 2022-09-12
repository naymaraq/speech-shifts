import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class GE2ELoss(nn.Module):
    
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(GE2ELoss, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        """
        assume x has the following structure
        x.shape -> (batch_size, num_views, emb_dim)
        at least 2 num_views are needed (num_views >= 2)
        """
        assert x.size()[1] >= 2

        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0,gsize): 
            idx = [*range(0,gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:,idx,:], 1)
            cos_sim_diag    = F.cosine_similarity(x[:,ii,:],exc_centroids)
            cos_sim         = F.cosine_similarity(x[:,ii,:].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0,2))
            cos_sim[range(0,stepsize),range(0,stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim,1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix,dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label = torch.from_numpy(numpy.asarray(range(0, stepsize))).to(x.device)
        nloss = self.criterion(cos_sim_matrix.view(-1,stepsize), torch.repeat_interleave(label,repeats=gsize,dim=0).to(x.device))

        return nloss
