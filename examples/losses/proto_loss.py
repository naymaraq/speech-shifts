import torch
import torch.nn as nn
import numpy

class ProtoLoss(nn.Module):
    __loss_name__ = "ProtoLoss"

    def __init__(self, **kwargs):
        super(ProtoLoss, self).__init__()
        self.criterion  = torch.nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        """
        assume x has the following structure
        x.shape -> (batch_size, num_views, emb_dim)
        x[:,0,:] are query vectors
        x[:,1:,:] are support vectors
        """
        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (torch.cdist(out_positive, out_anchor, p=2)**2)
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).to(x.device)
        nloss   = self.criterion(output, label)

        return nloss