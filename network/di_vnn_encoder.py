import torch
import torch.nn as nn
from utils.pt_util import SharedMLP
from network.vnn import MY_VNN_SimplePointnet,VNN_ResnetPointnet


class Model(nn.Module):
    def __init__(self, bn, latent_size, per_point_feat, mode='cnp'):
        super().__init__()
        assert mode in ['train', 'cnp']
        self.mode = mode
        self.vnn = MY_VNN_SimplePointnet(c_dim=latent_size, hidden_dim=per_point_feat[2],meta_output='equivariant_latent')
    def forward(self, x):
        if self.mode == 'train':
            # B x N x 3 -> B x latent_size
            #x = x.transpose(-1, -2)
            r = self.vnn(x[:,:,:])
            return r.reshape((r.shape[0],-1))
        elif self.mode == 'cnp':
            x = x.unsqueeze(1) #input is B,1,3
            r = self.vnn(x[:,:,:])         # (B, L)
            return r
        else:
            raise NotImplementedError
