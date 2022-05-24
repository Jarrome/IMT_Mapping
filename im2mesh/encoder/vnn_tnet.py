import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC
from im2mesh.layers_equi import *


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNN_TNet(nn.Module):
    def __init__(self, k=20):
        super().__init__()
        self.k=k
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, negative_slope=0.0, share_nonlinearity=False)
        
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, use_batchnorm=True, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, use_batchnorm=True, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, use_batchnorm=True, negative_slope=0.0)
        
        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, use_batchnorm=True, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, use_batchnorm=True, negative_slope=0.0)
        self.fc3 = VNLinear(256//3, 3)
        
        self.pool = meanpool
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).transpose(2, 3)
        feat = get_graph_feature_cross(x, k=self.k)
        x = self.conv_pos(feat)
        x = self.pool(x, dim=-1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x, dim=-1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x.transpose(1, 2)


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.tnet = VNN_TNet()
        
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = meanpool

    def forward(self, p):
        batch_size, T, D = p.size()
        
        trans = self.tnet(p)
        p = torch.bmm(p, trans)

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim
        
        self.tnet = VNN_TNet()

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()
        
        trans = self.tnet(p)
        p = torch.bmm(p, trans)

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c
