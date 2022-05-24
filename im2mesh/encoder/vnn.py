import torch
import torch.nn as nn
from im2mesh.layers_equi import *


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNN_DGCNN(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20):
        super(VNN_DGCNN, self).__init__()
        self.c_dim = c_dim
        self.k = k
        
        self.conv_0 = VNLinearLeakyReLU(2, hidden_dim)
        self.conv_1 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv_2 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        self.conv_3 = VNLinearLeakyReLU(hidden_dim*2, hidden_dim)
        
        self.conv_c = VNLinearLeakyReLU(hidden_dim*4, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1).transpose(2, 3)
        
        x = get_graph_mean(x, k=self.k)
        x_0 = self.conv_0(x)
        
        x = get_graph_mean(x_0, k=self.k)
        x_1 = self.conv_1(x)
        
        x = get_graph_mean(x_1, k=self.k)
        x_2 = self.conv_2(x)
        
        x = get_graph_mean(x_2, k=self.k)
        x_3 = self.conv_3(x)
        
        x = torch.cat((x_0, x_1, x_2, x_3), dim=1)
        x = self.conv_c(x)
        x = x.mean(dim=-1, keepdim=False)
        
        return x


class VNN_SimplePointnet(nn.Module):
    ''' DGCNN-based VNN encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        
        self.conv_pos = VNLinearLeakyReLU(3, 64, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(64, 2*hidden_dim)
        self.fc_0 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_1 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_2 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_3 = VNLinear(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)
        
        
        self.actvn_0 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_1 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_2 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_3 = VNLeakyReLU(2*hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        
    def forward(self, p):
        batch_size = p.size(0)
        '''
        p_trans = p.unsqueeze(1).transpose(2, 3)
        
        #net = get_graph_feature(p_trans, k=self.k)
        #net = self.conv_pos(net)
        #net = net.mean(dim=-1, keepdim=False)
        #net = torch.cat([net, p_trans], dim=1)
        
        net = p_trans
        aggr = p_trans.mean(dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, aggr], dim=1)
        '''
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.fc_0(self.actvn_0(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.fc_1(self.actvn_1(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_2(self.actvn_2(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.fc_3(self.actvn_3(net))
        
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std

        return c


class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool
        
        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)
        
        net = self.fc_pos(net)
        
        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)
        
        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        
        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c
