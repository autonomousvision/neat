import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ Decoder for occupancy and offset prediction.
    Args:
        dim (int): input dimension
        num_class (int): occupancy output dimension
        input_size (int): transformer feature dimension
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of ResNet blocks
        attention_iters (int): number of feature update iterations
        anchors (int): number of features per image
        seq_len (int): number of input images per viewpoint
    """

    def __init__(self, dim, num_class, input_size, hidden_size, n_blocks, 
                        attention_iters, n_cam, anchors, seq_len):
        super().__init__()
        self.n_blocks = n_blocks
        self.attention_iters = attention_iters

        # point projection
        self.fc_p = nn.Conv1d(dim, hidden_size, 1, bias=True)

        # attention field
        self.blocks_att = nn.ModuleList([
            ResnetBlockFC(input_size, hidden_size) for i in range(n_blocks)
        ])
        self.bn_att = CBatchNorm1d(input_size, hidden_size)
        self.fc_att = nn.Conv1d(hidden_size, seq_len * anchors * n_cam, 1)

        # occupancy and offset
        self.blocks = nn.ModuleList([
            ResnetBlockFC(input_size, hidden_size) for i in range(n_blocks)
        ])
        self.bn = CBatchNorm1d(input_size, hidden_size)
        self.fc_occ = nn.Conv1d(hidden_size, num_class, 1)
        self.fc_off = nn.Conv1d(hidden_size, 2, 1)
        
    def forward(self, p, c):
        
        # process points
        points = self.fc_p(p.transpose(-1,-2))

        # initial feature (equal attention to all locations)
        c_iter = c.mean(1).unsqueeze(-1).repeat(1,1,points.size(-1)) # (B, C, P)

        occ = []
        off = []
        attn = []
        if self.attention_iters > 0:
            for i in range(self.attention_iters):
                net_att = points
                for n in range(self.n_blocks):
                    net_att = self.blocks_att[n](net_att, c_iter)

                net_att = F.relu(self.bn_att(net_att, c_iter))

                # use a softmax to re-weight features
                att = self.fc_att(net_att)
                weights = F.softmax(att, dim=1).unsqueeze(-2) # (B, V * an * T, 1, P)
                c_iter = c.view(weights.size(0), -1, c.size(-1), 1) # (B, V * an * T, C, 1)
                c_iter = (weights*c_iter).sum(1) # (B, C, P)
                
                net = points
                for n in range(self.n_blocks):
                    net = self.blocks[n](net, c_iter)

                net = F.relu(self.bn(net, c_iter))

                # output predictions
                occ.append(self.fc_occ(net).squeeze(1))
                off.append(self.fc_off(net).squeeze(1))           
                attn.append(weights)
        else:
            net = points
            for n in range(self.n_blocks):
                net = self.blocks[n](net, c_iter)

            net = F.relu(self.bn(net, c_iter))

            # output predictions
            occ.append(self.fc_occ(net).squeeze(1))
            off.append(self.fc_off(net).squeeze(1))

        return occ, off, attn
      

class ResnetBlockFC(nn.Module):
    ''' Resnet building block of decoder.
    Args:
        c_dim (int): dimension of latent conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, c_dim, size_in, size_out=None, size_h=None):
        super().__init__()
        # attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        
        # submodules
        self.bn_0 = CBatchNorm1d(c_dim, size_in)
        self.bn_1 = CBatchNorm1d(c_dim, size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1, bias=False)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1, bias=False)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, bias=False)
            
        # initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        # submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        self.bn = nn.BatchNorm1d(f_dim, affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # unsqueeze if using global feature
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # batch norm
        net = self.bn(x)
        out = gamma * net + beta

        return out