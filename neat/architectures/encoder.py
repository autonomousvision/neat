import math

import torch 
from torch import nn
import torch.nn.functional as F

from torchvision import models


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class Encoder(nn.Module):
    """
    Full transformer encoder model for multi-sensor inputs.
    """

    def __init__(self, n_embd, n_head, block_exp, n_layer, n_cam,
                    vert_anchors, horz_anchors, seq_len, 
                    embd_pdrop, attn_pdrop, resid_pdrop):
        super().__init__()
        self.n_embd = n_embd

        # input image embedding
        self.img_emb = ImageCNN(vert_anchors, horz_anchors, normalize=True)

        # positional embedding parameter (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_cam * seq_len * vert_anchors * horz_anchors, n_embd))
        
        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, 
                        block_exp, attn_pdrop, resid_pdrop)
                        for layer in range(n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs, velocity):
        batch_size = velocity.size(0)
        viewpoints = len(inputs)
        resolution = inputs[0].size(-1)

        inputs = torch.stack(inputs, dim=1).view(-1,3,resolution,resolution)

        # forward the image model for token embeddings
        image_embeddings = self.img_emb(inputs).view(batch_size, viewpoints, self.n_embd, -1) # (B, V, C, an * T)
        image_embeddings = image_embeddings.transpose(2,3).reshape(batch_size, -1, self.n_embd) # (B, V * an * T, C)

        # project velocity to n_embd
        velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + image_embeddings + velocity_embeddings.unsqueeze(1)) # (B, V * an * T, C)
        x = self.blocks(x) # (B, V * an * T, C)
        x = self.ln_f(x) # (B, V * an * T, C)

        return x


class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        vert_anchors (int): number of vertical points in final average pooling
        horz_anchors (int): number of horizontal points in final average pooling
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, vert_anchors, horz_anchors, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))
        self.features = models.resnet34(pretrained=True)

    def forward(self, x):

        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)
        x = self.features.layer2(x)
        x = self.features.layer3(x)
        x = self.features.layer4(x)

        x = self.avgpool(x)

        return x


def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x