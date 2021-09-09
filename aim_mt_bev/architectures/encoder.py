import math

import torch 
from torch import nn
import torch.nn.functional as F

from torchvision import models


class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=False, **kwargs):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.model_type = kwargs.get('model_type')

        if self.model_type == 'resnet18':
            self.features = models.resnet18(pretrained=True)
            self.features.fc = nn.Sequential()

        elif self.model_type == 'resnet34':
            self.features = models.resnet34(pretrained=True)
            self.features.fc = nn.Sequential()

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            net = self.features(x)
            c += self.fc(net)
        return c


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