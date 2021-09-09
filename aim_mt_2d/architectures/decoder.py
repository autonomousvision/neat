import torch 
from torch import nn
import torch.nn.functional as F


class SegDecoder(nn.Module):

    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.scale = 4
        self.latent_dim = latent_dim
        self.num_class = config.num_class

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, 128, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, self.num_class, 3, 1, 1),
                    )

    def forward(self, x):
        x = sum(x)
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear')
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.deconv3(x)

        return x


class DepthDecoder(nn.Module):

    def __init__(self, config, latent_dim=512):
        super().__init__()
        self.config = config
        self.scale = 4
        self.latent_dim = latent_dim

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.latent_dim, 128, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 1, 3, 1, 1),
                    )

    def forward(self, x):
        x = sum(x)
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear')
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.deconv3(x)
        x = torch.sigmoid(x).squeeze(1)

        return x