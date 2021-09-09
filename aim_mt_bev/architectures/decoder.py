import torch 
from torch import nn
import torch.nn.functional as F


class BroadcastDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.im_size = 32
        self.scale = 4
        self.num_class = config.num_class
        self.init_grid()

        self.g = nn.Sequential(
                    nn.Conv2d(512+2, 128, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.ReLU(True),
                    )

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(True),
                    )

        self.deconv2 = nn.Sequential(
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(32, self.num_class, 3, 1, 1),
                    )

    def init_grid(self):
        x = torch.linspace(-1, 1, self.im_size)
        y = torch.linspace(-1, 1, self.im_size)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
        
        
    def broadcast(self, z):
        b = z.size(0)
        x_grid = self.x_grid.expand(b, 1, -1, -1).to(z.device)
        y_grid = self.y_grid.expand(b, 1, -1, -1).to(z.device)
        z = z.view((b, -1, 1, 1)).expand(-1, -1, self.im_size, self.im_size)
        z = torch.cat((z, x_grid, y_grid), dim=1)
        return z

    def forward(self, z):
        z = sum(z)
        z = self.broadcast(z)
        x = self.g(z)
        x = F.upsample(x, scale_factor=self.scale)
        x = self.deconv1(x)
        x = F.upsample(x, scale_factor=2)
        x = self.deconv2(x)
        return x