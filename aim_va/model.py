import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

from torchvision import models

from collections import deque



class MultiTaskImageNetwork(nn.Module):
    ''' Image + LiDAR network with waypoint output and pid controller
    Args:
        controller (nn.Module): controller network
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, device, input_size, pred_len, num_cameras=1, **kwargs):
        super(MultiTaskImageNetwork, self).__init__()
        self.pred_len = pred_len
        self.device = device
        self.input_size = input_size
        self.num_cameras = num_cameras

        # PID controller (alternative to learned controller)
        self.turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.image_encoder = ImageCNN(512, self.input_size, use_linear=False).to(self.device)
        self.join = nn.Sequential(
                            nn.Linear(512*self.num_cameras, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)

        self.decoder = nn.GRUCell(input_size=2, hidden_size=64).to(self.device)
        
        self.output = nn.Linear(64, 2).to(self.device)

    def forward(self, feature_emb, target_point):
        feature_emb_new = torch.cat((feature_emb), 1)

        z = self.join(feature_emb_new)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(
                    waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < 0.4 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0
        return steer, throttle, brake

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, input_dim=3, use_linear=False, **kwargs):
        super().__init__()
        self.use_linear = use_linear
            
        self.features = models.resnet34(pretrained=True)

        self.features.fc = nn.Sequential()

        self.features.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')


    def forward(self, inputs):
        c = 0
        for x in inputs:
            net = self.features(x)
            c += self.fc(net)
        return c

    def load_state_dict(self, state_dict):
        errors = super().load_state_dict(state_dict, strict=False)
