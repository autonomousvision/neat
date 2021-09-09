import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

from .controller import PIDController
from .decoder import *
from .encoder import *


class MultiTaskImageNetwork(nn.Module):
    ''' AIM with 2d semantics and depth
    Args:
        controller (nn.Module): controller network
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, config, device, **kwargs):
        super(MultiTaskImageNetwork, self).__init__()
        self.pred_len = config.pred_len
        self.config = config
        self.device = device

        # PID controller (alternative to learned controller)
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.image_encoder = ImageCNN(512, normalize=True, use_linear=False, 
                                    model_type=self.config.image_encoder_type).to(self.device)
        # self.velocity_encoder = nn.Sequential(
        #                     nn.Linear(1, 256),
        #                     nn.ReLU(inplace=True),
        #                     nn.Linear(256, 512),
        #                     nn.ReLU(inplace=True),
        #                 ).to(self.device)
        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        ).to(self.device)
        self.decoder = nn.GRUCell(input_size=4, hidden_size=64).to(self.device)
        self.output = nn.Linear(64, 2).to(self.device)

        self.seg_decoder = SegDecoder(config, 512).to(self.device)
        self.depth_decoder = DepthDecoder(config, 512).to(self.device)

    def forward(self, feature_emb, target_point):
        feature_emb = sum(feature_emb)
        z = self.join(feature_emb)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(self.device)

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            # x_in = x + target_point
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

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata