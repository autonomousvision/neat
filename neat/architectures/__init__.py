import numpy as np
import torch 
from torch import nn

from .controller import PIDController
from .decoder import Decoder
from .encoder import Encoder


class AttentionField(nn.Module):
    ''' Occupancy and offset prediction with a recurrent implicit function.
    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, config, device):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.tot_len = config.tot_len

        self.axis = config.axis
        self.offset = config.offset
        self.resolution = config.resolution
        self.max_throttle = config.max_throttle

        self.aim_dist = config.aim_dist
        self.angle_thresh = config.angle_thresh
        self.dist_thresh = config.dist_thresh
        self.red_light_mult = config.red_light_mult
        self.brake_speed = config.brake_speed
        self.brake_ratio = config.brake_ratio
        self.clip_delta = config.clip_delta

        self.device = device

        # PID controller (alternative to learned controller)
        self.turn_controller = PIDController(
                    config.turn_KP, config.turn_KI, config.turn_KD, config.turn_n)
        self.speed_controller = PIDController(
                    config.speed_KP, config.speed_KI, config.speed_KD, config.speed_n)
        
        # learnable modules
        self.encoder = Encoder(n_embd=config.n_embd,
                            n_head=config.n_head, 
                            block_exp=config.block_exp, 
                            n_layer=config.n_layer,
                            n_cam=config.num_camera, 
                            vert_anchors=config.vert_anchors, 
                            horz_anchors=config.horz_anchors, 
                            seq_len=config.seq_len, 
                            embd_pdrop=config.embd_pdrop, 
                            attn_pdrop=config.attn_pdrop, 
                            resid_pdrop=config.resid_pdrop).to(self.device)

        self.decoder = Decoder(dim=5, num_class=config.num_class, 
                            input_size=config.n_embd,
                            hidden_size=config.onet_hidden_size,
                            n_blocks=config.onet_blocks,
                            attention_iters=config.attention_iters,
                            n_cam=config.num_camera, 
                            anchors=config.anchors,
                            seq_len=config.seq_len).to(self.device)  
    
    def decode(self, p, t, c, **kwargs):
        ''' Returns occupancies and offsets for the sampled points.
        Args:
            p (tensor): points
            t (tensor): target point
            c (tensor): latent conditioned code c
        '''
        # concatenate query point (x, y, t) and target point (x_command, y_command)
        num_point = p.size(1)
        t = t.transpose(0,1).unsqueeze(1).repeat(1,num_point,1) # (B, P, 2)

        # divide target to match resolution used for reconstruction
        t = t / self.resolution

        # focus on front by offset along y
        t[:,:,1] += self.offset
        
        p = torch.cat((p,t), dim=-1) # (B, P, 5)
        
        occ, off, attn = self.decoder(p, c, **kwargs)
        return occ, off, attn

    def create_plan_grid(self, scale, res, batch_size):
        # create uniform sampling grid
        linspace_x = torch.linspace(-scale * self.axis/2, scale * self.axis/2, steps=res)
        linspace_y = self.offset - torch.linspace(0, scale * self.axis, steps=res)
        linspace_t = torch.linspace(0, self.tot_len-1, steps=(self.tot_len))
        grid_x, grid_y, grid_t = torch.meshgrid(linspace_x, linspace_y, linspace_t)
        grid_points = torch.stack((grid_x, grid_y, grid_t), dim=3).unsqueeze(0).repeat(batch_size,1,1,1,1)
        grid_points = grid_points.reshape(batch_size,-1,3).to(self.device, dtype=torch.float32)

        return grid_points

    def create_light_grid(self, x_steps, y_steps, batch_size):
        # create second sampling grid for traffic light state
        linspace_x = torch.linspace(0, self.axis/2, steps=x_steps) # currently hard-coded
        linspace_y = self.offset - torch.linspace(0, self.axis, steps=y_steps) # currently hard-coded
        linspace_t = torch.linspace(0, 1, steps=1)
        grid_x, grid_y, grid_t = torch.meshgrid(linspace_x, linspace_y, linspace_t)
        grid_points = torch.stack((grid_x, grid_y, grid_t), dim=3).unsqueeze(0).repeat(batch_size,1,1,1,1)
        grid_points = grid_points.reshape(batch_size,-1,3).to(self.device, dtype=torch.float32)

        return grid_points

    def plan(self, t, c, plan_grid, light_grid, res, passes, **kwargs):
        ''' Returns waypoints for driving.
        Args:
            t (tensor): target point
            c (tensor): latent conditioned code c
        '''
        batch_size = c.size(0)

        # loop over grid for required number of passes
        for p in range(passes):
            occ, off, _ = self.decode(plan_grid, t, c)
            plan_grid[:,:,:2] += off[-1].transpose(1,2)

        plan_grid = plan_grid.reshape(batch_size,res*res,-1,3) # (B, P, T, 3)

        # convert to meters by scaling
        grid_mean = plan_grid[:,:,:,:2].mean(1) * self.resolution # (B, T, 2)

        # undo offset used to focus on front of vehicle
        grid_mean[:,:,1] -= self.offset * self.resolution

        if self.red_light_mult < 1.0:
            occ, off, _ = self.decode(light_grid, t, c)
            red_light_occ = (torch.argmax(occ[-1], dim=1)==3).sum()
        else:
            red_light_occ = 0

        return grid_mean, red_light_occ
    
    def control_pid(self, waypoints, velocity, target, red_light):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        target = target.squeeze().data.cpu().numpy()
        if torch.is_tensor(red_light):
            red_light = red_light.data.cpu().numpy()


        # flip y (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        target[1] *= -1

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        # slow if red light affordance is active
        if red_light:
            desired_speed *= self.red_light_mult

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.angle_thresh and target[1] < self.dist_thresh)
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        speed = velocity[0].data.cpu().numpy()
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata