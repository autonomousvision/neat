import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset


class CARLA_waypoint(Dataset):
    """
    Dataset class for images and vehicle control in CARLA.
    """
    def __init__(self, root, config, points_per_batch=1024):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.points_per_batch = points_per_batch
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.converter = np.uint8(config.converter)

        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.seg_front = []
        self.seg_left = []
        self.seg_right = []
        self.seg_rear = []
        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        self.depth_rear = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []

        for sub_root in root:
            preload_file = os.path.join(sub_root, 'aim_multitask_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_rear = []
                preload_seg_front = []
                preload_seg_left = []
                preload_seg_right = []
                preload_seg_rear = []
                preload_depth_front = []
                preload_depth_left = []
                preload_depth_right = []
                preload_depth_rear = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []

                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used
                    
                    num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
                    
                    for seq in range(num_seq):
                        fronts = []
                        lefts = []
                        rights = []
                        rears = []
                        seg_fronts = []
                        seg_lefts = []
                        seg_rights = []
                        seg_rears = []
                        depth_fronts = []
                        depth_lefts = []
                        depth_rights = []
                        depth_rears = []
                        xs = []
                        ys = []
                        thetas = []

                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq*self.seq_len+1+i).zfill(4)}.png"
                            fronts.append(route_dir+"/rgb_front/"+filename)
                            lefts.append(route_dir+"/rgb_left/"+filename)
                            rights.append(route_dir+"/rgb_right/"+filename)
                            rears.append(route_dir+"/rgb_rear/"+filename)
                            seg_fronts.append(route_dir+"/seg_front/"+filename)
                            seg_lefts.append(route_dir+"/seg_left/"+filename)
                            seg_rights.append(route_dir+"/seg_right/"+filename)
                            seg_rears.append(route_dir+"/seg_rear/"+filename)
                            depth_fronts.append(route_dir+"/depth_front/"+filename)
                            depth_lefts.append(route_dir+"/depth_left/"+filename)
                            depth_rights.append(route_dir+"/depth_right/"+filename)
                            depth_rears.append(route_dir+"/depth_rear/"+filename)
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        # read files sequentially (future frames)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_seg_front.append(seg_fronts)
                        preload_seg_left.append(seg_lefts)
                        preload_seg_right.append(seg_rights)
                        preload_seg_rear.append(seg_rears)
                        preload_depth_front.append(depth_fronts)
                        preload_depth_left.append(depth_lefts)
                        preload_depth_right.append(depth_rights)
                        preload_depth_rear.append(depth_rears)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['seg_front'] = preload_seg_front
                preload_dict['seg_left'] = preload_seg_left
                preload_dict['seg_right'] = preload_seg_right
                preload_dict['seg_rear'] = preload_seg_rear
                preload_dict['depth_front'] = preload_depth_front
                preload_dict['depth_left'] = preload_depth_left
                preload_dict['depth_right'] = preload_depth_right
                preload_dict['depth_rear'] = preload_depth_rear
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.seg_front += preload_dict.item()['seg_front']
            self.seg_left += preload_dict.item()['seg_left']
            self.seg_right += preload_dict.item()['seg_right']
            self.seg_rear += preload_dict.item()['seg_rear']
            self.depth_front += preload_dict.item()['depth_front']
            self.depth_left += preload_dict.item()['depth_left']
            self.depth_right += preload_dict.item()['depth_right']
            self.depth_rear += preload_dict.item()['depth_rear']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['rears'] = []
        data['seg_fronts'] = []
        data['seg_lefts'] = []
        data['seg_rights'] = []
        data['seg_rears'] = []
        data['depth_fronts'] = []
        data['depth_lefts'] = []
        data['depth_rights'] = []
        data['depth_rears'] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]
        seq_seg_fronts = self.seg_front[index]
        seq_seg_lefts = self.seg_left[index]
        seq_seg_rights = self.seg_right[index]
        seq_seg_rears = self.seg_rear[index]
        seq_depth_fronts = self.depth_front[index]
        seq_depth_lefts = self.depth_left[index]
        seq_depth_rights = self.depth_right[index]
        seq_depth_rears = self.depth_rear[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution))))
            
            data['seg_fronts'].append(torch.from_numpy(self.converter[scale_and_crop_seg(Image.open(seq_seg_fronts[i]), scale=self.scale, crop=self.input_resolution)]))
            if not self.ignore_sides:
                data['seg_lefts'].append(torch.from_numpy(self.converter[scale_and_crop_seg(Image.open(seq_seg_lefts[i]), scale=self.scale, crop=self.input_resolution)]))
                data['seg_rights'].append(torch.from_numpy(self.converter[scale_and_crop_seg(Image.open(seq_seg_rights[i]), scale=self.scale, crop=self.input_resolution)]))
            if not self.ignore_rear:
                data['seg_rears'].append(torch.from_numpy(self.converter[scale_and_crop_seg(Image.open(seq_seg_rears[i]), scale=self.scale, crop=self.input_resolution)]))

            data['depth_fronts'].append(torch.from_numpy(get_depth(scale_and_crop_image(Image.open(seq_depth_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['depth_lefts'].append(torch.from_numpy(get_depth(scale_and_crop_image(Image.open(seq_depth_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['depth_rights'].append(torch.from_numpy(get_depth(scale_and_crop_image(Image.open(seq_depth_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['depth_rears'].append(torch.from_numpy(get_depth(scale_and_crop_image(Image.open(seq_depth_rears[i]), scale=self.scale, crop=self.input_resolution))))
        
            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

        data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)

        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['command'] = self.command[index]
        data['velocity'] = self.velocity[index]
        
        return data


def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized

    return normalized


def scale_and_crop_seg(image, scale=1, crop=256):
    """
    Scale and crop a seg image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width / scale), int(image.height / scale))
    if scale != 1:
        im_resized = image.resize((width, height), resample=Image.NEAREST)
    else:
        im_resized = image

    im_resized = np.asarray(im_resized)
    start_y = height//2 - crop//2
    start_x = width//2 - crop//2

    cropped_image = im_resized[start_y:start_y+crop, start_x:start_x+crop]
    
    if len(cropped_image.shape)==2: # topdown semantic image
        cropped_image = cropped_image.reshape((crop,crop,1))

    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width / scale), int(image.height / scale))
    if scale != 1:
        im_resized = image.resize((width, height))
    else:
        im_resized = image

    image = np.asarray(im_resized)
    start_y = height//2 - crop//2
    start_x = width//2 - crop//2

    cropped_image = image[start_y:start_y+crop, start_x:start_x+crop]
    
    if len(cropped_image.shape)==2: # for seg, depth
        cropped_image = cropped_image.reshape((crop,crop,1))

    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out