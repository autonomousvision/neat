import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset


class CARLA_Data(Dataset):
    """
    Dataset class for images, point clouds and vehicle control in CARLA.
    """
    def __init__(self, root, pred_len, class_converter, ignore_sides, ignore_rear, seq_len, input_scale, input_crop):
        
        self.seq_len = seq_len 
        self.pred_len = pred_len
        self.ignore_sides = ignore_sides
        self.ignore_rear = ignore_rear

        self.converter = class_converter 

        self.input_crop = input_crop #0.64
        self.input_scale = input_scale #1
        
        self.front = []
        if not self.ignore_sides:
            self.left = []
            self.right = []
        if not self.ignore_rear:
            self.rear = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []

        for sub_root in root:
            preload_file = os.path.join(sub_root, 'vis_abs_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                if not self.ignore_sides:
                    preload_left = []
                    preload_right = []
                if not self.ignore_rear:
                    preload_rear = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []

                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    file_list = os.listdir(route_dir+"/seg_front/")
                    file_list.sort()
                    measurement_file_list = os.listdir(route_dir+"/measurements/")
                    measurement_file_list.sort()

                    _usable_images = len(file_list) - pred_len - 1

                    num_seq = _usable_images // self.seq_len 
                    for seq in range(num_seq):
                        fronts = []
                        if not self.ignore_sides:
                            lefts = []
                            rights = []
                        if not self.ignore_rear:
                            rears = []
                        xs = []
                        ys = []
                        thetas = []

                        for i in range(self.seq_len):
                            
                            # segmentation images
                            filename = file_list[seq*self.seq_len+i].split(".")[0] 
                            fronts.append(route_dir+"/seg_front/"+filename+".png")
                            if not self.ignore_sides:
                                lefts.append(route_dir+"/seg_left/"+filename+".png")
                                rights.append(route_dir+"/seg_right/"+filename+".png")
                            if not self.ignore_rear:
                                rears.append(route_dir+"/seg_rear/"+filename+".png")
                            
                            # position
                            with open(route_dir + f"/measurements/{filename}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                            

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])

                        # read files sequentially (future frames)
                        curr_file_index = int(filename)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            #print("measurements: ", f"/measurements/{str(curr_file_index+i).zfill(4)}.json")
                            # position
                            
                            with open(route_dir + f"/measurements/{str(curr_file_index+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        if not self.ignore_sides:
                            preload_left.append(lefts)
                            preload_right.append(rights)
                        if not self.ignore_rear:
                            preload_rear.append(rears)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                if not self.ignore_sides:
                    preload_dict['left'] = preload_left
                    preload_dict['right'] = preload_right
                if not self.ignore_rear:
                    preload_dict['rear'] = preload_rear
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            if not self.ignore_sides:
                self.left += preload_dict.item()['left']
                self.right += preload_dict.item()['right']
            if not self.ignore_rear:
                self.rear += preload_dict.item()['rear']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        if not self.ignore_sides:
            data['lefts'] = []
            data['rights'] = []
        if not self.ignore_rear:
            data['rears'] = []

        seq_fronts = self.front[index]
        if not self.ignore_sides:
            seq_lefts = self.left[index]
            seq_rights = self.right[index]
        if not self.ignore_rear:
            seq_rears = self.rear[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        for i in range(self.seq_len):
            data['fronts'].append(seg_to_one_hot(np.array(
                scale_and_crop_image(seq_fronts[i], self.input_scale, self.input_crop)), self.converter))
            if not self.ignore_sides:
                data['lefts'].append(seg_to_one_hot(np.array(
                    scale_and_crop_image(seq_lefts[i], self.input_scale, self.input_crop)), self.converter))
                data['rights'].append(seg_to_one_hot(np.array(
                    scale_and_crop_image(seq_rights[i], self.input_scale, self.input_crop)), self.converter))
            if not self.ignore_rear:
                data['rears'].append(seg_to_one_hot(np.array(
                    scale_and_crop_image(seq_rears[i], self.input_scale, self.input_crop)), self.converter))
        
            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]


        # lidar and waypoint processing to local coordinates
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
        
        return data

def scale_and_crop_image(filename, scale, crop_width_factor):
    """
    Load, scale and crop an Image.
    """
    image = Image.open(filename)
    (width, height) = (image.width // scale, image.height // scale)
    im_resized = image.resize((width, height), resample=Image.NEAREST)
    image = np.asarray(im_resized)
    crop = int(crop_width_factor * width)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]

    return cropped_image


def seg_to_one_hot(y, converter):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    n_dims = len(np.unique(converter))
    y_tensor = torch.Tensor(converter[y])
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    y_one_hot = np.transpose(y_one_hot, (2,0,1))

    return y_one_hot


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