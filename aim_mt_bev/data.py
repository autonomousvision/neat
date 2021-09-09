import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset


class CARLA_points(Dataset):
    """
    Dataset class for images, point clouds and vehicle control in CARLA.
    """
    def __init__(self, root, config):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.tot_len = config.tot_len
        self.scale = config.scale
        self.crop = config.crop
        self.scale_topdown = config.scale_topdown
        self.crop_topdown = config.crop_topdown
        self.num_class = config.num_class
        self.converter = np.uint8(config.converter)
        self.axis = config.axis
        self.resolution = config.resolution
        self.offset = config.offset

        # initialize preload lists
        self.front = []
        self.left = []
        self.right = []
        self.topdown = []
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
            preload_file = os.path.join(sub_root, 'pl_'+str(config.seq_len)+'_'+str(config.pred_len)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_topdown = []
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
                    num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-config.pred_len-1)//config.seq_len
                    for seq in range(num_seq):
                        fronts = []
                        lefts = []
                        rights = []
                        topdowns = []
                        xs = []
                        ys = []
                        thetas = []

                        # read files sequentially (past and current frames)
                        for i in range(config.seq_len):
                            
                            file_number = seq*config.seq_len+i+1

                            # images
                            filename = f"{str(file_number).zfill(4)}.png"
                            fronts.append(route_dir+"/rgb_front/"+filename)
                            lefts.append(route_dir+"/rgb_left/"+filename)
                            rights.append(route_dir+"/rgb_right/"+filename)

                            # semantics
                            topdowns.append(route_dir+"/topdown/"+filename)
                            
                            # position
                            with open(route_dir + f"/measurements/{str(file_number).zfill(4)}.json", "r") as read_file:
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
                        for i in range(config.seq_len, config.tot_len):

                            file_number = seq*config.seq_len+i+1

                            # semantics
                            topdowns.append(route_dir+f"/topdown/{str(file_number).zfill(4)}.png")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(file_number).zfill(4)}.json", "r") as read_file:
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
                        preload_topdown.append(topdowns)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['topdown'] = preload_topdown
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

            # load from stored npy
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.topdown += preload_dict.item()['topdown']
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
        data['topdowns'] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_topdowns = self.topdown[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        full_semantics = []
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_fronts[i], scale=self.scale, crop=self.crop))))
            data['lefts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_lefts[i], scale=self.scale, crop=self.crop))))
            data['rights'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_rights[i], scale=self.scale, crop=self.crop))))
            
            semantics_unprocessed = (torch.from_numpy(np.array(
                scale_and_crop_image(seq_topdowns[i],
                scale=self.scale_topdown, crop=self.crop_topdown))))
            data['topdowns'].append(self.converter[semantics_unprocessed])

            full_semantics.append(semantics_unprocessed)

            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.  

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # future frames
        for i in range(self.seq_len, self.tot_len):
            
            semantics_unprocessed = (torch.from_numpy(np.array(
                scale_and_crop_image(seq_topdowns[i],
                scale=self.scale_topdown, crop=self.crop_topdown))))
            data['topdowns'].append(self.converter[semantics_unprocessed])

            full_semantics.append(semantics_unprocessed)

            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.  

        # waypoint and semantic processing to local coordinates
        waypoints = []
        for i in range(self.tot_len):
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


def scale_and_crop_image(filename, scale, crop):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    image = Image.open(filename)
    (width, height) = (int(image.width / scale), int(image.height / scale))
    if scale != 1:
        im_resized = image.resize((width, height))
    else:
        im_resized = image

    image = np.asarray(im_resized)
    start_y = height//2 - crop//2
    start_x = width//2 - crop//2

    cropped_image = image[start_y:start_y+crop, start_x:start_x+crop]
    
    if len(cropped_image.shape)==2: # topdown semantic image
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

    