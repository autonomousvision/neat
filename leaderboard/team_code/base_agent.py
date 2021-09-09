import time
import os
import datetime
import pathlib
import json

import cv2
import carla

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner

import numpy as np
from PIL import Image, ImageDraw

SAVE_PATH = os.environ.get('SAVE_PATH', None) # for saving episodes at rollout


class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self._sensor_data = {
            'width': 400,
            'height': 300,
            'fov': 100
        }

        self.weather_id = None

        self.save_path = None


        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print (string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            
            for sensor in self.sensors():
                if hasattr(sensor, 'save') and sensor['save']:
                    (self.save_path / sensor['id']).mkdir()

            (self.save_path / 'measurements').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'lidar').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'lidar_360').mkdir(parents=True, exist_ok=True)
            (self.save_path / 'topdown').mkdir(parents=True, exist_ok=True)

            for pos in ['front', 'left', 'right', 'rear']:
                for sensor_type in ['rgb', 'seg', 'depth']:
                    name = sensor_type + '_' + pos
                    (self.save_path / name).mkdir()

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True

        self._sensor_data['calibration'] = self._get_camera_to_car_calibration(self._sensor_data)
        
        self._sensors = self.sensor_interface._sensors_objects

    def _get_position(self, gps):
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        if SAVE_PATH is not None:
            return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'rgb_front'
                },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'seg_front'
                },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'depth_front'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'rgb_rear'
                },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': -1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'seg_rear'
                },
                {
                    'type': 'sensor.camera.depth',
                    'x': -1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'depth_rear'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'rgb_left'
                },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'seg_left'
                },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'depth_left'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'rgb_right'
                },
                {
                    'type': 'sensor.camera.semantic_segmentation',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'seg_right'
                },
                {
                    'type': 'sensor.camera.depth',
                    'x': 1.3, 'y': 0.0, 'z': 2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': self._sensor_data['width'], 'height': self._sensor_data['height'], 'fov': self._sensor_data['fov'],
                    'id': 'depth_right'
                },
                {
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0, 'rotation_frequency':10,
                    'id': 'lidar'
                },
                {
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0, 'rotation_frequency':20,
                    'id': 'lidar_360'
                },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                }
            ]
        else:
            return [
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                }
            ]

    def tick(self, input_data):
        weather = self._weather_to_dict(self._world.get_weather())

        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        traffic_lights = self._find_obstacle('*traffic_light*')
        stop_signs = self._find_obstacle('*stop*')

        depth = {}
        seg = {}
        for pos in ['front', 'left', 'right', 'rear']:
            seg_cam = 'seg_' + pos
            depth_cam = 'depth_' + pos
            _segmentation = np.copy(input_data[seg_cam][1][:, :, 2])

            depth[pos] = self._get_depth(input_data[depth_cam][1][:, :, :3])
            
            self._change_seg_tl(_segmentation, depth[pos], traffic_lights)
            self._change_seg_stop(_segmentation, depth[pos], stop_signs, seg_cam)
            
            seg[pos] = _segmentation

        rgb_front      = cv2.cvtColor(input_data['rgb_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear       = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left       = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right      = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        depth_front      = cv2.cvtColor(input_data['depth_front'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_left       = cv2.cvtColor(input_data['depth_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_right      = cv2.cvtColor(input_data['depth_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        depth_rear       = cv2.cvtColor(input_data['depth_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        return {
                'rgb_front': rgb_front,
                'seg_front': seg['front'],
                'depth_front': depth_front,
                'rgb_rear': rgb_rear,
                'seg_rear': seg['rear'],
                'depth_rear': depth_rear,
                'rgb_left': rgb_left,
                'seg_left': seg['left'],
                'depth_left': depth_left,
                'rgb_right': rgb_right,
                'seg_right': seg['right'],
                'depth_right': depth_right,
                'lidar' : input_data['lidar'][1],
                'lidar_360': input_data['lidar_360'][1],
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'weather': weather,
                }

    def _weather_to_dict(self, carla_weather):
        weather = {
            'cloudiness': carla_weather.cloudiness,
            'precipitation': carla_weather.precipitation,
            'precipitation_deposits': carla_weather.precipitation_deposits,
            'wind_intensity': carla_weather.wind_intensity,
            'sun_azimuth_angle': carla_weather.sun_azimuth_angle,
            'sun_altitude_angle': carla_weather.sun_altitude_angle,
            'fog_density': carla_weather.fog_density,
            'fog_distance': carla_weather.fog_distance,
            'wetness': carla_weather.wetness,
            'fog_falloff': carla_weather.fog_falloff,
        }

        return weather

    def _change_seg_stop(self, seg_img, depth_img, stop_signs, cam, _region_size=6):
        """Adds a stop class to the segmentation image

        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            stop_signs ([type]): [description]
            cam ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 6.
        """        
        for stop in stop_signs:

            _dist = self._get_distance(stop.get_transform().location)
                
            _region = np.abs(depth_img - _dist)

            seg_img[(_region < _region_size) & (seg_img == 12)] = 26

            # lane markings
            trigger = stop.trigger_volume

            _trig_loc_world = self._trig_to_world(np.array([[0], [0], [0], [1.0]]).T, stop, trigger)
            _x = self._world_to_sensor(_trig_loc_world, self._get_sensor_position(cam))[0,0]

            if _x > 0: # stop is in front of camera

                bb = self._create_2d_bb_points(trigger, 4)
                trig_loc_world = self._trig_to_world(bb, stop, trigger)
                cords_x_y_z = self._world_to_sensor(trig_loc_world, self._get_sensor_position(cam), True)

                #if cords_x_y_z.size: 
                cords_x_y_z = cords_x_y_z[:3, :]
                cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
                bbox = (self._sensor_data['calibration'] @ cords_y_minus_z_x).T

                camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)

                if np.any(camera_bbox[:,2] > 0):

                    camera_bbox = np.array(camera_bbox)

                    polygon = [(camera_bbox[i, 0], camera_bbox[i, 1]) for i in range(len(camera_bbox))]

                    img = Image.new('L', (self._sensor_data['width'], self._sensor_data['height']), 0)
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
                    _region = np.array(img)

                    seg_img[(_region == 1) & (seg_img == 6)] = 27


    def _trig_to_world(self, bb, parent, trigger):
        """Transforms the trigger coordinates to world coordinates

        Args:
            bb ([type]): [description]
            parent ([type]): [description]
            trigger ([type]): [description]

        Returns:
            [type]: [description]
        """        
        bb_transform = carla.Transform(trigger.location)
        bb_vehicle_matrix = self.get_matrix(bb_transform)
        vehicle_world_matrix = self.get_matrix(parent.get_transform())
        bb_world_matrix = vehicle_world_matrix @ bb_vehicle_matrix
        world_cords = bb_world_matrix @ bb.T
        return world_cords

    def _create_2d_bb_points(self, actor_bb, scale_factor=1):
        """
        Returns 2D floor bounding box for an actor.
        """

        cords = np.zeros((4, 4))
        extent = actor_bb.extent
        x = extent.x * scale_factor
        y = extent.y * scale_factor
        z = extent.z * scale_factor
        cords[0, :] = np.array([x, y, 0, 1])
        cords[1, :] = np.array([-x, y, 0, 1])
        cords[2, :] = np.array([-x, -y, 0, 1])
        cords[3, :] = np.array([x, -y, 0, 1])
        return cords

    def _get_sensor_position(self, cam):
        """returns the sensor position and rotation

        Args:
            cam ([type]): [description]

        Returns:
            [type]: [description]
        """        
        sensor_transform = self._sensors[cam].get_transform()
        
        return sensor_transform

    def _world_to_sensor(self, cords, sensor, move_cords=False):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = self.get_matrix(sensor)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)

        if move_cords:
            _num_cords = range(sensor_cords.shape[1])
            modified_cords = np.array([])
            for i in _num_cords:
                if sensor_cords[0,i] < 0:
                    for j in _num_cords:
                        if sensor_cords[0,j] > 0:
                            _direction = sensor_cords[:,i] - sensor_cords[:,j]
                            _distance = -sensor_cords[0,j] / _direction[0]
                            new_cord = sensor_cords[:,j] + _distance[0,0] * _direction * 0.9999
                            modified_cords = np.hstack([modified_cords, new_cord]) if modified_cords.size else new_cord
                else:
                    modified_cords = np.hstack([modified_cords, sensor_cords[:,i]]) if modified_cords.size else sensor_cords[:,i]

            return modified_cords
        else:
            return sensor_cords

    def get_matrix(self, transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def _change_seg_tl(self, seg_img, depth_img, traffic_lights, _region_size=4):
        """Adds 3 traffic light classes (green, yellow, red) to the segmentation image

        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            traffic_lights ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 4.
        """        
        for tl in traffic_lights:
            _dist = self._get_distance(tl.get_transform().location)
                
            _region = np.abs(depth_img - _dist)

            if tl.get_state() == carla.TrafficLightState.Red:
                state = 23
            elif tl.get_state() == carla.TrafficLightState.Yellow:
                state = 24
            elif tl.get_state() == carla.TrafficLightState.Green:
                state = 25
            else: #none of the states above, do not change class
                state = 18

            #seg_img[(_region >= _region_size)] = 0
            seg_img[(_region < _region_size) & (seg_img == 18)] = state

    def _get_dist(self, p1, p2):
        """Returns the distance between p1 and p2

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """        

        distance = np.sqrt(
                (p1[0] - p2[0]) ** 2 +
                (p1[1] - p2[1]) ** 2 +
                (p1[2] - p2[2]) ** 2)

        return distance

    def _get_distance(self, target):
        """Returns the distance from the (rgb_front) camera to the target

        Args:
            target ([type]): [description]

        Returns:
            [type]: [description]
        """        
        sensor_transform = self._sensors['rgb_front'].get_transform()

        distance = np.sqrt(
                (sensor_transform.location.x - target.x) ** 2 +
                (sensor_transform.location.y - target.y) ** 2 +
                (sensor_transform.location.z - target.z) ** 2)

        return distance

    def _get_depth(self, data):
        """Transforms the depth image into meters

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """        

        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
        normalized /=  (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized

        return in_meters

    def _find_obstacle(self, obstacle_type='*traffic_light*'):
        """Find all actors of a certain type that are close to the vehicle

        Args:
            obstacle_type (str, optional): [description]. Defaults to '*traffic_light*'.

        Returns:
            [type]: [description]
        """        
        obst = list()
        
        _actors = self._world.get_actors()
        _obstacles = _actors.filter(obstacle_type)


        for _obstacle in _obstacles:
            trigger = _obstacle.trigger_volume

            _obstacle.get_transform().transform(trigger.location)
            
            distance_to_car = trigger.location.distance(self._vehicle.get_location())

            a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
            b = np.sqrt(
                self._vehicle.bounding_box.extent.x ** 2 +
                self._vehicle.bounding_box.extent.y ** 2 +
                self._vehicle.bounding_box.extent.z ** 2)

            s = a + b + 10
           
            if distance_to_car <= s:
                # the actor is affected by this obstacle.
                obst.append(_obstacle)

                """self._debug.draw_box(carla.BoundingBox(_obstacle.get_transform().location, carla.Vector3D(0.5,0.5,2)),
                        _obstacle.get_transform().rotation, 
                        0.05, 
                        carla.Color(255,255,0,0),
                        0
                    )"""
                """self._debug.draw_box(carla.BoundingBox(trigger.location, carla.Vector3D(0.1,0.1,10)),
                        _obstacle.get_transform().rotation, 
                        0.05, 
                        carla.Color(255,0,0,0),
                        0
                    )"""
                
                """self._debug.draw_box(carla.BoundingBox(trigger.location, carla.Vector3D(0.1,0.1,2)),
                    _obstacle.get_transform().rotation, 
                    0.05, 
                    carla.Color(255,0,0,0),
                    0
                )"""
                """self._debug.draw_box(trigger,
                    _obstacle.get_transform().rotation, 
                    0.05, 
                    carla.Color(255,0,0,0),
                    0
                )"""

        return obst

    def _get_camera_to_car_calibration(self, sensor):
        """returns the calibration matrix for the given sensor

        Args:
            sensor ([type]): [description]

        Returns:
            [type]: [description]
        """        
        calibration = np.identity(3)
        calibration[0, 2] = sensor["width"] / 2.0
        calibration[1, 2] = sensor["height"] / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor["width"] / (2.0 * np.tan(sensor["fov"] * np.pi / 360.0))
        return calibration

    def save(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // 10

        pos = self._get_position(tick_data['gps'])
        theta = tick_data['compass']
        speed = tick_data['speed']

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
                'weather': self.weather_id,
                'junction':         self.junction,
                'vehicle_hazard':   self.vehicle_hazard,
                'light_hazard':     self.traffic_light_hazard,
                'walker_hazard':    self.walker_hazard,
                'stop_sign_hazard': self.stop_sign_hazard,
                'angle':            self.angle
                }

        for sensor in self.sensors():
            if 'camera' in sensor['type'] and 'map' not in sensor['id']:
                Image.fromarray(tick_data[sensor['id']]).save(self.save_path / sensor['id'] / ('%04d.png' % frame))
            elif 'lidar' in sensor['type']:
                np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)
                np.save(self.save_path / 'lidar_360' / ('%04d.npy' % frame), tick_data['lidar_360'], allow_pickle=True)

        Image.fromarray(tick_data['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))
        measurements_file = self.save_path / 'measurements' / ('%04d.json' % frame)
        with open(measurements_file, 'w') as f:
            json.dump(data, f, indent=4)