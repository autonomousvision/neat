import os
import json
import datetime
import pathlib
import time
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from aim_va.model import MultiTaskImageNetwork
from aim_va.class_converter import sub_classes
from aim_va.data import seg_to_one_hot
from mmseg.apis import inference_segmentor, init_segmentor


SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'WaypointSegmentationAgent'


def crop_image(image, crop_width_factor):
	"""
	crop a PIL image, returning a channels-first numpy array.
	"""
	image = Image.fromarray(image)
	(width, height) = (image.width, image.height)
	image = np.asarray(image)
	crop = int(crop_width_factor * width)
	start_x = height//2 - crop//2
	start_y = width//2 - crop//2
	cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
	return cropped_image[:,:]


def scale_image(image, scale):
	"""
	Scale an image
	"""
	image = Image.fromarray(image)
	(width, height) = (image.width // scale, image.height // scale)
	im_resized = image.resize((width, height), resample=Image.NEAREST)
	image = np.asarray(im_resized)
	return image


class WaypointSegmentationAgent(autonomous_agent.AutonomousAgent):
	
	def setup(self, path_to_conf_file):
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		args_file = open(os.path.join(path_to_conf_file, 'args.txt'), 'r')
		self.args = json.load(args_file)

		args_file.close()
		self.input_buffer = {'seg_center': deque()}
		

		self.converter = sub_classes[self.args['classes']]

		num_segmentation_classes = len(np.unique(self.converter)) 

		self.net = MultiTaskImageNetwork('cuda', num_segmentation_classes, self.args['pred_len'], 1)
		
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		seg_checkpoint_path = os.path.join(path_to_conf_file, 'iter_80000.pth')
		seg_conf_path = os.path.join(path_to_conf_file, 'fcn_r50-d8_512x1024_80k_carla_full.py')
		self.segmentation_net = init_segmentor(seg_conf_path, seg_checkpoint_path)

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'seg_center').mkdir()

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 800, 'height': 600, 'fov': 100,
					'id': 'rgb_center'
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

	def tick(self, input_data):
		self.step += 1
		_rgb = crop_image(input_data['rgb_center'][1][:, :, :3], self.args['input_crop']) # from 800 * 600 to 512 * 512

		result = inference_segmentor(self.segmentation_net, _rgb) 

		seg = result[0].astype(np.uint8)

		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		self.speed = speed

		result = {
				'seg_center': scale_image(seg, 2), # from 512 * 512 to 256 * 256
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		
		tick_data = self.tick(input_data)

		if self.step < self.args['seq_len']:
			seg_center = seg_to_one_hot(np.array((Image.fromarray(tick_data['seg_center'], 'L'))), self.converter).unsqueeze(0)
			self.input_buffer['seg_center'].append(seg_center.to('cuda', dtype=torch.float32))

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		encoding = []
		seg_center = seg_to_one_hot(np.array((Image.fromarray(tick_data['seg_center']))), self.converter).unsqueeze(0)
		self.input_buffer['seg_center'].popleft()
		self.input_buffer['seg_center'].append(seg_center.to('cuda', dtype=torch.float32))
		encoding.append(self.net.image_encoder(list(self.input_buffer['seg_center'])))
		
		pred_wp = self.net(encoding, target_point)
		
		steer, throttle, brake = self.net.control_pid(pred_wp, gt_velocity)

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)

		if SAVE_PATH is not None and self.step % 10 == 0:
			self.save(tick_data)

		return control

	def save(self, tick_data):
		frame = self.step // 10

		Image.fromarray(tick_data['seg_center']).save(self.save_path / 'seg_center' / ('%04d.png' % frame))

	def destroy(self):
		del self.net
		del self.segmentation_net