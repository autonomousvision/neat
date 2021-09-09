import os
import time
import datetime
import pathlib
import json
import random

import numpy as np
import cv2
import carla

from PIL import Image, ImageDraw

from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController


WEATHERS = {
		'Clear': carla.WeatherParameters.ClearNoon,

		'Cloudy': carla.WeatherParameters.CloudySunset,

		'Wet': carla.WeatherParameters.WetSunset,

		'MidRain': carla.WeatherParameters.MidRainSunset,

		'WetCloudy': carla.WeatherParameters.WetCloudySunset,

		'HardRain': carla.WeatherParameters.HardRainNoon,

		'SoftRain': carla.WeatherParameters.SoftRainSunset,
}

azimuths = [45.0 * i for i in range(8)]

daytimes = {
	'Night': -80.0,
	'Twilight': 0.0,
	'Dawn': 5.0,
	'Sunset': 15.0,
	'Morning': 35.0,
	'Noon': 75.0,
}

WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
	return 'AutoPilot'


def _numpy(carla_vector, normalize=False):
	result = np.float32([carla_vector.x, carla_vector.y])

	if normalize:
		return result / (np.linalg.norm(result) + 1e-4)

	return result


def _location(x, y, z):
	return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
	return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
	A = np.stack([v1, -v2], 1)
	b = p2 - p1

	if abs(np.linalg.det(A)) < 1e-3:
		return False, None

	x = np.linalg.solve(A, b)
	collides = all(x >= 0) and all(x <= 1) # how many seconds until collision

	return collides, p1 + x[0] * v1


class AutoPilot(MapAgent):

	# for stop signs
	PROXIMITY_THRESHOLD = 30.0  # meters
	SPEED_THRESHOLD = 0.1
	WAYPOINT_STEP = 1.0  # meters

	def setup(self, path_to_conf_file):
		super().setup(path_to_conf_file)

	def _init(self):
		super()._init()

		self.angle = 0.0
		self.junction = False

		self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
		self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

		# for stop signs
		self._target_stop_sign = None # the stop sign affecting the ego vehicle
		self._stop_completed = False # if the ego vehicle has completed the stop sign
		self._affected_by_stop = False # if the ego vehicle is influenced by a stop sign

		self._vehicle.set_light_state(carla.VehicleLightState.HighBeam)

	def _get_angle_to(self, pos, theta, target):
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta),  np.cos(theta)],
			])

		aim = R.T.dot(target - pos)
		angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
		angle = 0.0 if np.isnan(angle) else angle 

		return angle

	def _get_control(self, target, far_target, pos, speed, theta, _draw=None):

		# Steering.
		angle_unnorm = self._get_angle_to(pos, theta, target)
		angle = angle_unnorm / 90

		steer = self._turn_controller.step(angle)
		steer = np.clip(steer, -1.0, 1.0)
		steer = round(steer, 3)

		# Acceleration.
		angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
		should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
		target_speed = 4.0 if should_slow else 7.0
		brake = self._should_brake()
		target_speed = target_speed if not brake else 0.0

		self.should_slow = int(should_slow)
		self.should_brake = int(brake)
		self.angle = angle
		self.angle_unnorm = angle_unnorm
		self.angle_far_unnorm = angle_far_unnorm

		delta = np.clip(target_speed - speed, 0.0, 0.25)
		throttle = self._speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, 0.75)

		if brake:
			steer *= 0.5
			throttle = 0.0

		return steer, throttle, brake, target_speed

	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		# change weather for visual diversity
		if self.step % 10 == 0 and self.save_path is not None:
			index = random.choice(range(len(WEATHERS)))
			dtime, altitude = random.choice(list(daytimes.items()))
			altitude = np.random.normal(altitude, 10)
			self.weather_id = WEATHERS_IDS[index] + dtime

			weather = WEATHERS[WEATHERS_IDS[index]]
			weather.sun_altitude_angle = altitude
			weather.sun_azimuth_angle = np.random.choice(azimuths)

			self._world.set_weather(weather)

		self.step += 1
		gps = self._get_position(input_data['gps'][1][:2])
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]

		near_node, near_command = self._waypoint_planner.run_step(gps)
		far_node, far_command = self._command_planner.run_step(gps)

		_draw = None

		steer, throttle, brake, target_speed = self._get_control(near_node, far_node, gps, speed, compass, _draw)

		control = carla.VehicleControl()
		control.steer = steer + 1e-2 * np.random.randn()
		control.throttle = throttle
		control.brake = float(brake)

		# don't save data while evaluating
		if self.step % 10 == 0 and self.save_path is not None:
			data = self.tick(input_data)
			self.save(far_node, near_command, steer, throttle, brake, target_speed, data)

		return control

	def _should_brake(self):
		actors = self._world.get_actors()

		vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
		light = self._is_light_red(actors.filter('*traffic_light*'))
		walker = self._is_walker_hazard(actors.filter('*walker*'))
		stop_sign = self._is_stop_sign_hazard(actors.filter('*stop*'))

		self.vehicle_hazard = 1 if vehicle is not None else 0
		self.traffic_light_hazard = 1 if light is not None else 0
		self.walker_hazard = 1 if walker is not None else 0
		self.stop_sign_hazard = 1 if stop_sign is not None else 0

		return any(x is not None for x in [vehicle, light, walker, stop_sign])

	@staticmethod
	def point_inside_boundingbox(point, bb_center, bb_extent):

		A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
		B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
		D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
		M = carla.Vector2D(point.x, point.y)

		AB = B - A
		AD = D - A
		AM = M - A
		am_ab = AM.x * AB.x + AM.y * AB.y
		ab_ab = AB.x * AB.x + AB.y * AB.y
		am_ad = AM.x * AD.x + AM.y * AD.y
		ad_ad = AD.x * AD.x + AD.y * AD.y

		return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

	def _get_forward_speed(self, transform=None, velocity=None):
		""" Convert the vehicle transform directly to forward speed """
		if not velocity:
			velocity = self._vehicle.get_velocity()
		if not transform:
			transform = self._vehicle.get_transform()

		vel_np = np.array([velocity.x, velocity.y, velocity.z])
		pitch = np.deg2rad(transform.rotation.pitch)
		yaw = np.deg2rad(transform.rotation.yaw)
		orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
		speed = np.dot(vel_np, orientation)
		return speed

	def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
		"""
		Check if the given actor is affected by the stop
		"""
		affected = False
		# first we run a fast coarse test
		current_location = actor.get_location()
		stop_location = stop.get_transform().location
		if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
			return affected

		stop_t = stop.get_transform()
		transformed_tv = stop_t.transform(stop.trigger_volume.location)

		# slower and accurate test based on waypoint's horizon and geometric test
		list_locations = [current_location]
		waypoint = self._world.get_map().get_waypoint(current_location)
		for _ in range(multi_step):
			if waypoint:
				waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
				if not waypoint:
					break
				list_locations.append(waypoint.transform.location)

		for actor_location in list_locations:
			if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
				affected = True

		return affected

	def _is_stop_sign_hazard(self, stop_sign_list):
		if self._affected_by_stop:
			if not self._stop_completed:
				current_speed = self._get_forward_speed()
				if current_speed < self.SPEED_THRESHOLD:
					self._stop_completed = True
					return None
				else:
					return self._target_stop_sign
			else:
				# reset if the ego vehicle is outside the influence of the current stop sign
				if not self._is_actor_affected_by_stop(self._vehicle, self._target_stop_sign):
					self._affected_by_stop = False
					self._stop_completed = False
					self._target_stop_sign = None
				return None

		ve_tra = self._vehicle.get_transform()
		ve_dir = ve_tra.get_forward_vector()

		wp = self._world.get_map().get_waypoint(ve_tra.location)
		wp_dir = wp.transform.get_forward_vector()

		dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

		if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
			for stop_sign in stop_sign_list:
				if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
					# this stop sign is affecting the vehicle
					self._affected_by_stop = True
					self._target_stop_sign = stop_sign
					
					return self._target_stop_sign

		return None

	def _is_light_red(self, lights_list):
		if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
			affecting = self._vehicle.get_traffic_light()

			for light in self._traffic_lights:
				if light.id == affecting.id:
					return affecting

		return None

	def _is_walker_hazard(self, walkers_list):
		z = self._vehicle.get_location().z
		p1 = _numpy(self._vehicle.get_location())
		v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

		for walker in walkers_list:
			v2_hat = _orientation(walker.get_transform().rotation.yaw)
			s2 = np.linalg.norm(_numpy(walker.get_velocity()))

			if s2 < 0.05:
				v2_hat *= s2

			p2 = -3.0 * v2_hat + _numpy(walker.get_location())
			v2 = 8.0 * v2_hat

			collides, collision_point = get_collision(p1, v1, p2, v2)

			if collides:
				return walker

		return None

	def _is_vehicle_hazard(self, vehicle_list):
		z = self._vehicle.get_location().z

		o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
		p1 = _numpy(self._vehicle.get_location())
		s1 = max(10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
		v1_hat = o1
		v1 = s1 * v1_hat

		for target_vehicle in vehicle_list:
			if target_vehicle.id == self._vehicle.id:
				continue

			o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
			p2 = _numpy(target_vehicle.get_location())
			s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
			v2_hat = o2
			v2 = s2 * v2_hat

			p2_p1 = p2 - p1
			distance = np.linalg.norm(p2_p1)
			p2_p1_hat = p2_p1 / (distance + 1e-4)

			angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
			angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

			# to consider -ve angles too
			angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
			angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)

			if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
				continue
			elif angle_to_car > 30.0:
				continue
			elif distance > s1:
				continue

			return target_vehicle

		return None