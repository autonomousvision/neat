import os
import sys
import json
import time
import random
import argparse
import multiprocessing
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


def interpolate_trajectory(world_map, waypoints_trajectory, hop_resolution=1.0):
    """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    :param world: an reference to the CARLA world so we can use the planner
    :param waypoints_trajectory: the current coarse trajectory
    :param hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    :return: the full interpolated route both in GPS coordinates and also in its original form.
    """

    dao = GlobalRoutePlannerDAO(world_map, hop_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    # Obtain route plan
    route = []
    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        for wp_tuple in interpolated_trace:
            route.append((wp_tuple[0].transform, wp_tuple[1]))
            # print (wp_tuple[0].transform.location, wp_tuple[1])

    return route


def parse_routes_file(route_filename, single_route=None):
    """
    Returns a list of route elements that is where the challenge is going to happen.
    :param route_filename: the path to a set of routes.
    :param single_route: If set, only this route shall be returned
    :return:  List of dicts containing the waypoints, id and town of the routes
    """

    list_route_descriptions = []
    tree = ET.parse(route_filename)
    for route in tree.iter("route"):
        route_town = route.attrib['town']
        route_id = route.attrib['id']
        if single_route and route_id != single_route:
            continue

        waypoint_list = []  # the list of waypoints that can be found on this route
        for waypoint in route.iter('waypoint'):
            waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
                                                y=float(waypoint.attrib['y']),
                                                z=float(waypoint.attrib['z'])))

            # Waypoints is basically a list of XML nodes

        list_route_descriptions.append({
            'id': route_id,
            'town_name': route_town,
            'trajectory': waypoint_list
        })

    return list_route_descriptions


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)

    routes_list = parse_routes_file(args.routes_file)

    for idx, route in enumerate(routes_list):
        town_name = route['town_name']
        world = client.load_world(town_name)
        world_map = world.get_map()

        weather = carla.WeatherParameters(sun_altitude_angle=90.0)
        world.set_weather(weather)

        interpolated_route = interpolate_trajectory(world_map, route['trajectory'])

        for idx, wp in enumerate(interpolated_route):
            waypoint = wp[0].location
            color = carla.Color(r=0, g=255, b=0)
            if idx == 0: color = carla.Color(r=255, g=0, b=0) # start of route
            if idx == len(interpolated_route)-1: color = carla.Color(r=0, g=0, b=255) # end of route

            world.debug.draw_string(wp[0].location, 'O', draw_shadow=False, color=color, life_time=100000, persistent_lines=True)

        if args.save_path is not None:
            camera.listen(lambda image: image.save_to_disk(os.path.join(args.save_path, '%d'%idx)))

        print (len(interpolated_route))
        user_input = input('Continue [y/n]: ')
        if user_input == 'n':
            exit()


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser()

    parser.add_argument('--routes_file', type=str, required=True, help='file containing the route waypoints')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    if args.save_path is not None and not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    main()