#!/usr/bin/env python


import glob
import os
import sys
import time
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import csv



def get_scene_layout(carla_map, precision = 1.0):
    """
    Function to extract the full scene layout to be used as a full scene description to be
    given to the user
    :return: a dictionary describing the scene.
    """

    def _lateral_shift(transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    topology = [x[0] for x in carla_map.get_topology()]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    # A road contains a list of lanes, a each lane contains a list of waypoints
    map_dict = dict()

    for waypoint in topology:
        waypoints = [waypoint]
        nxt = waypoint.next(precision)
        if len(nxt) > 0:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                else:
                    break

        lane = {
            "waypoints": waypoints
        }

        if map_dict.get(waypoint.road_id) is None:
            map_dict[waypoint.road_id] = {}
        map_dict[waypoint.road_id][waypoint.lane_id] = lane

    # Generate waypoints graph
    waypoints_graph = dict()
    for road_key in map_dict:
        for lane_key in map_dict[road_key]:
            # List of waypoints
            lane = map_dict[road_key][lane_key]

            for i in range(0, len(lane["waypoints"])):

                # Waypoint Position
                wl = lane["waypoints"][i].transform.location

                # Waypoint Orientation
                wo = lane["waypoints"][i].transform.rotation

                # Waypoint dict
                waypoint_dict = {
                    "road_id": road_key,
                    "lane_id": lane_key,
                    "id": lane["waypoints"][i].id,
                    "position": wl,
                    "orientation": wo
                }
                waypoints_graph[map_dict[road_key][lane_key]["waypoints"][i].id] = waypoint_dict

    return waypoints_graph

def create_csv(path):

    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["ID", "loc_x", "loc_y", "loc_z", "pitch", "yaw", "roll"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):

    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def draw_waypoint(debug, map, precision, csv_path):
    create_csv(csv_path)
    graph = get_scene_layout(map, precision)
    cnt = 0
    for v in graph.values():
        loc = v["position"]
        idn = v["id"]
        rot = v["orientation"]
        row = [str(cnt), str(loc.x), str(loc.y), str(loc.z), str(rot.pitch), str(rot.yaw), str(rot.roll)]
        write_csv(csv_path, row)
        debug.draw_point(loc, persistent_lines=True)
        if cnt == 1624:
            debug.draw_string(loc, str(cnt), False, carla.Color(255, 162, 0), 200, persistent_lines=False)
        cnt += 1

    print("number of points: ", cnt)



if __name__ == '__main__':


    client = carla.Client("localhost", 2000)

    client.set_timeout(4.0)
    world = client.get_world()
    map = world.get_map()
    debug = world.debug
    csv_path = './waypoint.csv'
    draw_waypoint(debug, map, 5.0, csv_path)

    

