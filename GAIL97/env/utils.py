import glob
import os
import sys
import math
import numpy as np
import pandas as pd

from collections import OrderedDict

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



###################################
#   Parameters
###################################

FPS = 20
CrossingRate = 0.0  # allow walkers to cross road
SafyDis = 3.0       # The safy distance between spawned surrounding car and ego car



###################################
#   Waypoints
###################################

class Waypoints:
    def __init__(self, csv_path):
        waypoints = pd.read_csv(csv_path)
        waypoints = waypoints.dropna()
        location = ['loc_x', 'loc_y', 'loc_z']
        rotation = ['pitch', 'yaw', 'roll']

        self.locs = np.array(waypoints[location])
        self.rots = np.array(waypoints[rotation])
        self.cnt = len(waypoints['loc_x'])

    def get_transform(self, id):
        x, y, z = self.locs[id]
        pitch, yaw, roll = self.rots[id]

        return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll))

    def render(self, debug):
        for i in range(self.cnt):
            x, y, z = self.locs[i]
            loc = carla.Location(x, y, z)

            debug.draw_point(loc, persistent_lines=True)
            debug.draw_string(loc, str(i), False, carla.Color(255, 162, 0), 200, persistent_lines=False)



####################################
#    Helper Functions     
####################################

def get_weather(weather):
    weather_dict = {
        'ClearNoon' : carla.WeatherParameters.ClearNoon,
        'CloudyNoon' : carla.WeatherParameters.CloudyNoon,
        'WetNoon' : carla.WeatherParameters.WetNoon,
        'HardRainNoon' : carla.WeatherParameters.HardRainNoon,
        #ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset
    }
    return weather_dict[weather]


def get_PED_area(scene, waypoints):
    origin = waypoints.locs[scene['ped_center']]
    PED_AREA = {
        'MINX': origin[0] - scene['ped_range'],
        'MAXX': origin[0] + scene['ped_range'],
        'MINY': origin[1] - scene['ped_range'],
        'MAXY': origin[1] + scene['ped_range'],
    }
    return PED_AREA


def get_area(scene):
    AREA = {
        'MINX': scene['area'][0],
        'MAXX': scene['area'][1],
        'MINY': scene['area'][2],
        'MAXY': scene['area'][3],
    }
    return AREA


def INSIDE(loc, AREA):
    return (loc is not None) and loc.x >= AREA['MINX'] and loc.x <= AREA['MAXX'] and \
        loc.y >= AREA['MINY'] and loc.y <= AREA['MAXY'] 


def get4areas(AREA):
    areas=[]
    area1={
        'MINX': AREA['MINX']+15,
        'MAXX': AREA['MINX']+20,
        'MINY': AREA['MINY'],
        'MAXY': AREA['MINY']+20,
    }
    area2={
        'MINX': AREA['MINX']+15,
        'MAXX': AREA['MINX']+20,
        'MINY': AREA['MAXY']-20,
        'MAXY': AREA['MAXY'],
    }
    area3={
        'MINX': AREA['MAXX']-20,
        'MAXX': AREA['MAXX']-16,
        'MINY': AREA['MINY'],
        'MAXY': AREA['MINY']+20,
    }
    area4={
        'MINX': AREA['MAXX']-20,
        'MAXX': AREA['MAXX']-16,
        'MINY': AREA['MAXY']-20,
        'MAXY': AREA['MAXY'],
    }
    areas.append(area1)
    areas.append(area2)
    areas.append(area3)
    areas.append(area4)
    return areas


def get_task_type(loc, area, task_type):
    if loc.x < area['MINX']:
        return task_type[0]

    if loc.x > area['MAXX']:
        return task_type[1]

    if loc.y < area['MINY']:
        return task_type[2]

    if loc.y > area['MAXY']:
        return task_type[3]
    
    assert False, 'loc must be in out of the area when calling this function'



######################################
#    Calc distance
######################################

def get_closet_wp(path, cur_loc, debug, draw_dis):

    lent = len(path)
    min_dis = 1e6
    min_ind = 0
    for i in range(lent):
        temp_loc = path[i][0].transform.location
        dis = (temp_loc.x - cur_loc.x) * (temp_loc.x - cur_loc.x) + (temp_loc.y - cur_loc.y) * (temp_loc.y - cur_loc.y)
        if dis < min_dis:
            min_dis = dis
            min_ind = i

    if draw_dis:
        temp_loc = path[min_ind][0].transform.location
        debug.draw_line(cur_loc, temp_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=1.0, persistent_lines=False)

    min_dis = math.sqrt(min_dis)

    x1 = path[min_ind+1][0].transform.location.x - path[min_ind][0].transform.location.x + 0.001
    y1 = path[min_ind+1][0].transform.location.y - path[min_ind][0].transform.location.y + 0.001

    x2 = cur_loc.x - path[min_ind][0].transform.location.x
    y2 = cur_loc.y - path[min_ind][0].transform.location.y 

    min_height = (x1 * y2 - y1 * x2) / math.sqrt(x1**2 + y1**2)
    #print('height', min_height)

    return path[min_ind][0], path[min_ind+1][0], min_height


def get_polygan(debug, veh_loc, yaw_loc, neigh_loc, next_loc, draw_dis):

    if draw_dis:
        debug.draw_line(veh_loc, yaw_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(yaw_loc, next_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(next_loc, neigh_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(neigh_loc, veh_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)

    vec_1 = (veh_loc.x - neigh_loc.x, veh_loc.y - neigh_loc.y)
    vec_2 = (next_loc.x - neigh_loc.x, next_loc.y - neigh_loc.y)
    vec_3 = (veh_loc.x - next_loc.x, veh_loc.y - next_loc.y)
    vec_4 = (veh_loc.x - yaw_loc.x, veh_loc.y - yaw_loc.y)
    vec_5 = (next_loc.x - yaw_loc.x, next_loc.y - yaw_loc.y)

    len_1 = math.sqrt(vec_1[0]**2 + vec_1[1]**2)
    len_2 = math.sqrt(vec_2[0]**2 + vec_2[1]**2)
    len_3 = math.sqrt(vec_3[0]**2 + vec_3[1]**2)
    len_4 = math.sqrt(vec_4[0]**2 + vec_4[1]**2)
    len_5 = math.sqrt(vec_5[0]**2 + vec_5[1]**2)

    circum_1 = len_1 + len_2 + len_3
    circum_2 = len_3 + len_4 + len_5

    if len_1 < 0.1:
        area_1 = 0.
    else:
        area_1 = math.sqrt(circum_1/2 * (circum_1/2 - len_1) * (circum_1/2 - len_2) * (circum_1/2 - len_3))
    if len_5 < 0.1:
        area_2 = 0.
    else:
        area_2 = math.sqrt(circum_2/2 * (circum_2/2 - len_3) * (circum_2/2 - len_4) * (circum_2/2 - len_5))

    return area_1 + area_2



######################################
#    Render
######################################

def draw_waypoint_union(debug, l0, l1, to_show, color=carla.Color(0, 0, 255), lt=100):

    debug.draw_line( l0.transform.location + carla.Location(z=0.25), l1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(l1.transform.location + carla.Location(z=0.25), 0.01, color, lt, False)
    debug.draw_string(l1.transform.location, str(to_show), False, carla.Color(255, 162, 0), 200, persistent_lines=False)


def draw_area(debug, AREA, color = (0, 255, 0)):
    a0 = carla.Location(x = AREA['MINX'], y = AREA['MINY'], z = 5)
    a1 = carla.Location(x = AREA['MINX'], y = AREA['MAXY'], z = 5)
    a2 = carla.Location(x = AREA['MAXX'], y = AREA['MINY'], z = 5)
    a3 = carla.Location(x = AREA['MAXX'], y = AREA['MAXY'], z = 5)
    
    color = carla.Color(color[0], color[1], color[2])
    thickness = 1
    debug.draw_line(a0, a1, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a1, a3, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a3, a2, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a2, a0, thickness = thickness, color = color, life_time = 100.)



######################################
#    Lidar
######################################

def pre_process_lidar(points):
    lidar_feature = [1.0] * 720
    # print("points: ", points)
    raw_lidar = [np.array([-1.0, -1.0, -1.0])] * 1440
    i = 0
    for point in points:
        raw_lidar[i] = np.array([point[0], point[1], point[2]])
        i = (i+1)%1440
        point = np.array([point[0], point[1]])
        rel_dis = get_distance(point)
        rel_deg = get_angle(point)

        if rel_deg < 720 and rel_dis < lidar_feature[rel_deg]:
            lidar_feature[rel_deg] = rel_dis

    # print("lidar_list: ", raw_lidar)
    # print("number of points: ", i)
    return np.array(lidar_feature), np.array(raw_lidar)


def get_distance(point):
    d_x = point[0]
    d_y = point[1]
    dis = math.sqrt(d_x*d_x+d_y*d_y)
    if dis < 0.1:
        print("Too close! Impossible!!", d_x, " ", d_y)
        return 0.1/40.0
    rel_dis = dis/40.0
    # print("rel_dis", rel_dis)
    rel_dis = min(1.0, rel_dis)

    return rel_dis

def get_angle(point):

    if point[0] - 0.0 < 1e-3 and point[0] > 0.0:
        point[0] = 1e-3
    if 0.0 - point[0] < 1e-3 and point[0] < 0.0:
        point[0] = -1e-3

    angle = math.atan2(point[1], point[0])
    while (angle > math.pi):
        angle -= 2 * math.pi
    while (angle < -math.pi):
        angle += 2 * math.pi
    assert angle>=-np.pi and angle<=np.pi
    degree = math.degrees(angle)
    # print("degree", degree)

    rel_degree = int((4*degree + 720.0) % 1440)
    # print("rel_degree", rel_degree)

    return rel_degree


def get_matrix(rotation):
    """
    Creates matrix from carla transform.
    """
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    matrix = np.matrix(np.identity(2))
    # matrix[0, 0] = c_y
    # matrix[0, 1] = -s_y
    # matrix[1, 0] = s_y
    # matrix[1, 1] = c_y
    matrix[0, 0] = -s_y
    matrix[0, 1] = -c_y
    matrix[1, 0] = c_y
    matrix[1, 1] = -s_y
    return matrix


def sensor_to_world(cords, sensor_transform):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor_transform.rotation)
    world_cords = np.dot(sensor_world_matrix, np.transpose(cords))
    world_cords[0,0] += sensor_transform.location.x
    world_cords[0,1] += sensor_transform.location.y
    world_cords = world_cords.tolist()
    return world_cords[0]


def render_LIDAR(points, lidar_transform, debug):
    for point in points:
        point = np.array([point[0], point[1]])
        rel_angle = get_angle(point)
        point_loc = sensor_to_world(point, lidar_transform)
        point_loc = carla.Location(x=point_loc[0], y=point_loc[1], z=1.5)
        if rel_angle<720:
            debug.draw_point(point_loc, 0.1, carla.Color(255, 162, 0), 0.05, False)


def get_intention(vec_1, vec_2):
    l_1 = math.sqrt(vec_1[0]**2+vec_1[1]**2)
    l_2 = math.sqrt(vec_2[0]**2+vec_2[1]**2)
    if l_1<0.1 or l_2<0.1:
        return -1.
    return (vec_1[0]*vec_2[0]+vec_1[1]*vec_2[1])/l_1/l_2