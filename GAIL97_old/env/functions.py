import glob
import os
import sys
import math
import numpy as np
import pandas as pd

from PIL import Image
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
EnableTwoWheel = False

SafyDis = 3.0  # The safy distance between spawned surrounding car and ego car
CrossingRate = 0.0 # allow walkers to cross road

DrawArea = False
DrawNavi = False

####################################
#    Functions      
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


class WaypointsFromCSV:
    def __init__(self, csv_path):
        waypoints = pd.read_csv(csv_path)
        waypoints = waypoints.dropna()
        location = ['loc_x', 'loc_y', 'loc_z']
        rotation = ['pitch', 'yaw', 'roll']

        self.loc = np.array(waypoints[location])
        self.rot = np.array(waypoints[rotation])

    def get_transform(self, id):
        x, y, z = self.loc[id]
        pitch, yaw, roll = self.rot[id]

        return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll))


def get_PED_area(scene, waypoints):
    origin = waypoints.loc[scene['ped_center']]
    PED_AREA = {
        'MINX': origin[0] - scene['ped_range'],
        'MAXX': origin[0] + scene['ped_range'],
        'MINY': origin[1] - scene['ped_range'],
        'MAXY': origin[1] + scene['ped_range'],
    }
    return PED_AREA


def get_area(scene):
    AREA = {
        'MINX': scene['range'][0],
        'MAXX': scene['range'][1],
        'MINY': scene['range'][2],
        'MAXY': scene['range'][3],
    }
    return AREA


def INSIDE(loc, AREA):
    return (loc is not None) and loc.x > AREA['MINX'] and loc.x < AREA['MAXX'] and \
        loc.y > AREA['MINY'] and loc.y < AREA['MAXY'] 


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
#    Visualization
######################################

def draw_waypoint_union(debug, l0, l1, to_show, color=carla.Color(0, 0, 255), lt=100):
    debug.draw_line(l0, l1, thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(l1, 0.1, color, lt, False)
    debug.draw_string(l1, str(to_show), False, carla.Color(255, 162, 0), 200, persistent_lines=False)


def draw_area(debug, AREA):
    a0 = carla.Location(x = AREA['MINX'], y = AREA['MINY'], z = 25)
    a1 = carla.Location(x = AREA['MINX'], y = AREA['MAXY'], z = 25)
    a2 = carla.Location(x = AREA['MAXX'], y = AREA['MINY'], z = 25)
    a3 = carla.Location(x = AREA['MAXX'], y = AREA['MAXY'], z = 25)
    
    color = carla.Color(0, 255,0)
    thickness = 1
    debug.draw_line(a0, a1, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a1, a3, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a3, a2, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a2, a0, thickness = thickness, color = color, life_time = 100.)


######################################
#    Sensors
######################################

from carla import ColorConverter as cc

CAMERAS = [
    {
        'name': 'FrontRGB',
        'type': 'sensor.camera.rgb',
        'width' : 800,
        'height' : 600,
        'fov': 100.0,
        'pos' : carla.Transform(carla.Location(x=1.0, z=1.6)),
        'convertor': cc.Raw,
    },

{
    'name': 'Lidar',
    'type': 'sensor.lidar.ray_cast',
    'range' : 20.0,
    'channels': 1,
    'upper_fov': 0.0,
    'lower_fov': -5.0,
    'rotation_frequency': float(FPS),
    'points_per_second': FPS*360,
    'pos' : carla.Transform(carla.Location(x=0.0, z=1.15)),
    'convertor': None
    },
]

def pre_process_image(image):
    """
    Arguments:
        image {width 800, height 600}
    
    Returns:
        image {width 200, height 88}
    """ 

    image = image[115 : 510, :]
    image = np.array(Image.fromarray(image).resize((200, 88)))
    # image = scipy.misc.imresize(image, [88, 200])
    # image = image.astype(np.float32)
    # image = np.multiply(image, 1.0 / 255.0)
    return image


def get_distance(point):
    d_x = point[0]
    d_y = point[1]
    dis = math.sqrt(d_x*d_x+d_y*d_y)
    if dis < 0.1:
        print("Too close! Impossible!!", d_x, " ", d_y)
        return 0.1/20.0
    rel_dis = dis/20.0
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

    rel_degree = int((degree+360.0) % 360)
    # print("rel_degree", rel_degree)

    return rel_degree


def pre_process_lidar(points):

    lidar_feature = [1.0] * 360
    # print("points: ", points)
    for point in points:
        point = np.array([point[0], point[1]])
        rel_dis = get_distance(point)
        rel_deg = get_angle(point)
        if rel_dis < lidar_feature[rel_deg]:
            lidar_feature[rel_deg] = rel_dis
    # print("lidar_list: ", lidar_feature)
    return np.array(lidar_feature)

def pre_process_measure(measure_dict, center, lane_type):

    measure = []
    measure.append(measure_dict['speed']/30.0)
    loc = measure_dict['location']
    measure.append((loc.x - center[0])/25.0)
    measure.append((loc.y - center[1])/25.0)
    # print("center: ", center)
    angle = np.radians(measure_dict['rotation'].yaw)
    while (angle > math.pi):
        angle -= 2 * math.pi
    while (angle < -math.pi):
        angle += 2 * math.pi
    assert angle >= -np.pi and angle <= np.pi
    measure.append(angle)
    measure.extend(lane_type)
    # print("measure: ", measure)

    return measure