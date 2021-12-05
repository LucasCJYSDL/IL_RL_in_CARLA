import glob
import os
import sys
import random
import numpy as np
import pandas as pd


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

###################################
#   Constants
##################################

FPS = 20

WalkerSpawnRate = 10
CrossingRate = 0.0
EnableTwoWheel = False

DrawArea = True
DrawNavi = True
RENDER = True


####################################
#    Experiment Settings      
####################################

weathers = [ 
    (carla.WeatherParameters.ClearNoon, 'Clear Noon'),
    (carla.WeatherParameters.CloudyNoon, 'Cloudy Noon'),
    (carla.WeatherParameters.WetNoon, 'Wet Noon'),
    (carla.WeatherParameters.SoftRainNoon, 'Soft Rain Noon'),
    (carla.WeatherParameters.MidRainyNoon, 'Mid Rain Noon'),
    (carla.WeatherParameters.HardRainNoon, 'Hard Rain Noon'),
]
#ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset


class WaypointsFromCSV:
    def __init__(self, csv_path):
        waypoints = pd.read_csv(csv_path)
        waypoints = waypoints.dropna()
        location = ['loc_x', 'loc_y', 'loc_z']
        rotation = ['pitch', 'yaw', 'roll']

        self.loc = np.array(waypoints[location])
        self.rot = np.array(waypoints[rotation])

        print('Number of Waypoints : ', self.loc.shape[0])

    def get_transform(self, id):
        x, y, z = self.loc[id]
        pitch, yaw, roll = self.rot[id]

        return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll))

waypoints = WaypointsFromCSV('./waypoint.csv')   


Scenes = {
    'ped_center': [741, 610, 968, 1374, 2030, 413, 1636, 77],
    'ped_range' : [30.0, 30.0, 40.0, 40.0, 40.0, 30.0, 40.0, 30.0],
    'range' : [[-102.0, -60.0, 115.0, 152.0], [-27.0, 25.0, 111.0, 153.0], [133.0, 176.0, -210.0, -176.0], [-18.0, 25.0, -155.0, -117.0], [-102.0, -62.0, -18.0, 28.0]
               ,[75.0, 102.0, -90.0, -61.0], [-20.0, 31.0, -210.0, -175.0], [135.0, 155.0, -92.0, -53.0]],
    'height': [5.0, 5.0, 12.0, 12.0, 5.0, 12.0, 5.0, 12.0],
    'wal_min' : [16, 16, 16, 16, 16, 16, 16, 16],
    'wal_max' : [30, 50, 40, 50, 50, 30, 50, 25],
}

def GetScene(ID):
    origin = waypoints.loc[Scenes['ped_center'][ID]]
    PED_AREA = {
        'MINX': origin[0] - Scenes['ped_range'][ID],
        'MAXX': origin[0] + Scenes['ped_range'][ID],
        'MINY': origin[1] - Scenes['ped_range'][ID],
        'MAXY': origin[1] + Scenes['ped_range'][ID],
        'HEIGHT': Scenes['height'][ID]
    }

    AREA = {
        'MINX': Scenes['range'][ID][0],
        'MAXX': Scenes['range'][ID][1],
        'MINY': Scenes['range'][ID][2],
        'MAXY': Scenes['range'][ID][3],
        'HEIGHT' : Scenes['height'][ID]
    }

    NumOfWal = [Scenes['wal_min'][ID], Scenes['wal_max'][ID]]

    return AREA, PED_AREA, NumOfWal


Experiments = [[1760, 1917, 1883], [1219, 593, 2015, 659], [2662, 2232, 2158, 2470], [2737, 1927, 1725, 2204], [1620, 487],
               [2227, 2253], [2338, 2144, 1401, 2156], [1746, 1563]]
#TODO: choose more end points
Experiments_end = [[[1367, 1844], [2732, 2327], [2205, 2627]], [], [], [], [], [], [], []]

Wait_ticks = [200, 200, 200, 200, 200, 150, 200, 100]
task_types = [[[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0]],
              [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0], [1, -1, 2, 0]],
              [[0, 2, 1, -1], [0, 2, 1, -1], [2, 0, -1, 1], [2, 0, -1, 1]],
              [[-1, 1, 0, 2], [-1, 1, 0, 2], [1, -1, 2, 0], [1, -1, 2, 0]],
              [[1, -1, 2, 0], [1, -1, 2, 0]],
              [[0, 2, 1, -1], [1, -1, 2, 0]],
              [[0, 2, 1, -1], [-1, 1, 0, 2], [-1, 1, 0, 2], [2, 0, -1, 1]],
              [[-1, 1, 0, 2], [1, -1, 2, 0]]]
'''
lane_types = [[[0,1,1],[1,1,0],[0,1,1]], [[0,1,1],[0,1,0],[0,1,0],[0,1,1]],
             [[0,1,0],[0,1,0],[0,1,0],[0,1,1]], [[0,1,1],[1,1,0],[1,1,0],[0,1,1]],
             [[1,1,0],[0,1,1]], [[0,0,1], [0,1,0]], [[0,1,0],[0,1,1],[1,1,0],[0,1,0]], [[1,1,0],[0,1,1]]
]
'''
Scene_num = len(Experiments)


def GetConfig(scene_id, pose_id):

    pose = Experiments[scene_id][pose_id]
    pose_end = random.choice(Experiments_end[scene_id][pose_id])
    task_type = task_types[scene_id][pose_id]
    #lane_type = lane_types[scene_id][pose_id]
    wait_tick = Wait_ticks[scene_id]
    weather = random.choice(weathers) # you can specify it

    AREA, PED_AREA, NumOfWal = GetScene(scene_id)
    start = waypoints.get_transform(pose)
    end = waypoints.get_transform(pose_end)
    center_loc = waypoints.loc[Scenes['ped_center'][scene_id]]

    # spectator = waypoints.get_transform(pose)
    # spectator.location.z += 50

    config = {
        'pose_id' : pose_id,
        'AREA' : AREA,
        'PED_AREA': PED_AREA,
        'NumOfWal' : NumOfWal,
        'wait_tick' : wait_tick,
        'start_id' :  pose,
        'start' : start,
        'end': end,
        'task_type': task_type,
        #'lane_type': lane_type,
        'weather' : weather,
        'center' : center_loc
        # 'spectator' : spectator,
    }

    return config



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
# {
#     'name': 'FrontSemantic',
#     'type': 'sensor.camera.semantic_segmentation',
#     'width' : 800,
#     'height' : 600,
#     'fov': 100.0,
#     'pos' : carla.Transform(carla.Location(x=0.7, z=1.6)),
#     'convertor': cc.CityScapesPalette
#     },
{
    'name': 'Lidar',
    'type': 'sensor.lidar.ray_cast',
    'range' : 20.0,
    'channels': 1,
    'upper_fov': 0.0,
    'lower_fov': 0.0,
    'rotation_frequency': float(FPS),
    'points_per_second': FPS*360,
    'pos' : carla.Transform(carla.Location(x=1.7, z=1.12)),
    'convertor': None
    },
]

''' More cameras

{
        'name': 'BirdRGB',
        'type': 'sensor.camera.rgb',
        'width' : 640,
        'height' : 480,
        'pos' : carla.Transform(carla.Location(x=0.0, z=100.0), carla.Rotation(pitch=-90)),
        'convertor': cc.Raw,
    },
{
    'name': 'FrontDepth',
    'type': 'sensor.camera.depth',
    'width' : '640',
    'height' : '480',
    'pos' : carla.Transform(carla.Location(x=0.7, z=1.6)),
    'convertor': cc.LogarithmicDepth,
}
'''


if __name__ == '__main__':
    for i in range(Scene_num):
        pose_id_num = np.zeros((len(Experiments[i])))
        print(pose_id_num)
        for episode_id in range(100):
            pose = episode_id%len(Experiments[i])
            config = GetConfig(i,pose)
            print(episode_id,":")
            print(config)
            pose_id_num[config['pose_id']] += 1
        print(pose_id_num)
