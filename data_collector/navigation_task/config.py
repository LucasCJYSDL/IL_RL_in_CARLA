import glob
import os
import sys
import random
import numpy as np
import pandas as pd

'''
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
'''

import carla

###################################
#   Constants
##################################

WaitTick = 10
FPS = 10
DisNearTarget = 10

WalkerSpawnRate = 10
# TODO: double check here CrossingRate
CrossingRate = 1.0
EnableTwoWheel = False
SafyDis = 1.0

DrawArea = True
DrawNavi = True
RENDER = False

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
    'center': [529, 1220, 741, 1161, 1026, 394, 841, 1545],
    'range' : [60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 150.0],
    'height': [5.0, 5.0, 5.0, 5.0, 12.0, 12.0, 12.0, 12.0],

    'veh_min' : [5, 5, 5, 5, 4, 5, 5, 16],
    'veh_max' : [15, 15, 15, 15, 13, 15, 15, 50],
    'wal_min' : [16, 16, 16, 16, 16, 16, 16, 50],
    'wal_max' : [50, 50, 50, 50, 50, 50, 50, 150],
}

def GetScene(ID):
    origin = waypoints.loc[Scenes['center'][ID]]

    AREA = {
        'MINX': origin[0] - Scenes['range'][ID],
        'MAXX': origin[0] +  Scenes['range'][ID],
        'MINY': origin[1] - Scenes['range'][ID],
        'MAXY': origin[1] + Scenes['range'][ID],
        'HEIGHT' : Scenes['height'][ID]
    }

    NumOfVeh = [Scenes['veh_min'][ID], Scenes['veh_max'][ID]]
    NumOfWal = [Scenes['wal_min'][ID], Scenes['wal_max'][ID]]

    return AREA, NumOfVeh, NumOfWal


Experiments = {
    'Straight' : [[531, 2101, 0], [73, 211, 1], [41, 1933, 2], [2576, 2244, 3], [644, 1623, 4]],
    'SingleTurn' : [[531, 1558, 0], [73, 1430, 1], [41, 685, 2], [2576, 2457, 3], [686, 2304, 6]],
    'Navigation' : [[644, 730, 4], [1973, 2027, 5], [686, 718, 6], [859, 531, 7], [1731, 531, 7]],
}
#ExpNames = ['Straight', 'SingleTurn', 'Navigation']
ExpNames = ['Navigation']


def GetConfig(exp_name):
    #pose_id = random.randint(0, len(Experiments[exp_name])-1) # you can specify it
    pose_id = 2
    pose = Experiments[exp_name][pose_id]

    weather = random.choice(weathers) # you can specify it

    AREA, NumOfVeh, NumOfWal = GetScene(pose[2])
    start = waypoints.get_transform(pose[0])
    end = waypoints.get_transform(pose[1])

    spectator = waypoints.get_transform(pose[0])
    spectator.location.z += 50

    config = {
        'Exp_Name' : exp_name,
        'pose_id' : pose_id,

        'AREA' : AREA,
        'NumOfVeh' : NumOfVeh,
        'NumOfWal' : NumOfWal,

        'start_id' :  pose[0],
        'start' : start,
        'end_id' : pose[1],
        'end' : end,

        'weather' : weather,
        'spectator' : spectator,
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
        'width' : 640,
        'height' : 480,
        'pos' : carla.Transform(carla.Location(x=1.0, z=1.6)),
        'convertor': cc.Raw,
    },
    {
        'name': 'BirdRGB',
        'type': 'sensor.camera.rgb',
        'width' : 640,
        'height' : 480,
        'pos' : carla.Transform(carla.Location(x=0.0, z=100.0), carla.Rotation(pitch=-90)),
        'convertor': cc.Raw,
    },
]

''' More cameras


{
    'name': 'FrontDepth',
    'type': 'sensor.camera.depth',
    'width' : '640',
    'height' : '480',
    'pos' : carla.Transform(carla.Location(x=0.7, z=1.6)),
    'convertor': cc.LogarithmicDepth,
},
{
    'name': 'FrontSemantic',
    'type': 'sensor.camera.semantic_segmentation',
    'width' : '640',
    'height' : '480',
    'pos' : carla.Transform(carla.Location(x=0.7, z=1.6)),
    'convertor': cc.CityScapesPalette
}
'''

if __name__ == '__main__':
    for i in range(10):
        print(waypoints.get_transform(i))