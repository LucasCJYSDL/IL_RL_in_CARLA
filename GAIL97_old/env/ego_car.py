import glob
import os
import sys
import math
import queue
import time
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from env.functions import DrawNavi, CAMERAS
from env.functions import draw_waypoint_union

from env.agents.navigation.global_route_planner import RoadOption


class EgoCar:

    def __init__(self, world, client, scene, waypoints, planner):
        self.world = world
        self.client = client
        self.scene = scene
        self.waypoints = waypoints
        self.planner = planner
        self.debug = self.world.debug

        bp = self.world.get_blueprint_library().filter("vehicle.audi.a2")[0] 
        self.start = self.waypoints.get_transform(self.scene['start'])
        self.vehicle = self.world.spawn_actor(bp, self.start)

        self.sensors = [] # store camera, Lidar
        self.measures = [] # store collision, lane invasion etc.


    def step(self, control):
        self.vehicle.apply_control(control)


    def set_sensors(self):
        for camera in CAMERAS:
            self.sensors.append(CameraSensor(self.world, self.vehicle, camera))


    def set_measures(self):
        self.measures = [
            CollisionSensor(self.world, self.vehicle, 'Collision'),
            LaneInvasionSensor(self.world, self.vehicle, 'LaneInvasion'),
        ]


    def get_sensors(self, frame0):
        sensor_dict = {}
        for sensor in self.sensors:
            sensor_dict[sensor.name] = sensor.get_data(frame0)
        return sensor_dict


    def get_measures(self):
        measure_dict = {measure.name : measure.get_data() for measure in self.measures}

        v = self.vehicle.get_velocity()
        measure_dict['speed'] = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2) # km/h

        current = self.vehicle.get_transform()
        measure_dict['location'] = current.location
        measure_dict['rotation'] = current.rotation

        return measure_dict


    def get_direction(self, cur_loc, tar_loc):
        """
        Returns:
            new_direction(int) -- 0: go straight, 1: follow, 2:left, 3:right
        """
        plan = self.planner.trace_route(cur_loc, tar_loc)
        direction = plan[0][1]

        if DrawNavi:
            for i in range(len(plan) - 1):
                w0 = plan[i][0]
                w1 = plan[i+1][0]
                draw_waypoint_union(self.debug, w0, w1, '')

        if direction in {RoadOption.STRAIGHT}: # go straight
            new_direction = 0

        elif direction in {RoadOption.VOID, RoadOption.LANEFOLLOW} :  # follow lane
            new_direction = 1

        elif direction in {RoadOption.CHANGELANELEFT, RoadOption.LEFT}: # turn left
            new_direction = 2
        
        elif direction in {RoadOption.CHANGELANERIGHT, RoadOption.RIGHT}:  # turn right
            new_direction = 3

        return new_direction
    

    def destroy(self):
        for x in self.sensors + self.measures:
            x.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in [self.vehicle]])


class CameraSensor:

    def __init__(self, world, vehicle, camera):
        self.world = world
        self.name = camera['name']
        self.type = camera['type']
        bp = self.world.get_blueprint_library().find(self.type)
        if self.type.startswith('sensor.camera'):
            self.width = camera['width']
            self.height = camera['height']
            self.fov = camera['fov']
            bp.set_attribute("image_size_x", str(self.width))
            bp.set_attribute("image_size_y", str(self.height))  # set resolution
            bp.set_attribute("fov", str(self.fov))

        elif self.type.startswith('sensor.lidar'):

            bp.set_attribute('range', str(camera['range']))
            bp.set_attribute('channels', str(camera['channels']))
            bp.set_attribute('upper_fov', str(camera['upper_fov']))
            bp.set_attribute('lower_fov', str(camera['lower_fov']))
            bp.set_attribute('rotation_frequency', str(camera['rotation_frequency']))
            bp.set_attribute('points_per_second', str(camera['points_per_second']))

        self.sensor = self.world.spawn_actor(bp, camera['pos'], attach_to= vehicle)
        self.convertor = camera['convertor']

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self, frame0):
        event = self.que.get()
        assert(event.frame == frame0) # ensure synchronous
        if self.type.startswith('sensor.camera'):
            event.convert(self.convertor)
            img = np.array(event.raw_data) # BGRA 32-bit pixels
            img = img.reshape((self.height, self.width, 4))[:, :, :3] # BGR
            return img
        elif self.type.startswith('sensor.lidar'):
            points = np.frombuffer(event.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            # print(points.shape)
            # print("frame:", event.frame)
            # print("channels:", event.channels)
            # print("horizontal_range:", event.horizontal_angle)
            # print("lidar:", event.transform.location.x, " ", event.transform.location.y, " ", event.transform.location.z)
            # print(type(points))
            # event.save_to_disk(l_dir+'/'+str(frame0)+'.ply')

            return points

    def destroy(self):
        self.sensor.destroy()


class CollisionSensor:

    def __init__(self, world, vehicle, name):
        self.world = world
        self.name = name

        self.history = 0.0 # the intensity of the collision, 0.0 means pristine
        self.other_actor = None

        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to= vehicle)

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)
    
    def get_data(self):
        """
        Returns:
            self.hitory -- the max collision intensity
            self.other_actor -- the bp of the collided actor
        """

        if not self.que.empty():
            event = self.que.get()
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2) # calc the intensity

            if intensity > self.history:
                self.history = intensity
                self.other_actor = event.other_actor.type_id

        return (self.history, self.other_actor)

    def destroy(self):
        self.sensor.destroy()


class LaneInvasionSensor:

    def __init__(self, world, vehicle, name):
        self.world = world
        self.name = name

        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to= vehicle)

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self):
        """
        Returns:
            set() -- a set contains all the lane that invaded
        """
        
        if self.que.empty():
            return set()
        
        event = self.que.get()
        return set(x.type for x in event.crossed_lane_markings)

    def destroy(self):
        self.sensor.destroy()