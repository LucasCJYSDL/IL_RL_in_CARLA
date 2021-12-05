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

from config import CAMERAS, DrawNavi
from agents.navigation.global_route_planner import RoadOption


def draw_waypoint_union(debug, w0, w1, color=carla.Color(0, 0, 255), lt=5):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, False)


class EgoCar:

    def __init__(self, world, config, planner):
        self.world = world
        self.start = config['start']
        self.end = config['end']
        self.planner = planner
        self.debug = self.world.debug

        bp = self.world.get_blueprint_library().filter("vehicle.audi.a2")[0] 
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

        current = self.vehicle.get_location()
        measure_dict['distance'] = current.distance(self.end.location)
        measure_dict['direction'] = self.get_direction(current, self.end.location)

        return measure_dict

    def get_direction(self, cur_loc, tar_loc):
        plan = self.planner.trace_route(cur_loc, tar_loc)
        direction = plan[0][1]

        if DrawNavi:
            for i in range(len(plan) - 1):
                w0 = plan[i][0]
                w1 = plan[i+1][0]
                draw_waypoint_union(self.debug, w0, w1)

        return direction
    
    def destroy(self):
        for x in self.sensors + self.measures:
            x.destroy()
        self.vehicle.destroy()


class CameraSensor:

    def __init__(self, world, vehicle, camera):
        self.world = world
        self.name = camera['name']
        self.width = camera['width']
        self.height = camera['height']

        bp = self.world.get_blueprint_library().find(camera['type']) 
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))  # set resolution

        self.sensor = self.world.spawn_actor(bp, camera['pos'], attach_to= vehicle)
        self.convertor = camera['convertor']

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self, frame0):
        event = self.que.get()
        assert(event.frame == frame0) # ensure synchronous

        event.convert(self.convertor)
        img = np.array(event.raw_data) # BGRA 32-bit pixels
        img = img.reshape((self.height, self.width, 4))[:, :, :3] # BGR
        return img

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
        '''
        :return: max collision intensity (float), other_actor (e.g static.builging, vehicle.*.*, walker.*.*)
        '''

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
        '''
        :return:    a set of the lanes invaded by the ego-car
        '''

        if self.que.empty():
            return set()
        
        event = self.que.get()
        return set(x.type for x in event.crossed_lane_markings)

    def destroy(self):
        self.sensor.destroy()