import glob
import os
import sys
import math
import random
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


from env.sensors import camera_list, CameraSensor, CollisionSensor, LaneInvasionSensor
from env.utils import draw_waypoint_union, get_closet_wp, get_polygan

from env.PID.agent.basic_agent import BasicAgent
from env.PID.planner.local_planner import RoadOption



class EgoCar:

    def __init__(self, world, client, scene, waypoints):
        self.world = world
        self.client = client
        self.scene = scene
        self.waypoints = waypoints

        self.start = self.waypoints.get_transform(self.scene['start'])
        self.end = self.waypoints.get_transform(self.scene['end'])

        bp = self.world.get_blueprint_library().filter("vehicle.audi.a2")[0] 
        self.vehicle = self.world.spawn_actor(bp, self.start)

        self.agent = BasicAgent(self.vehicle)
        self.world.tick()

        self.path = self.agent.set_destination(self.end.location, draw_navi=False)
 
        self.sensors = [] # camera, Lidar, collision, lane invasion
        self.control = None


    def set_sensors(self):
        for camera in camera_list:
            self.sensors.append(CameraSensor(self.world, self.vehicle, camera))

        self.sensors.append(CollisionSensor(self.world, self.vehicle, 'Collision'))
        self.sensors.append(LaneInvasionSensor(self.world, self.vehicle, 'LaneInvasion'))


    def step(self, control, lateral, longitude, debug=False):
        # change traffic light green
        lights_list = self.world.get_actors().filter("*traffic_light*")
        for light in lights_list:
            light.set_state(carla.TrafficLightState.Green)

        # control
        self.control = control
        PID_control = self.agent.run_step(debug=False)

        if lateral == 'PID' or lateral == 'PID_NOISE':
            self.control.steer = PID_control.steer
        
        if longitude == 'PID':
            self.control.throttle = PID_control.throttle
            self.control.brake = PID_control.brake

        if lateral == 'PID_NOISE':  # add noise when collecting data 
            noise_control = self.control

            if random.randint(0, 10) == 0:   # 1/10 probability
                noise_control.steer = np.clip(self.control.steer + random.uniform(-0.2, 0.2), -1., 1.)

            self.vehicle.apply_control(noise_control)
        else:
            self.vehicle.apply_control(self.control)
        
        if debug:
            print("Control: ", self.control.steer, self.control.throttle - self.control.brake)


    def get_sensors(self, frame0, debug=False):
        data = {sensor.name : sensor.get_data(frame0) for sensor in self.sensors}

        v = self.vehicle.get_velocity()
        data['speed'] = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2) # km/h

        current = self.vehicle.get_transform()
        data['location'] = current.location

        dx = self.end.location.x - current.location.x
        dy = self.end.location.y - current.location.y
        data['angle_diff'] = math.radians(math.degrees(math.atan2(dy, dx)) - current.rotation.yaw)

        tot_x = self.end.location.x - self.start.location.x
        tot_y = self.end.location.y - self.start.location.y
        data['dis_diff'] = math.sqrt(dx**2 + dy**2) / math.sqrt(tot_x**2 + tot_y**2)

        if self.control:
            data['control'] = self.control
        else:
            data['control'] = self.vehicle.get_control()

        neighbor_wp, next_wp, min_dis = get_closet_wp(self.path, current.location, self.world.debug, draw_dis=False)
        data['min_dis'] = min_dis

        # Area
        #next_loc_yaw = carla.Location(x=current.location.x+math.cos(current.rotation.yaw), y=current.location.y+math.sin(current.rotation.yaw), z=current.location.z)
        #area = get_polygan(self.world.debug, current.location, next_loc_yaw, neighbor_wp.transform.location, next_wp.transform.location, draw_dis=True)
        #data['area'] = area
        #data['min_dis'] = area if min_dis > 0 else - area

        #self.world.debug.draw_point(neighbor_wp.transform.location, 0.1, carla.Color(0, 255, 0), 0.2, False)#green
        #self.world.debug.draw_point(next_wp.transform.location, 0.1, carla.Color(255, 0, 0), 0.2, False) #red
        #self.world.debug.draw_point(next_loc_yaw, 0.1, carla.Color(0, 0, 255), 0.2, False) #blue
        #self.world.debug.draw_point(current.location, 0.1, carla.Color(255, 162, 0), 0.2, False) #orange

        if debug:
            print('current yaw:', current.rotation.yaw)
            print('taget yaw:', math.degrees(math.atan2(dy, dx)))
            print('angle_diff', data['angle_diff'])

            print("minmum distance: ", min_dis)

            print('speed', data['speed'])
            print('dis_diff', data['dis_diff'])

        return data


    def destroy(self):
        for x in self.sensors:
            x.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in [self.vehicle]])
