import glob
import os
import sys
import random
import numpy as np

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

from env.ego_car import EgoCar
from env.surroundings import Vehicles, Walkers

from env.utils import Waypoints, get_weather, get_area, INSIDE, get_task_type
from env.utils import FPS, draw_area, render_LIDAR, pre_process_lidar


class Env:

    def __init__(self, port, debug=False):
        print('# Initializing Env')

        self.client = carla.Client("localhost", port)   # connect to server
        self.client.set_timeout(4.0)

        self.world = self.client.get_world()
        self._settings = self.world.get_settings()

        self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
            no_rendering_mode= False,
            synchronous_mode= True,
            fixed_delta_seconds= 1. / FPS))

        self.waypoints = Waypoints(os.path.join(os.path.dirname(__file__), 'waypoint.csv'))
        if debug:
            print('# waypoints number = ', self.waypoints.cnt)
            self.waypoints.render(self.world.debug)

        self.ego_car = self.vehicles = self.walkers = None


    def reset(self, scene, debug=False, draw_area=False):
        self.scene = scene
        self.area = get_area(self.scene)
        self.world.set_weather(get_weather(self.scene['weather']))

        self.ego_car = EgoCar(self.world, self.client, self.scene, self.waypoints)
        self.vehicles = Vehicles(self.world, self.client, self.scene, self.ego_car)
        self.walkers = Walkers(self.world, self.client, self.scene, self.waypoints, self.ego_car)

        if debug:
            spector = self.waypoints.get_transform(scene['ped_center'])
            spector.location.z = 30
            spector.rotation.pitch = -90
            self.world.get_spectator().set_transform(spector)

        self.vehicles.start()
        self.walkers.start()

        for _ in range(self.scene['Wait_ticks']):
            self.world.tick()
        
        self.ego_car.set_sensors()  # sensor: camera, Lidar, collison, lane invasion, info
        self.frame = self.start_frame = self.world.tick()

        self.reset_metrics()
        data = self.ego_car.get_sensors(self.frame)
        state = self.get_state(data)
        info = self.get_info(data, state)

        return state, info


    def reset_metrics(self):
        self.res = OrderedDict()

        self.res['success'] = False
        self.res['time_out'] = False
        self.res['lane_invasion'] = False
        self.res['collision'] = False
        self.res['invasion_time'] = 0

        self.res['total_ego_jerk'] = 0
        #self.res['mean_ego_jerk'] = 0.0

        self.res['total_other_jerk'] = 0
        #self.res['mean_other_jerk'] = 0.0

        self.res['total_min_dis'] = 0.0
        #self.res['mean_min_dis'] = 0.0


    def step(self, action, lateral, longitude):
        assert action.shape == (2,)
        steer = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], 0.0, 1.0) if action[1] > 0.0 else 0.0
        brake = np.clip(abs(action[1]), 0.0, 1.0) if action[1] < 0.0 else 0.0
        control = carla.VehicleControl(steer = steer, throttle = throttle, brake = brake, reverse=False)

        self.ego_car.step(control, lateral, longitude) 

        self.vehicles.step()
        self.walkers.step()
        self.frame = self.world.tick()

        data = self.ego_car.get_sensors(self.frame)
        state = self.get_state(data)
        info = self.get_info(data, state)
        reward, done = self.get_reward_done(data, info)  # update self.res

        return state, reward, done, info


    def get_state(self, data, debug=False):
        """return (image, lidar, measure)"""

        image = data['FrontSemantic']
        image = image[115 : 510, :]
        image = np.array(Image.fromarray(image).resize((200, 88))) #haven't / 255, in order to save menmory

        points = data['Lidar'][0]
        lidar, lidar_raw = pre_process_lidar(points)
        if debug:
            render_LIDAR(points, data['Lidar'][1], self.world.debug)

        measure = []
        measure.append(data['speed'] / 30.0)
        measure.append(data['min_dis'])
        measure.append(data['angle_diff'] / 1.57)
        measure.append(data['dis_diff'])
        measure.extend(self.scene['lane_type']) #(3,)
        measure = np.array(measure)

        self.res['total_min_dis'] += abs(data['min_dis'])

        return (image, lidar, measure)


    def get_info(self, data, state):
        info = {}
        info['big_semantic'] = data['FrontSemantic']
        info['small_semantic'] = state[0] # 88 * 200
        info['FrontRGB'] = data['FrontRGB']

        info['a_t'] = [data['control'].steer, data['control'].throttle - data['control'].brake]
        return info


    def get_reward_done(self, data, info):
        reward = []
        done = False

        reward.append(data['Collision'][0] > 0.0)
        reward.append(len(data['LaneInvasion']) > 0)

        if data['Collision'][0] > 0.0:
            done = True
            self.res['collision'] = True
        
        if len(data['LaneInvasion']) > 0:
            self.res['invasion_time'] += 1
            if self.res['invasion_time'] >= 5:  #lane invasion for too many times
                done = True
                self.res['lane_invasion'] = True


        reward.append(0) # don't success
        if not INSIDE(data['location'], self.area):
            done = True
            task_type = get_task_type(data['location'], self.area, self.scene['task_type'])

            if self.scene['branch'] == task_type:
                self.res['success'] = True
                reward[-1] = 1 # success


        # Comfort
        if abs(info['a_t'][0]) > 0.9:
            self.res['total_ego_jerk'] += 1
        
        if abs(info['a_t'][1]) > 0.9:
            self.res['total_ego_jerk'] += 1
        
        self.res['total_other_jerk'] += self.walkers.get_disruption()

        return np.array(reward), done # rwd: (3,), done: bool


    def destroy(self):
        for x in [self.vehicles, self.walkers, self.ego_car]:
            if x:
                x.destroy()
        self.ego_car = self.vehicles = self.walkers = None


    def close(self, expected_end_steps):
        self.destroy()

        self.res['tot_step'] = self.frame - self.start_frame
        if self.res['tot_step'] == expected_end_steps:
            self.res['time_out'] = True

        #self.res['mean_ego_jerk'] = self.res['total_ego_jerk'] / self.res['tot_step']
        #self.res['mean_other_jerk'] = self.res['total_other_jerk'] / self.res['tot_step']
        #self.res['mean_min_dis'] = self.res['total_min_dis'] / self.res['tot_step']
        
        return self.res


    def __del__(self):
        self.destroy()
        self.world.apply_settings(self._settings)
