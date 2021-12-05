import glob
import os
import sys
import random
import numpy as np

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

from env.agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from env.agents.navigation.global_route_planner import GlobalRoutePlanner, RoadOption

from env.functions import FPS
from env.functions import get_weather, WaypointsFromCSV, pre_process_image, pre_process_lidar, pre_process_measure, get_area, INSIDE, get_task_type


class Env:

    def __init__(self, img_agent, port):
        print('# Initializing Env')

        self.img_agent = img_agent

        self.client = carla.Client("localhost", port)   # connect to server
        self.client.set_timeout(4.0)

        self.world = self.client.get_world()
        self.debug = self.world.debug

        self.waypoints = WaypointsFromCSV(os.path.join(os.path.dirname(__file__), 'waypoint.csv'))

        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
            no_rendering_mode= False,
            synchronous_mode= True,
            fixed_delta_seconds= 1. / FPS))
        
        #self.world.get_spectator().set_transform(...) 

        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1) #build global planner
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        self.vehicles = None
        self.walkers = None
        self.ego_car = None


    def reset(self, scene):
        """
        Returns:
            state {tuple (3, )} -- image_feature(512, ), speed(1,), image(88*200)
        """

        self.scene = scene
        self.area = get_area(scene)
        self.world.set_weather(get_weather(self.scene['weather']))
        
        self.branch = self.scene['branch']
        self.task_type = self.scene['task_type']

        self.ego_car = EgoCar(self.world, self.client, self.scene, self.waypoints, self.planner)
        self.vehicles = Vehicles(self.world, self.client, self.scene, self.waypoints, self.ego_car)
        self.world.tick()

        self.walkers = Walkers(self.world, self.client, self.scene, self.waypoints)
        self.world.tick()

        self.vehicles.start()
        self.walkers.start()

        for t in range(self.scene['Wait_ticks']):
            self.world.tick()
        
        self.ego_car.set_sensors() # sensor: camera, Lidar
        self.ego_car.set_measures() # measures: collison, location, lane invasion
        
        self.reward_ls = []
        self.stop_time = 0
        
        self.start_frame = self.world.tick()
        self.frame = self.start_frame
        self.success = False

        self.last_sensor_dict = self.ego_car.get_sensors(self.frame)
        self.last_measure_dict = self.ego_car.get_measures()
        return self.get_state(self.last_sensor_dict, self.last_measure_dict)
    

    def step(self, action):
        """
        Arguments:
            action {np.array, (3, )} -- (steer, throttle, brake) 
        
        Returns:
            state {tuple (3, )} -- image_feature(512, ), speed(1,), image(88*200)
            reward {int}
            done {bool}
        """

        assert action.shape == (2,)
        steer = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], 0.0, 1.0) if action[1] > 0.0 else 0.0
        brake = np.clip(abs(action[1]), 0.0, 1.0) if action[1] < 0.0 else 0.0

        control = carla.VehicleControl(steer = steer, throttle = throttle, brake = brake, reverse=False)

        self.ego_car.step(control)
        self.vehicles.step()
        self.walkers.step()

        self.frame = self.world.tick()
        self.last_sensor_dict = self.ego_car.get_sensors(self.frame)
        self.last_measure_dict = self.ego_car.get_measures()

        state = self.get_state(self.last_sensor_dict, self.last_measure_dict)
        reward, done = self.get_reward_done(self.last_measure_dict, control)

        self.reward_ls.append(reward)
        return state, reward, done


    def get_state(self, sensor_dict, measure_dict):
        img = pre_process_image(sensor_dict['FrontRGB'])
        img_feature = self.img_agent.compute_feature(img)
        lidar_feature = pre_process_lidar(sensor_dict['Lidar'])
        measure = pre_process_measure(measure_dict, self.waypoints.loc[self.scene['ped_center']], self.scene['lane_type'])
        return (img_feature, lidar_feature, measure, img) # speed (1, ) lidar_feature (360, )


    def get_reward_done(self, measure_dict, control):
        speed = measure_dict['speed'] # km/h
        loc = measure_dict['location']
        steer = control.steer

        r_v = - min(2, max(0, speed - 15) / 5)# [0, 2]
        r_s = - 2 * max(0, steer * steer - 0.3) # [0, 2]

        r_c = - min(30, measure_dict['Collision'][0]) # [0, 20]
        r_o = - min(5, 5 * len(measure_dict['LaneInvasion'])) # [0, 5]

        reward = r_v + r_s + r_c + r_o - 0.1
        done = False

        if speed < 1.0:
            self.stop_time += 1
        else:
            self.stop_time = 0
        
        #if self.stop_time >= 200:
        #    done = True

        if measure_dict['Collision'][0] > 0.0:
            done = True

        if not INSIDE(loc, self.area):
            done = True
            task_type = get_task_type(loc, self.area, self.task_type)

            if self.branch == task_type:
                reward += 40 # success
                self.success = True
            else:
                reward -= 10 # fail
        
        return reward, done


    def destroy(self):
        for x in [self.vehicles, self.walkers, self.ego_car]:
            if x:
                x.destroy()
        self.ego_car = self.vehicles = self.walkers = None


    def close(self, mode):
        self.destroy()

        res = OrderedDict()
        res[mode + '_frame'] = self.frame - self.start_frame
        res[mode + '_success'] = self.success
        res[mode + '_collison'] = self.last_measure_dict['Collision'][0] > 0.0
        res[mode + '_tot_reward'] = sum(self.reward_ls)
        return res


    def __del__(self):
        self.destroy()
        self.world.apply_settings(self._settings)



'''
def random_control(self):
    controls = {  # steer, throttle, brake
        'forward': (0, 1, 0),
        'stay': (0, 0, 0),
        'back': (0, 0, 0.5),
        'left': (-1, 0.5, 0),
        'right': (1, 0.5, 0)
    }
    #controls_name = ['stay', 'forward']
    controls_name = ['forward', 'stay', 'back', 'left', 'right']

    return [np.array(controls[random.choice(controls_name)])]

def main():
    env = Env()

    for episode in range(2): 
        sensors, measures = env.reset()

        for i in range(2000): 
            print('=========== Episode %d, Frame %d : =========' % (episode, i))
            action = env.random_control()
            (sensor_dict, measure_dict), reward, done = env.step(action[0])

            if RENDER:
                for x in sensor_dict:
                    render_image(x, sensor_dict[x])
                print('Measures: ', measure_dict)
                print('Reward: ', reward, 'Done: ', done)

            #if done:
                #break
        
        env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
'''