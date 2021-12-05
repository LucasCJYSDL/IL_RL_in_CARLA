import glob
import os
import sys
import random
import time
import cv2
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from ego_car import EgoCar
from surroundings import Vehicles, Walkers

from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.global_route_planner import GlobalRoutePlanner, RoadOption

from config import ExpNames, GetConfig
from config import WaitTick, RENDER, FPS, DisNearTarget


def render_image(name, img):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(1)


class Carla97_Env:

    def __init__(self):
        print("Connecting to server.")

        self.client = carla.Client("localhost", 2000)   # connect to server
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()

        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
            no_rendering_mode= False,
            synchronous_mode= True,
            fixed_delta_seconds= 1. / FPS))

        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1) #build global planner
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        self.ego_car = None
        self.vehicles = None
        self.walkers = None

    def reset(self):
        print("\nReset the env.")

        self.exp_name = random.choice(ExpNames)
        self.config = GetConfig(self.exp_name)
        print(self.config)

        self.world.set_weather(self.config['weather'][0]) # set weather
        self.world.get_spectator().set_transform(self.config['spectator'])

        self.ego_car = EgoCar(self.world, self.config, self.planner)
        self.vehicles = Vehicles(self.world, self.client, self.config, self.ego_car)
        
        for t in range(WaitTick): # wait until the car reaches the ground
            self.world.tick()

        self.walkers = Walkers(self.world, self.client, self.config)
        self.world.tick()

        self.vehicles.start()
        self.walkers.start()
        self.world.tick()
        
        self.ego_car.set_sensors() # sensor: camera, Lidar
        self.ego_car.set_measures() # measures: collison, location, lane invasion
        
        self.start_frame = self.world.tick()
        self.frame = self.start_frame
        return self.ego_car.get_sensors(self.start_frame), self.ego_car.get_measures()

    def step(self, action):
        ''' action(tuple)  (steer, throttle, brake) '''

        control = carla.VehicleControl(throttle = action[1], steer = action[0], brake = action[2], reverse=False)
        
        self.ego_car.step(control)
        self.vehicles.step()
        self.walkers.step()

        self.frame = self.world.tick()
        sensor_dict = self.ego_car.get_sensors(self.frame)
        measure_dict = self.ego_car.get_measures()

        return sensor_dict, measure_dict, self.get_reward(measure_dict, control), self.get_done(measure_dict)

    def get_done(self, measure_dict):
        if measure_dict['Collision'][0] > 0.0 or measure_dict['distance'] < DisNearTarget:
            return True
        return False
  
    def get_reward(self, measure_dict, control):
        direction = measure_dict['direction']
        speed = measure_dict['speed']
        reward = 0

        if direction in {RoadOption.STRAIGHT}: # go straight
            if abs(control.steer) > 0.2:
                reward -= 20
            reward += min(35, speed)

        elif direction in {RoadOption.VOID, RoadOption.LANEFOLLOW} :  # follow lane
            reward += min(25, speed)
    
        elif direction in {RoadOption.CHANGELANELEFT, RoadOption.LEFT}: # turn left
            if control.steer > 0:
                reward -= 15
            if speed <= 20:
                reward += speed
            else:
                reward += 40 - speed
        
        elif direction in {RoadOption.CHANGELANERIGHT, RoadOption.RIGHT}:  # turn right
            if control.steer < 0:
                reward -= 15
            if speed <= 20:
                reward += speed
            else:
                reward += 40 - speed
        
        reward -= 100 * len(measure_dict['LaneInvasion'])
        reward -= min(200, measure_dict['Collision'][0])

        if measure_dict['distance'] < DisNearTarget:
            reward += 200

        return reward

    def random_action(self):
        controls = {  # steer, throttle, brake
            'forward': (0, 1, 0),
            'stay': (0, 0, 0),
            'back': (0, 0, 1),
            'left': (-1, 0.5, 0),
            'right': (1, 0.5, 0)
        }
        controls_name = ['stay', 'forward']
        #controls_name = ['forward', 'stay', 'back', 'left', 'right']

        return controls[random.choice(controls_name)]

    def destroy(self):
        for x in [self.vehicles, self.walkers, self.ego_car]:
            if x:
                x.destroy()
        
        self.ego_car = None
        self.vehicles = None
        self.walkers = None
        print("destroy successfully")

    def close(self):
        self.destroy()
        print(self.config)

    def __del__(self):
        self.destroy()
        self.world.apply_settings(self._settings)


def main():
    env = Carla97_Env()

    for episode in range(2): 
        sensors, measures = env.reset()

        for i in range(2000): 
            print('=========== Episode %d, Frame %d : =========' % (episode, i))
            action = env.random_action()
            sensor_dict, measure_dict, reward, done = env.step(action)

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