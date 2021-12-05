#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)

"""
This module implements Carla Environment to interact with carla server.
"""
import os
import sys
import random
#import cv2
import numpy as np

import carla
from agent.basic_agent import BasicAgent
from surroundings import Vehicles, Walkers
from ego_car import EgoCar
from planner.local_planner import RoadOption
import config



class CarlaEnv:

    """
    Class offers API to interact with carla server, also deals with controls and measures of ego-car
    """

    def __init__(self):
        """
        Constructor
        """
        # connect to carla server
        self.client = carla.Client("localhost", 2000)

        assert self.client != None

        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        
        assert self.world != None

        self.debug = self.world.debug
        self._settings_bak = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode = False, 
            synchronous_mode = True,
            fixed_delta_seconds = 1.0 / config.FPS))
        
        # ego car object
        self.ego_car = None

        # class object for managing walkers in carla world
        self.walkers = None

        # class object for managing vehicles in carla world
        self.vehicles = None

        self.exp_name = None
        self.config = None
        
        self.start_frame = None
        self.frame = None

    def reset(self):
        """
        Reset the environment.
        This function is gym-like, reset the environment and returns the (initial) observation
        
        :return: sensors observation and measurements
        """
        print("\nReset the env.")

        self.exp_name = random.choice(config.ExpNames)
        self.config = config.GetConfig(self.exp_name)
        print("Config:\n", self.config)

        # set environment weather
        self.world.set_weather(self.config['weather'][0])
        self.world.get_spectator().set_transform(self.config['spectator'])

        # EgoCar object
        self.ego_car = EgoCar(self.world, self.config)

        # Note that we need to pass ego_car to spare safe distance between ego_car and the environment cars to be spawned
        self.vehicles = Vehicles(self.world, self.client, self.config, self.ego_car)

        # :) wait until the spawned cars reach the ground
        # TODO: check if exist sideffect ? the agent of ego car may already drive this ego car for several timestamps
        for t in range(config.WaitTick):
            self.world.tick()

        self.walkers = Walkers(self.world, self.client, self.config)
        self.world.tick()

        # set environment cars to start engine (autopilot)
        self.vehicles.start()
        # set walkers to move
        self.walkers.start()
        self.world.tick()

        self.ego_car.set_sensors()
        self.ego_car.set_measures()

        self.start_frame = self.world.tick()
        self.frame = self.start_frame

        # now return the observation
        # TODO: check the nonsync problem of cameras and measures?
        # so far we pass the frame parameter to ensure that the observed data of cameras is synchronous
        sensor_dict = self.ego_car.get_sensors(self.frame)
        measure_dict = self.ego_car.get_measures() 
        return sensor_dict, measure_dict

    def step(self, action, use_network_control = False):
        """
        Step, forward/simulate the environment.
        This function is gym-like, apply action and simulate the environment for one step and returns the observation
        
        :param action: action tuple (steer, throttle, brake)
        :param use_network_control: boolean flag of whether using network output or the artificial agent control
        :return:  sensors observation and measurements
        """
        control = carla.VehicleControl(
                                    throttle = action[1],
                                    steer = action[0],
                                    brake = action[2],
                                    reverse = False)
        if use_network_control:
            self.ego_car.apply_control(control)
        else:
            # we use artificial agent's control output
            self.ego_car.step(debug = True)
        
        # environment step
        self.vehicles.step()
        self.walkers.step()

        # tick !
        self.frame = self.world.tick()

        # now return the observation
        # TODO: check the nonsync problem of cameras and measures?
        # so far we pass the frame parameter to ensure that the observed data of cameras is synchronous
        sensor_dict = self.ego_car.get_sensors(self.frame)
        measure_dict = self.ego_car.get_measures() 
        return sensor_dict, measure_dict, self.get_reward(measure_dict, control), self.get_done(measure_dict)    
    
    def get_done(self, measure_dict):
        if measure_dict['Collision'][0] > 0.0 or measure_dict['distance'] < config.DisNearTarget:
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

        if measure_dict['distance'] < config.DisNearTarget:
            reward += 200

        return reward

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
        self.world.apply_settings(self._settings_bak)


def render_image(name, img):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(1)

def main():
    env = CarlaEnv()
    print("constructed")
    for episode in range(5):
        sensors, measures = env.reset()
        for i in range(2000):
            print('=========== Episode %d, Frame %d : =========' % (episode, i))

            # since we don't use network's output as action, just use dummy action here
            action = [0., 0., 0.]
            sensor_dict, measure_dict, reward, done = env.step(action)

            if config.RENDER:
                import cv2
                for x in sensor_dict:
                    render_image(x, sensor_dict[x])
            print('Measures: ', measure_dict)
            print('Reward: ', reward, 'Done: ', done) 

            if done:
                break
        env.close()              

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit by user.")
    finally:
        print("\nExit.")