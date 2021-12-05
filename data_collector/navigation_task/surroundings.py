import glob
import os
import sys
import random
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

from config import WalkerSpawnRate, CrossingRate, EnableTwoWheel, SafyDis, DrawArea, WaitTick


def INSIDE(loc, AREA):
    return (loc is not None) and loc.x > AREA['MINX'] and loc.x < AREA['MAXX'] and \
        loc.y > AREA['MINY'] and loc.y < AREA['MAXY'] #and loc.z < AREA['HEIGHT']


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


class Vehicles:

    def __init__(self, world, client, config, ego_car):
        self.world = world
        self.client = client
        self.number = config['NumOfVeh']
        self.AREA = config['AREA']
        self.ego_car = ego_car

        if DrawArea:
            draw_area(self.world.debug, self.AREA)

        self.bps = self.world.get_blueprint_library().filter('vehicle.*.*')
        if not EnableTwoWheel:
            self.bps = [x for x in self.bps if int(x.get_attribute('number_of_wheels')) == 4]

        self.spawns = self.world.get_map().get_spawn_points()
        self.spawns = [x for x in self.spawns if self.ego_car.start.location.distance(x.location) > SafyDis]

        inner_spawns = [x for x in self.spawns if INSIDE(x.location, self.AREA)]
        outer_spawns = [x for x in self.spawns if not INSIDE(x.location, self.AREA)]
        density = (self.number[0] + self.number[1]) / 2 / len(inner_spawns)

        self.vehicles = []
        self.generate_vehicles(inner_spawns, round(density * len(inner_spawns)))
        self.generate_vehicles(outer_spawns, round(density * len(outer_spawns)))

    
    def generate_vehicles(self, spawns, total):
        random.shuffle(spawns)

        for i in range(total):
            bp = random.choice(self.bps)
            vehicle = self.world.spawn_actor(bp, spawns[i])
            self.vehicles.append(vehicle)
    
    def start(self):
        for x in self.vehicles:
            x.set_autopilot(True)
        #self.client.apply_batch_sync([carla.command.SetAutopilot(x, True) for x in self.vehicles])
    
    
    def step(self):
        self.start()
        vehs = [x for x in self.vehicles if INSIDE(x.get_location(), self.AREA)]

        print('Available Car: ', len(vehs), ' number: ', self.number)

        '''
        if len(inner_vehs) < self.number[0]:
            #self.destroy()
            self.generate_vehicles(1) #self.number[1] - len(inner_vehs))
            
            if outer_vehs == []:
                self.generate_vehicles(self.outer_spawns, len(self.outer_spawns))
                self.last_supply = frame

            elif frame % UpdateFrame == 0:
                self.vehicles = [x for x in self.vehicles if INSIDE(x.get_location(), self.inner)]
                self.client.apply_batch([carla.command.DestroyActor(x) for x in outer_vehs])
            else:
                print('hhhh')
        '''

    def destroy(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])
        self.vehicles = []


class Walkers:

    def __init__(self, world, client, config):
        self.world = world
        self.client = client
        self.number = config['NumOfWal']
        self.AREA = config['AREA']

        self.bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.control_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        self.walkers = []
        self.controllers = []
        self.generate_walkers(self.number[1])

        self.world.set_pedestrians_cross_factor(CrossingRate)
    
    def generate_walkers(self, total):
        for i in range(total):
            bp = random.choice(self.bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            while True:
                location = self.world.get_random_location_from_navigation()
                if not INSIDE(location, self.AREA):
                    continue
            
                spawn_point = carla.Transform()
                spawn_point.location = location
                walker = self.world.try_spawn_actor(bp, spawn_point)

                if walker is not None:
                    break
            
            controller = self.world.spawn_actor(self.control_bp, carla.Transform(), walker)
            self.walkers.append(walker)
            self.controllers.append(controller)
    
    def start(self):
        for controller in self.controllers:
            while True:
                location = self.world.get_random_location_from_navigation()
                if INSIDE(location, self.AREA):
                    break

            controller.start()
            controller.go_to_location(location)
            controller.set_max_speed(1.3 + random.random() * 0.5)

    def step(self):
        wals = [x for x in self.walkers if INSIDE(x.get_location(), self.AREA)]
        print('Available Walker: ', len(wals), ' number: ', self.number)

    def destroy(self):
        for x in self.controllers:
            x.stop()
        
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])

        self.walkers = []
        self.controllers = []