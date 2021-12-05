import glob
import os
import sys
import random
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from env.utils import CrossingRate, SafyDis, get_angle
from env.utils import INSIDE, get_area, get_PED_area, get_intention, draw_area,get4areas


class Vehicles:

    def __init__(self, world, client, scene, ego_car):
        self.world = world
        self.client = client
        self.scene = scene
        self.ego_car = ego_car

        self.number = scene['NumOfVeh']
        self.bps = self.world.get_blueprint_library().filter('vehicle.*.*')
        self.bps = [x for x in self.bps if int(x.get_attribute('number_of_wheels')) == 4]

        self.spawns = self.world.get_map().get_spawn_points()
        self.spawns = [x for x in self.spawns if self.ego_car.start.location.distance(x.location) > SafyDis]

        self.AREA = get_area(self.scene)

        inner_spawns = [x for x in self.spawns if INSIDE(x.location, self.AREA)]
        outer_spawns = [x for x in self.spawns if not INSIDE(x.location, self.AREA)]
        density = random.randint(self.number[0], self.number[1]) / len(inner_spawns)

        self.vehicles = []
        self.generate_vehicles(inner_spawns, round(density * len(inner_spawns)))
        self.generate_vehicles(outer_spawns, round(density * len(outer_spawns)))

        self.world.tick()

    
    def generate_vehicles(self, spawns, total):
        random.shuffle(spawns)

        for i in range(total):
            bp = random.choice(self.bps)
            vehicle = self.world.spawn_actor(bp, spawns[i])
            self.vehicles.append(vehicle)
    

    def start(self):
        for x in self.vehicles:
            x.set_autopilot(True)
        self.client.apply_batch_sync([carla.command.SetAutopilot(x, True) for x in self.vehicles])
    
    
    def step(self, debug=False):
        self.start()
        if debug:
            vehs = [x for x in self.vehicles if INSIDE(x.get_location(), self.AREA)]
            print('Available Car: ', len(vehs), ' number: ', self.number)
    

    def destroy(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])
        self.vehicles = []



class Walkers:

    def __init__(self, world, client, scene, waypoints, ego_car):
        self.world = world
        self.client = client
        self.scene = scene
        self.waypoints = waypoints
        self.ego_car = ego_car

        self.number = scene['NumOfWal']
        self.AREA = get_PED_area(self.scene, self.waypoints)
        self.areas=get4areas(self.AREA)

        self.bps = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        self.control_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        self.walkers = []
        self.controllers = []
        self.generate_walkers(random.randint(self.number[0], self.number[1]))

        self.world.set_pedestrians_cross_factor(CrossingRate)
        self.world.tick()
        
    
    def generateInOneArea(self, total, area):
        controllers=[]
        for i in range(total):
            bp = random.choice(self.bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'false')

            while True:
                location = self.world.get_random_location_from_navigation()
                location.x=random.uniform(area['MINX'],area['MAXX'])
                location.y=random.uniform(area['MINY'],area['MAXY'])
                if not INSIDE(location,area):
                    continue
                    
                spawn_point = carla.Transform()
                spawn_point.location = location
                walker = self.world.try_spawn_actor(bp, spawn_point)

                if walker is not None:
                    break
            
            controller = self.world.spawn_actor(self.control_bp, carla.Transform(), walker)
            self.walkers.append(walker)
            controllers.append(controller)
        self.controllers.append(controllers)

    
    def generate_walkers(self, total):
        for i in range(4):
            self.generateInOneArea(math.ceil(total/4), self.areas[i])
    
    def start(self):
        self.direction_AtoB_start(0,1)
        self.direction_AtoB_start(2,3)

        #draw_area(self.world.debug,self.areas[0])
        #draw_area(self.world.debug,self.areas[1])
        #draw_area(self.world.debug,self.areas[2])
        #draw_area(self.world.debug,self.areas[3])


    def direction_AtoB_start(self,A,B):
        
        controllerA=self.controllers[A]
        controllerB=self.controllers[B]
        #print(len(controllerA),len(controllerB))
        areaA=self.areas[A]
        areaB=self.areas[B]
        #print(areaA,areaB)
        for controller in controllerA:
            location = self.world.get_random_location_from_navigation()

            location.x=random.uniform(areaB['MINX'],areaB['MAXX'])
            location.y=random.uniform(areaB['MINY'],areaB['MAXY'])
            
            controller.start()
            controller.go_to_location(location)
            controller.set_max_speed(1.3 + random.random() * 0.5)

        for controller in controllerB:
            location = self.world.get_random_location_from_navigation()
            location.x=random.uniform(areaA['MINX'],areaA['MAXX'])
            location.y=random.uniform(areaA['MINY'],areaA['MAXY'])

            controller.start()
            controller.go_to_location(location)
            controller.set_max_speed(1.3 + random.random() * 0.5)


    def step(self, debug=False):
        if debug:
            wals = [x for x in self.walkers if INSIDE(x.get_location(), self.AREA)]
            print('Available Walker: ', len(wals), ' number: ', self.number)


    def get_disruption(self):
        cnt = 0
        for walker in self.walkers:
            ego_loc = self.ego_car.vehicle.get_location()
            other_loc = walker.get_location()
            dis = ego_loc.distance(other_loc)
            flag=0
            if dis < 5.0:
                angle = get_angle([other_loc.x-ego_loc.x, other_loc.y-ego_loc.y])
                acce = walker.get_acceleration()
                vel = walker.get_velocity()
                vec_1 = [vel.x, vel.y]
                vec_2 = [other_loc.x-ego_loc.x, other_loc.y-ego_loc.y]
                intention = get_intention(vec_1, vec_2)
                #print("intention: ", intention)
                if (abs(angle - 180) <= 45 or angle <= 45 or angle >= 315) and dis < 2.5:
                    #self.world.debug.draw_point(walker.get_location() + carla.Location(z=0.25), 0.1, carla.Color(0, 255, 0), 1.0, False)
                    flag=1
                if (abs(angle-180)<=45 or angle<=45 or angle>=315) and intention<-0.6: #beside the eg0-vehicle
                    # print("distance: ", ego_loc.distance(other_loc))
                    # if abs(acce.x)>1.5 or abs(acce.y)>1.5 or math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)<0.1:
                    if abs(acce.x) > 1.5 or abs(acce.y) > 1.5:
                        flag=1
                        # print("close acceleration: (%.2f, %.2f, %.2f)" % (acce.x, acce.y, acce.z))
                        # self.world.debug.draw_point(walker.get_location() + carla.Location(z=0.25), 0.1, carla.Color(0, 0, 255), 1.0, False)
                    if math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)<0.1:
                        flag=1
                    #     print("close velocity: (%.2f, %.2f, %.2f)" % (vel.x, vel.y, vel.z))
                        #self.world.debug.draw_point(walker.get_location() + carla.Location(z=0.25), 0.1, carla.Color(255, 0, 0), 1.0, False)
            cnt+=flag
        return cnt


    def destroy(self):
        for controllers in self.controllers:
            for x in controllers:
                x.stop()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in controllers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])

        self.walkers = []
        self.controllers = []