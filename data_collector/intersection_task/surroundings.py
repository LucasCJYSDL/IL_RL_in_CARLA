import glob
import os
import sys
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from config import CrossingRate


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


class Walkers:

    def __init__(self, world, client, config):
        self.world = world
        self.client = client
        self.number = config['NumOfWal']
        self.AREA = config['PED_AREA']
        self.config = config

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
        wals = [x for x in self.walkers if INSIDE(x.get_location(), self.config["AREA"])]
        print('Available Walker: ', len(wals), ' number: ', self.number)

    def destroy(self):
        for x in self.controllers:
            x.stop()
        
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.controllers])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walkers])

        self.walkers = []
        self.controllers = []

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(4.0)
    world = client.get_world()

    setting = world.get_settings()
    world.apply_settings(
        carla.WorldSettings(  # set synchronous mode
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1. / 20)
    )

    from config import GetConfig

    configs = GetConfig(0, 0)
    world.set_weather(configs['weather'][0])
    walkers = Walkers(world, client, configs)
    world.tick()
    walkers.start()
    #wait for the people to go to the road
    for _ in range(250):
        world.tick()
    start_frame = world.tick()
    print("start_frame: ", start_frame)

    for i in range(3000):
        walkers.step()
        # if vehicle_actor.is_at_traffic_light():
        #     traffic_light = vehicle_actor.get_traffic_light()
        #     if traffic_light.get_state() == carla.TrafficLightState.Red:
        #         # world.hud.notification("Traffic light changed! Good to go!")
        #         traffic_light.set_state(carla.TrafficLightState.Green)
        frame = world.tick()
        print("current_frame: ", frame)

    walkers.destroy()

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')

