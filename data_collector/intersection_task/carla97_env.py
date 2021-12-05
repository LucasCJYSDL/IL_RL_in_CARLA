import glob
import os
import sys
import math
# import cv2
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
from surroundings import Walkers, INSIDE
from config import GetConfig, RENDER, FPS, Experiments, Scene_num
from utils import write_hdf5, mkdir, pre_process, draw_waypoint_union, get_distance, get_angle, sensor_to_world

from agent.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agent.navigation.global_route_planner import GlobalRoutePlanner, RoadOption

#
# def render_image(name, img):
#     cv2.namedWindow(name)
#     cv2.imshow(name, img)
#     cv2.waitKey(1)


class Carla97_Env:

    def __init__(self):
        print("Connecting to server.")

        self.client = carla.Client("localhost", 2000)   # connect to server
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.debug = self.world.debug
        self._settings = self.world.get_settings()
        self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
            no_rendering_mode= False,
            synchronous_mode= True,
            fixed_delta_seconds= 1. / FPS))
        
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1) #build global planner
        self.planner = GlobalRoutePlanner(dao)
        self.planner.setup()

        self.ego_car = None
        self.walkers = None

    def reset(self, scene_id, pose_id):
        print("\nReset the env.")

        self.config = GetConfig(scene_id, pose_id)
        print(self.config)
        self.task_type = self.config['task_type']
        #self.lane_type = self.config['lane_type']
        self.center = self.config['center']
        self.world.set_weather(self.config['weather'][0]) # set weather
        # self.world.get_spectator().set_transform(self.config['spectator'])
        self.ego_car = EgoCar(self.world, self.client, self.config, self.planner)
        self.walkers = Walkers(self.world, self.client, self.config)
        self.world.tick()

        self.walkers.start()
        for t in range(self.config['wait_tick']):  # wait until the car reaches the ground
            self.world.tick()
        self.ego_car.set_sensors() # sensor: camera, Lidar
        self.ego_car.set_measures() # measures: collison, location, lane invasion
        self.ego_car.start()
        self.start_frame = self.world.tick()
        self.frame = self.start_frame
        return self.ego_car.get_sensors(self.start_frame), self.ego_car.get_measures()

    def step(self):

        self.ego_car.step()
        self.walkers.step()
        self.frame = self.world.tick()
        sensor_dict = self.ego_car.get_sensors(self.frame)
        measure_dict = self.ego_car.get_measures()

        return sensor_dict, measure_dict, self.get_done(measure_dict), self.frame-self.start_frame

    def get_done(self, measure_dict):
        if measure_dict['Collision'][0] > 0.0:
            print("Collision occurs!!!")
            return -2

        if not INSIDE(measure_dict['location'], self.config['AREA']):
            print("Out of range!!!")
            loc = measure_dict['location']
            area = self.config['AREA']
            if loc.x <= area['MINX']:
                return self.task_type[0]
            if loc.x >= area['MAXX']:
                return self.task_type[1]
            if loc.y <= area['MINY']:
                return self.task_type[2]
            if loc.y >= area['MAXY']:
                return self.task_type[3]

        return None

    def destroy(self):
        for x in [self.walkers, self.ego_car]:
            if x:
                x.destroy()
        
        self.ego_car = None
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
    p_dir = "./ExpertData"
    mkdir(p_dir)
    for scene in range(Scene_num):
        if scene != 0:
            continue
        file_name = os.path.join(p_dir, "scene_" + str(scene) + ".h5")

        for episode in range(1):
            group_name = "episode_" + str(episode)
            sensors_list, measures_list = [], []
            pose = episode%len(Experiments[scene])

            sensor_dict, measure_dict = env.reset(scene, pose)

            for i in range(2000):
                print('=========== Scene %d, Episode %d, Pose %d, Frame %d : =========' % (scene, episode, pose, i))

                sensor_dict_t1, measure_dict_t1, done, timestamp = env.step()
                sensor_dict['FrontRGB'] = pre_process(sensor_dict['FrontRGB'])

                if i%2 == 1:
                    print('## direction is ', measure_dict['direction'])

                    sensors_list.append(sensor_dict)
                    measures_list.append(measure_dict)
                    points, lidar_transform = sensor_dict['Lidar']
                    lidar = lidar_transform.location
                    #env.debug.draw_point(lidar, 0.1, carla.Color(0, 0, 255), 1, False)
                    cnt = 0
                    lidar_list = [1.0] * 360
                    #print("points: ", points)
                    for point in points:
                        point = np.array([point[0], point[1]])
                        rel_dis = get_distance(point)
                        rel_deg = get_angle(point)
                        if rel_dis<lidar_list[rel_deg]:
                            lidar_list[rel_deg] = rel_dis
                        point_loc = sensor_to_world(point, lidar_transform)
                        point_loc = carla.Location(x=point_loc[0], y=point_loc[1], z=1.5)
                        #env.debug.draw_point(point_loc, 0.1, carla.Color(255, 162, 0), 1, False)
                        #env.debug.draw_string(point_loc, str(cnt), False, carla.Color(255, 162, 0), 1, persistent_lines=False)
                        cnt += 1
                    #print("lidar_list: ", lidar_list)
                    sensor_dict['Lidar'] = np.array(lidar_list)
                    #print("after change: ", sensor_dict['Lidar'].shape)

                if done != None:
                    lent = len(sensors_list)
                    print("length: ", lent)
                    task_type_list = [[done]] * lent
                    print(task_type_list)
                    break
                
                sensor_dict, measure_dict = sensor_dict_t1, measure_dict_t1

            env.close()
            sensors_list, measures_list, task_type_list = np.array(sensors_list), np.array(measures_list), np.array(task_type_list)
            write_hdf5(file_name, group_name, sensors_list, measures_list, task_type_list, env.center)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')