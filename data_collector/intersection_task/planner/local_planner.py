#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)
#
# This script is mainly based on 0.9.x carla's local_planner. 
# Note that this may be modified in future to integrate some rules.  :)

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from enum import Enum
from collections import deque
import sys
import random

import carla

sys.path.append("..")
from controller.controller import VehiclePIDController, PIDLateralController, PIDLongitudinalController
from tools.misc import distance_vehicle, draw_waypoints

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    (similar to the high-level commands in 0.8.4 carla)
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a trajectory of waypoints that is generated on-the-fly.
    The low-level motion of the vehicle is computed by using two PID controllers, one is used for the lateral control 
    and the other for the longitudinal control (cruise speed).
    """    
    
    # minimum distance to target waypoint as a percentage (e.g. within 90% of total distance)
    # This constant may be modified or even removed in future. 
    MIN_DISTANCE_PERCENTAGE = 0.9

    def __init__(self, vehicle, opt_dict = None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param opt_dict: dictionary of arguments with the following semantics:
                        dt -- time difference between physics control in seconds. This is typically fixed from server side
                              using the arguments -benchmark -fps = F . In this case dt = 1/F
                        
                        target_speed -- desired cruise speed in Km/h

                        sampling_radius -- search radius for next waypoints in seconds: e.g. 0.5 seconds ahead

                        lateral_control_dict -- dictionary of arguments to setup the lateral PID controller 
                                                {'K_P':, 'K_D':, 'K_I':, 'dt', 'buffersize'}
                        
                        longitudinal_control_dict -- dictionary of arguments to setup the longitudinal PID controller
                                                     {'K_P':, 'K_D':, 'K_I':, 'dt', 'buffersize}
        """
        self._vehicle = vehicle
        self._map = self._vehicle.get_world().get_map()
        self.debug = self._vehicle.get_world().debug

        self._dt = None
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        self._current_waypoint = None
        self._target_road_option = None
        self._next_waypoints = None
        self.target_waypoint = None
        self._vehicle_controller = None
        self._global_plan = None

        # queue with tuples of (waypoint, RoadOption)
        self._waypoints_queue = deque(maxlen = 20000)
        self._buffer_size = 5
        self._waypoints_buffer = deque(maxlen = self._buffer_size)

        # initialize controller
        self._init_controller(opt_dict)

    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
        print("Destroying ego-vehicle!")

    def get_waypoints_queue(self):
        return self._waypoints_queue

    def reset_vehicle(self):
        self._vehicle = None
        print("Resetting ego-vehicle!")

    def _init_controller(self, opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._dt = 1.0 / 20.0
        self._target_speed = 20.0 # Km/h
        self._sampling_radius = self._target_speed * 1.0 / 3.6 # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        args_lateral_dict = {
            'K_P' : 1.95,
            'K_D' : 0.01,
            'K_I' : 1.4,
            'dt'  : self._dt,
            'buffersize' : 30
        }

        args_longitudinal_dict = {
            'K_P' : 1.0, 
            'K_D' : 0,
            'K_I' : 1,
            'dt'  : self._dt,
            'buffersize' : 10
        }

        # parameters overload
        if opt_dict:
            if 'dt' in opt_dict:
                self._dt = opt_dict['dt']
            if 'target_speed' in opt_dict:
                self._target_speed = opt_dict['target_speed']
            if 'sampling_radius' in opt_dict:
                self._sampling_radius =self._target_speed * opt_dict['sampling_radius'] / 3.6
            if 'lateral_control_dict' in opt_dict:
                args_lateral_dict = opt_dict['lateral_control_dict']
            if 'longitudinal_control_dict' in opt_dict:
                args_longitudinal_dict = opt_dict['longitudinal_control_dict']

        # get current waypoint according to vehicle's current location
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())    
        # initialize vehicle PID controller
        self._vehicle_controller = VehiclePIDController(self._vehicle, 
                                                        args_lateral = args_lateral_dict,
                                                        args_longitudinal = args_longitudinal_dict)           
        # flag whether using global plan or not
        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0], RoadOption.LANEFOLLOW))

        self._target_road_option = RoadOption.LANEFOLLOW
        # fill waypoint trajectory queue
        self._compute_next_waypoints(k = 200)

    def set_speed(self, speed):
        """
        Request new target speed

        :param speed: new target speed in Km/h
        :return:
        """
        self._target_speed = speed

    def _compute_next_waypoints(self, k = 1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
        k = min(available_entries, k)

        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            # sample next waypoints of last waypoint
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                # get the road option list with respect to the next_waypoints list
                road_options_list = _retrieve_options(next_waypoints, last_waypoint)
                
                # TODO: here should be checked whether random choice is appropriate or not
                choice = random.randint(0, len(road_options_list) - 1)
                road_option = road_options_list[choice]
                next_waypoint = next_waypoints[choice]
            
            self._waypoints_queue.append((next_waypoint, road_option))    
    
    def set_global_plan(self, current_plan):
        """
        Assign the global planned trajectory into self._waypoints_queue
        
        :param current_plan: the current global planned waypoints list
        :return:
        """
        self._waypoints_queue.clear()
        for elem in current_plan:
            self._waypoints_queue.append(elem)
        self._target_road_option = RoadOption.LANEFOLLOW
        self._global_plan = True

        '''
        print("global plan set up. Visualizing waypoints list in green.")
        print(self._waypoints_queue)
        for i in range(len(self._waypoints_queue) -1):
            w0 = self._waypoints_queue[i][0]
            w1 = self._waypoints_queue[i + 1][0]
            self.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(0,255,0), life_time=5, persistent_lines=True)
            self.debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, carla.Color(0,255,0), 5, True)
        '''


    def run_step(self,speed_factor=1.0,debug = True):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return:
        """

        # not enough waypoints in the horizon ? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k = 100)

        # special case
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            return control

        # Buffering the waypoints
        if not self._waypoints_buffer:
            # empty buffer
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                else:
                    break
        
        '''
        for i in range(len(self._waypoints_buffer) -1):
            w0 = self._waypoints_buffer[i][0]
            w1 = self._waypoints_buffer[i + 1][0]
            self.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(0,255,0), life_time=5, persistent_lines=False)
            self.debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, carla.Color(0,255,0), 5, False)
        '''
        
        # current vehicle waypoint
        vehicle_transform = self._vehicle.get_transform()
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())    
        # target waypoint
        self.target_waypoint, self._target_road_option = self._waypoints_buffer[0]
        # moving using PID controllers
        control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint,speed_factor)

        # purge the queue of obsolete waypoints
        max_index = -1
        for i, (waypoint, _) in enumerate(self._waypoints_buffer):
            if waypoint.transform.location.distance(vehicle_transform.location) < self._min_distance:
                max_index = i
        
        if max_index >= 0:
            for i in range(max_index + 1):
                self._waypoints_buffer.popleft()
        
        #print("local target_waypoint : ", self.target_waypoint)
        if debug:
            #draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], self._vehicle.get_location().z + 1.0)
            print("Local Planner Control output: throttle %f steer %f brake %f" % (control.throttle, control.steer, control.brake))
        return control

    def done(self):
        vehicle_transform = self._vehicle.get_transform()
        # TODO: check whether the conditions below is right
        # return len(self._waypoints_queue) == 0 and all([distance_vehicle(wp, vehicle_transform) < self._min_distance for wp in self._waypoints_queue])
        return NotImplementedError
    def get_next_waypoint(self):
        # not enough waypoints in the horizon ? => add more!
        if not self._global_plan and len(self._waypoints_queue) < int(self._waypoints_queue.maxlen * 0.5):
            self._compute_next_waypoints(k = 100)

        # special case
        if len(self._waypoints_queue) == 0:
            return None

        # Buffering the waypoints
        if not self._waypoints_buffer:
            # empty buffer
            for i in range(self._buffer_size):
                if self._waypoints_queue:
                    self._waypoints_buffer.append(self._waypoints_queue.popleft())
                else:
                    break
        target_waypoint, target_road_option = self._waypoints_buffer[0]
        return target_waypoint


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in 
    list_waypoints. The resule is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
                candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to 
        # the beggining of an intersection, therefore the 
        # variation in angle is small
        # TODO: figure out why
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)
    
    return options

def _compute_connection(current_waypoint, next_waypoint):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
                RoadOption.STARIGHT
                RoadOption.LEFT
                RoadOption.RIGHT
    TODO: why no lane change options here?
    """    
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < 1.0:
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT