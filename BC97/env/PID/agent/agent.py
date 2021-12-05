#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

from enum import Enum

import carla
import sys
import math
sys.path.append('..')
from ..tools.misc import is_within_distance_ahead, compute_magnitude_angle
from .utils import get_vec_dist, get_angle

class AgentState(Enum):
    """
    AGENT_STATE represents the possible states of a roaming agent
    """
    NAVIGATING = 1
    BLOCKED_BY_VEHICLE = 2
    BLOCKED_RED_LIGHT = 3
    BLOCKED_BY_PEDESTRIAN = 4


class Agent(object):
    """
    Base class to define agents in CARLA
    """

    def __init__(self, vehicle):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        self._vehicle = vehicle
        self._proximity_threshold = 10.0  # meters
        self._local_planner = None
        self._world = self._vehicle.get_world()
        self._map = self._vehicle.get_world().get_map()
        self._last_traffic_light = None
        self.param = {
            'stop4TL': True,  # Stop for traffic lights
            'stop4P': True,  # Stop for pedestrians
            'stop4V': True,  # Stop for vehicles
            'coast_factor': 2,  # Factor to control coasting
            'tl_min_dist_thres': 6,  # Distance Threshold Traffic Light
            'tl_max_dist_thres': 20,  # Distance Threshold Traffic Light
            'tl_angle_thres': 0.5,  # Angle Threshold Traffic Light
            'p_dist_hit_thres': 35,  # Distance Threshold Pedestrian
            'p_angle_hit_thres': 0.15,  # Angle Threshold Pedestrian
            'p_dist_eme_thres': 8,  # Distance Threshold Pedestrian
            'p_angle_eme_thres': 0.5,  # Angle Threshold Pedestrian
            'v_dist_thres': 20,  # Distance Threshold Vehicle
            'v_angle_thres': 0.40  # Angle Threshold Vehicle

        }       
    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()

        if debug:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            control.hand_brake = False
            control.manual_gear_shift = False

        return control

    def _is_light_red(self, lights_list):
        """
        Method to check if there is a red light affecting us. This version of
        the method is compatible with both European and US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        if self._map.name == 'Town01' or self._map.name == 'Town02':
            return self._is_light_red_europe_style(lights_list)
        else:
            return self._is_light_red_us_style(lights_list)

    def _is_light_red_europe_style(self, lights_list):
        """
        This method is specialized to check European style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                  affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_waypoint = self._map.get_waypoint(traffic_light.get_location())
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(traffic_light.get_transform(),
                                        self._vehicle.get_transform(),
                                        self._proximity_threshold):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return (True, traffic_light)

        return (False, None)

    def _is_light_red_us_style(self, lights_list, debug=False):
        """
        This method is specialized to check US style traffic lights.

        :param lights_list: list containing TrafficLight objects
        :return: a tuple given by (bool_flag, traffic_light), where
                 - bool_flag is True if there is a traffic light in RED
                   affecting us and False otherwise
                 - traffic_light is the object itself or None if there is no
                   red traffic light affecting us
        """
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if ego_vehicle_waypoint.is_junction:
            # It is too late. Do not block the intersection! Keep going!
            return (False, None)

        if self._local_planner.target_waypoint is not None:
            if self._local_planner.target_waypoint.is_junction:
                min_angle = 180.0
                sel_magnitude = 0.0
                sel_traffic_light = None
                for traffic_light in lights_list:
                    loc = traffic_light.get_location()
                    magnitude, angle = compute_magnitude_angle(loc,
                                                               ego_vehicle_location,
                                                               self._vehicle.get_transform().rotation.yaw)
                    if magnitude < 60.0 and angle < min(25.0, min_angle):
                        sel_magnitude = magnitude
                        sel_traffic_light = traffic_light
                        min_angle = angle

                if sel_traffic_light is not None:
                    if debug:
                        print('=== Magnitude = {} | Angle = {} | ID = {}'.format(
                            sel_magnitude, min_angle, sel_traffic_light.id))

                    if self._last_traffic_light is None:
                        self._last_traffic_light = sel_traffic_light

                    if self._last_traffic_light.state == carla.TrafficLightState.Red:
                        return (True, self._last_traffic_light)
                else:
                    self._last_traffic_light = None

        return (False, None)

    def _is_vehicle_hazard(self, vehicle_list,next_waypoint):
        """
        Check if a given vehicle is an obstacle in our way. To this end we take
        into account the road and lane the target vehicle is on and run a
        geometry test to check if the target vehicle is under a certain distance
        in front of our ego vehicle.

        WARNING: This method is an approximation that could fail for very large
         vehicles, which center is actually on a different lane but their
         extension falls within the ego vehicle lane.

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
                 -speed factorv for control speed(if no obstacles,speed_factor=1)
        """
        
        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        speed_factor=1.0
        block_vehicle=None
        if next_waypoint is None:
            return (False, None,speed_factor)
        #wp_vector/wp_dis:vector/distance from current waypoint to next target waypoint
        #print("Now in _is_vehicle_hazard")
        wp_vector,wp_dis= get_vec_dist(next_waypoint.transform.location.x,next_waypoint.transform.location.y,ego_vehicle_location.x,ego_vehicle_location.y)
        #print("wp_vector, wp_dis:", wp_vector, wp_dis)
        #print("vehicle_list length:",len(vehicle_list))
        
        for target_vehicle in vehicle_list:
            # do not account for the ego vehicle
            speed_factor_tmp=1.0
            if target_vehicle.id == self._vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
            '''
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            '''
            #print('target road id, lane id:',target_vehicle_waypoint.road_id, target_vehicle_waypoint.lane_id)
            #print('ego road id, lane id:',ego_vehicle_waypoint.road_id,ego_vehicle_waypoint.lane_id)

            #check whether ego_vehicle needs to change speed_factor,if more than one vehicle blocks,choose the nearest
            speed_factor_tmp=self.stop_vehicle(ego_vehicle_location,target_vehicle.get_location(),wp_vector)
            #print("speed_factor_tmp: ", speed_factor_tmp)
            if speed_factor > speed_factor_tmp:
                speed_factor=speed_factor_tmp
                block_vehicle=target_vehicle
        #print("speed_factor: {}".format(speed_factor))
        if speed_factor<1.0:
            return (True,block_vehicle,speed_factor)
        else:
            return (False, None,speed_factor)

    def _is_pedestrian_hazard(self, pedestrians_list,next_waypoint):

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        speed_factor=1
        block_pedestrian=None
        if next_waypoint is None:
            return (False, None,speed_factor)
        #wp_vector/wp_dis:vector/distance from current waypoint to next target waypoint
        wp_vector,wp_dis= get_vec_dist(next_waypoint.transform.location.x,next_waypoint.transform.location.y,ego_vehicle_location.x,ego_vehicle_location.y)
        for pedestrian in pedestrians_list:
            speed_factor_tmp=1.0
            # if the object is not in our lane it's not an obstacle
            pedestrian_waypoint = self._map.get_waypoint(pedestrian.get_location())
            '''
            if pedestrian_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    pedestrian_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            '''
            #check whether ego_vehicle needs to change speed_factor,if more than one pedestrian blocks,choose the nearest
            speed_factor_tmp=self.stop_pedestrian(ego_vehicle_location,pedestrian.get_location(),wp_vector)
            if speed_factor>speed_factor_tmp:
                speed_factor=speed_factor_tmp
                block_pedestrian=pedestrian
        if speed_factor<1.0:
            return (True, block_pedestrian,speed_factor)
        else:
            return (False, None,speed_factor)

    def emergency_stop(self):
        """
        Send an emergency stop command to the vehicle
        :return:
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control

    def is_pedestrian_hitable(self, pedestrian):

        """
        Determine if a certain pedestrian is in a hitable zone
        Check if pedestrians are on the road (Out of the sidewalk)
        :return:
        """

        x_agent = pedestrian.transform.location.x
        y_agent = pedestrian.transform.location.y

        return self._map.is_point_on_lane([x_agent, y_agent, 38])

    def is_pedestrian_on_hit_zone(self, p_dist, p_angle):
        """
        Draw a semi circle with a big radius but small period from the circunference.
        Pedestrians on this zone will cause the agent to reduce the speed

        """
        return math.fabs(p_angle) < self.param['p_angle_hit_thres'] and \
                    p_dist < self.param['p_dist_hit_thres']

    def is_pedestrian_on_near_hit_zone(self, p_dist, p_angle):

        return math.fabs(p_angle) < self.param['p_angle_eme_thres'] and \
                    p_dist < self.param['p_dist_eme_thres']

    # Main function for stopping for pedestrians
    def stop_pedestrian(self, location, pedestrian_location, wp_vector):

        speed_factor_p_temp = 1.0
        x_agent = pedestrian_location.x
        y_agent = pedestrian_location.y
        p_vector, p_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
        p_angle = get_angle(p_vector, wp_vector)
        # CASE 1: Pedestrian is close enough, slow down
        if self.is_pedestrian_on_hit_zone(p_dist, p_angle):
            speed_factor_p_temp = p_dist / (
                    self.param['coast_factor'] * self.param['p_dist_hit_thres'])
        #print("In stop pedestrian CASE 1")
        # CASE 2: Pedestrian is very close to the ego-agent
        if self.is_pedestrian_on_near_hit_zone(p_dist, p_angle):
            speed_factor_p_temp = 0
        #print("In stop pedestrian CASE 2")
        return speed_factor_p_temp

    """ **********************
        VEHICLE FUNCTIONS
        **********************
    """

    def stop_vehicle(self, location, vehicle_location, wp_vector, debug=False):

        speed_factor_v_temp = 1.0
        x_agent = vehicle_location.x
        y_agent = vehicle_location.y
        v_vector, v_dist = get_vec_dist(x_agent, y_agent, location.x, location.y)
        v_angle = get_angle(v_vector, wp_vector)

        #print(v_vector, v_angle)

        # CASE 1: Slowing down for a vehicle (Vehicle Following).
        if (-0.5 * self.param['v_angle_thres'] / self.param['coast_factor'] < v_angle <
                self.param['v_angle_thres'] / self.param['coast_factor'] and v_dist < self.param[
                    'v_dist_thres'] * self.param['coast_factor']) or (
                -0.5 * self.param['v_angle_thres'] / self.param['coast_factor'] < v_angle <
                self.param['v_angle_thres'] and v_dist < self.param['v_dist_thres']):
            if debug:
                print("In stop vehicle CASE 1")
            speed_factor_v_temp = v_dist / (self.param['coast_factor'] * self.param['v_dist_thres'])
            if debug:
                print("Speed factor:",speed_factor_v_temp)
        # CASE 2: Stopping completely for the lead vehicle.
        if (-0.5 * self.param['v_angle_thres'] * self.param['coast_factor'] < v_angle <
                self.param['v_angle_thres'] * self.param['coast_factor'] and v_dist < self.param[
                                                    'v_dist_thres'] / self.param['coast_factor']):
            if debug:
                print("In stop vehicle CASE 2")
            speed_factor_v_temp = 0
            if debug:
                print("Speed factor:",speed_factor_v_temp)

        #print("In stop_vehicle, computed speed factor is :", speed_factor_v_temp)
        return speed_factor_v_temp
