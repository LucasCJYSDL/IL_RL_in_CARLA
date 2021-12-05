#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)
#
# This script implements the low-level PID controller for ego car.
# Please note that this is based mainly on the 0.9.x version carla's controller,
# and this PID controller is a "pure" one, different from data-collector's controller,
# where rules are also considered.
# Note : this script may be modified in future to integrate rules.

""" This module contains PID controllers to perform lateral and longitudinal control """

from collections import deque

import sys
import math
import numpy as np

import carla

sys.path.append("..")

from tools.misc import get_speed

class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P = 1.0, K_D = 0.0, K_I = 0.0, dt = 0.03, buffersize = 30):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds        
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen = buffersize)
    
    def run_step(self, target_speed, debug = False,speed_factor=1.0):
        """
        Execute one step of longitudinal control to reach a given target speed.

        :param target_speed: target speed in Km/h
        :param debug: is debug mode enabled
        :return: throttle control in the range [0, 1]
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print("Current speed = {}".format(current_speed))

        return self._pid_control(target_speed, current_speed,speed_factor)

    def _pid_control(self, target_speed, current_speed,speed_factor=1.0):
        """
        Estimate the throttle of the vehicle based on the PID equations.

        :param target_speed: target speed in Km/h
        :param current_speed: current speed of the vehicle in Km/h
        :return: throttle control in the range [0, 1]
        """
        """
        origin version
        """
        # _error = target_speed - current_speed
        # self._e_buffer.append(_error)

        # if len(self._e_buffer) >= 2:
        #     # we can compute differential and integral
        #     _differential = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
        #     _integral = sum(self._e_buffer) * self._dt
        # else:
        #     _differential = 0.0
        #     _integral = 0.0

        # return np.clip( (self._K_P * _error) + (self._K_I * _integral) + (self._K_D * _differential), 0.0, 1.0)

        """ 
        0.8 version ，gives throttle and brake
        """
        _error =current_speed - target_speed*speed_factor 
        self._e_buffer.append(_error)

        if len(self._e_buffer) >= 2:
            # we can compute differential and integral
            _differential = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _integral = sum(self._e_buffer) * self._dt
        else:
            _differential = 0.0
            _integral = 0.0
        pid_gain=(self._K_P * _error) + (self._K_I * _integral) + (self._K_D * _differential)
        throttle = min(max(0- 1.3 * pid_gain, 0),1.0)
        brake=0.0
        '''
        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * 0.75, 1)
        else:
            brake = 0       
        '''
        return throttle,brake
        
        """
        easy version, if throttle is negative,brake=-throttle,throttle=0
        
        """
        # _error = target_speed*speed_factor - current_speed
        # self._e_buffer.append(_error)

        # if len(self._e_buffer) >= 2:
        #     # we can compute differential and integral
        #     _differential = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
        #     _integral = sum(self._e_buffer) * self._dt
        # else:
        #     _differential = 0.0
        #     _integral = 0.0
        # brake=0.0
        # throttle=(self._K_P * _error) + (self._K_I * _integral) + (self._K_D * _differential)
        # if throttle<0.0:
        #     brake=min(-throttle,1.0)
        #     throttle=0.0
        # else:
        #     brake=0.0
        #     throttle=min(throttle,1.0)

        # return throttle,brake


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """      

    def __init__(self, vehicle, K_P = 1.0, K_D = 0.0, K_I = 0.0, dt = 0.03, buffersize = 10):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds        
        """
        self._vehicle = vehicle
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=buffersize)
    
    def run_step(self, waypoint):
        """
        Execute one step of lateral control to steer the vehicle towards a ceratain waypoint.

        :param waypoint: target waypoint
        :return: steering control in the range [-1, 1] where:
            -1 represents maximum steering to left
            +1 represents maximum steering to right
        """
        return self._pid_control(waypoint, self._vehicle.get_transform())
    
    def _pid_control(self, waypoint, vehicle_transform):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_transform.location
        v_end = v_begin + carla.Location(x = math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                         y = math.sin(math.radians(vehicle_transform.rotation.yaw)))

        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x,
                          waypoint.transform.location.y - v_begin.y, 0.0])   
        _dot = math.acos(np.clip(np.dot(w_vec, v_vec) / 
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0
        
        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            # we can compute differential and integral
            _differential = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _integral = sum(self._e_buffer) * self._dt
        else:
            _differential = 0.0
            _integral = 0.0
        
        return np.clip( (self._K_P * _dot) + (self._K_D * _differential) + (self._K_I * _integral), -1.0, 1.0)

class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers (lateral and longitudinal) to perform the 
    low level control of a vehicle from client side.
    Actually, we can ignore this class and just use PIDLateralController and PIDLongitudinalController directly.
    """

    def __init__(self, vehicle, args_lateral = None, args_longitudinal = None):
        """
        :param vehicle: actor to apply to local planner logic onto
        :params args_lateral: dictionary of arguments to set the lateral PID controller using the following semantics:
                              K_P -- Proportional term
                              K_D -- Differential term
                              K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal PID controller using the following semantics:
                              K_P -- Proportional term
                              K_D -- Differential term
                              K_I -- Integral term
        
        """
        # TODO: the following default PID parameters need to be fine tuned
        if not args_lateral:
            args_lateral = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}
        if not args_longitudinal:
            args_longitudinal = {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.0}
        
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lat_controller = PIDLateralController(self._vehicle, **args_lateral)        

    def run_step(self, target_speed, waypoint,speed_factor=1.0):
        """
        Execute one step of control invoking both lateral and longitudinal PID controllers to reach a target waypoint
        at a given target speed.

        :param target_speed: desired vehicle speed
        :param waypoint: target location encoded as a waypoint
        :return: carla.VehicleControl object
        """
        # TODO: In future, rules may be considered to integrate into this method
        throttle,brake = self._lon_controller.run_step(target_speed,speed_factor)
        steering = self._lat_controller.run_step(waypoint)

        control  = carla.VehicleControl()
        control.steer = steering
        control.throttle = throttle
        control.brake = brake
        control.hand_brake = False
        control.manual_gear_shift = False

        return control
