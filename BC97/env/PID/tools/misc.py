#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)
#
# This script implements some auxiliary functions. Mainly copied from original misc.py
""" Module with auxiliary functions. """

import math

import numpy as np

import carla

def draw_waypoints(world, waypoints, z = 0.5, arrow_size = 0.3, life_time = 1.0):
    """
    Draw a list of waypoints at a certain height given in  z.

    :param world: carla.world object
    :param waypoints: list or iterable container with the waypoints to draw
    :param z: height in meters
    :return:
    """
    for wp in waypoints:
        t = wp.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x = math.cos(angle), y = math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size = arrow_size, life_time = life_time, color = carla.Color(255,0,0))

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h
    :param vehicle: the vehicle for which speed is calculated
    :return: speed as a float in Km/h
    """ 
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True
    
    if norm_target > max_distance:
        return False
    
    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0

def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)

def distance_vehicle(waypoint, vehicle_transform):
    """
    Compute the distance between a waypoint and a vehicle

    :param waypoint: carla.Waypoint objects
    :param vehicle_transform: location of the vehicle
    :return: distance between the waypoint and the vehicle
    """
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx ** 2 + dy ** 2)

def vector(location_a, location_b):
    """
    Compute the unit vector from location_a to location_b
    :param location_a: carla.Location objects
    :param location_b: carla.Location objects
    :return: the unit vector from location_a to location_b
    """
    x = location_b.x - location_a.x
    y = location_b.y - location_a.y
    z = location_b.z - location_a.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]