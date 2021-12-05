#!/usr/bin/env python

# authors: Zhu Zeyu (zhuzeyu@pku.edu.cn)

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """


import carla
import sys

sys.path.append('..')
from agent.agent import Agent, AgentState
from planner.local_planner import LocalPlanner
from planner.global_route_planner import GlobalRoutePlanner
from planner.global_route_planner_dao import GlobalRoutePlannerDAO
from tools.misc import draw_waypoints

def draw_waypoint_union(debug, w0, w1, color=carla.Color(0, 0, 255), lt=60):
    debug.draw_line(
        w0.transform.location + carla.Location(z=0.25),
        w1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=True)
    debug.draw_point(w1.transform.location + carla.Location(z=0.25), 0.1, color, lt, True)

class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.02,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None

    def set_destination(self, destination):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router

        :param destination:   carla.Location object
        """

        print("vehicle start location in basic_agent.py :", self._vehicle.get_location())
        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(destination)

        route_trace = self._trace_route(start_waypoint, end_waypoint)
        assert route_trace

        self._local_planner.set_global_plan(route_trace)

        for i in range(len(route_trace) - 1):
            w0 = route_trace[i][0]
            w1 = route_trace[i + 1][0]
            debug = self._vehicle.get_world().debug
            assert debug
            draw_waypoint_union(debug,w0, w1)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def get_trace(self):
        return self._local_planner.get_waypoints_queue()

    def run_step(self, debug=True):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        pedestrians_list=actor_list.filter("*walker.pedestrian*")
        # check possible obstacles
        next_waypoint=self._local_planner.get_next_waypoint()

        #draw_waypoints(self._vehicle.get_world(), [next_waypoint], self._vehicle.get_location().z + 1.0)
        
        #speed_factor:a discount factor of throttle
        speed_factor=1.0
        #check vehicle,if any vehicle blocks the way,give a speed_factorv
        #print("Calling self._is_vehicle_hazard")
        vehicle_state, vehicle,speed_factorv = self._is_vehicle_hazard(vehicle_list,next_waypoint)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))
                print("blocked by", vehicle)
                vehicle_wp = self._map.get_waypoint(vehicle.get_location())
                print("on road {} and lane {}".format(vehicle_wp.road_id, vehicle_wp.lane_id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
        #check pedestrian,if any pedestrian blocks the way,give a speed_factorp

        #print("speed factor vehicle {}".format(speed_factorv))

        pedestrian_state, pedestrian,speed_factorp = self._is_pedestrian_hazard(pedestrians_list,next_waypoint)
        if pedestrian_state:
            if debug:
                print('!!! PEDESTRAIN BLOCKING AHEAD [{}])'.format(pedestrian.id))

            self._state = AgentState.BLOCKED_BY_PEDESTRIAN
        #speed_factor changes the speed gradually
        #print("speed factor pedestrian {}".format(speed_factorp))

        speed_factor=min(speed_factorp,speed_factorv)
        #print("speed_factor {}".format(speed_factor))
        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            print("EMERGENCY STOP BY LIGHT!!!!!")
            control = self.emergency_stop()
        else:
            # standard local planner behavior
            control = self._local_planner.run_step(speed_factor ,debug=debug)
            #control = self._local_planner.run_step(1.0,debug=debug)
            if speed_factor==0:
                print("EMERGENCY STOP BY SPEED FACTOR 0!!!!!")   
                control=self.emergency_stop()
            else:
                print("PID CONTROL!!!!")
                print("speed_factor=",speed_factor)
                #control = self._local_planner.run_step(speed_factor ,debug=debug)
                self._state = AgentState.NAVIGATING
            
        #print(control)
        return control

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        # TODO: note that local_planner 's done() is Not Implemented
        return self._local_planner.done()
