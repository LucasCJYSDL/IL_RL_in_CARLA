from carla.client import CarlaClient,make_carla_client
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla.carla_server_pb2 import Control
from carla.planner.planner import  Planner
from carla.tcp import TCPConnectionError
from carla.client import VehicleControl
import  Env.carla_config  as carla_config

#from environment.plot_position import plot_position
import random
import logging
import time
import numpy as np
import math
from Utils.recording import Recording


def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

class Env(object):

	def __init__(self, log_dir,image_agent,city="/Game/Maps/Town01", continue_experiment=False, save_images=False):
		self.log_dir = log_dir
		self.recording=Recording(name_to_save=self.log_dir, continue_experiment=continue_experiment, save_images=save_images)
		self.game = None  #carla client 
		self.map = city
		self.host = 'localhost'
		self.port = 2000
		self.Image_agent = image_agent
		self.speed_up_steps = 20
		self.weather = -1
		self.current_position = None
		self.total_reward = 0
		self.planner = None
		self.carla_setting = None
		self.number_of_vehicles = None
		self.control = None
		self.nospeed_times = 0
		self.reward = 0
		self._distance_for_success = 2.0

		self.load_config()
		self.setup_client_and_server()

	def load_config(self):
		self.vehicle_pair = carla_config.NumberOfVehicles
		self.pedestrian_pair = carla_config.NumberOfPedestrians
		self.weather_set = carla_config.set_of_weathers
		#[straight,one_curve,navigation,navigation]
		if self.map=="/Game/Maps/Town01":
			self.poses = carla_config.poses_town01()
		elif self.map=="/Game/Maps/Town02":
			self.poses = carla_config.poses_town02()
		else:
			print("Unsupported Map Name")

	def setup_client_and_server(self):
		self.game = CarlaClient(self.host, self.port, timeout=99999999) #carla  client
		self.game.connect(connection_attempts=100)
		#if failed, it will raise exceptions
	
	def reset(self):
		self.nospeed_times =0
		self.id_experiment = random.randint(0, len(self.poses)-1)
		pose_type = self.poses[self.id_experiment]
		self.current_position = self.start_point = random.choice(pose_type)  #start and  end  index
		self.number_of_vehicles = random.randint( self.vehicle_pair[0],self.vehicle_pair[1])
		self.number_of_pedestrians = random.randint( self.pedestrian_pair[0],self.pedestrian_pair[1])
		# self.number_of_vehicles = 0
		# self.number_of_pedestrians = 0
		self.weather = random.choice(self.weather_set)

		logging.info('======== !!!! ==========')
		logging.info(' Start Position %d End Position %d ', self.current_position[0], self.current_position[1])

		self.recording.log_poses(self.current_position[0], self.current_position[1], self.weather)
		
		settings = carla_config.make_carla_settings()
		settings.set(
			NumberOfVehicles=self.number_of_vehicles,
			NumberOfPedestrians=self.number_of_pedestrians,
			WeatherId= self.weather
		)

		self.carla_setting = settings
		self.scene = self.game.load_settings(settings)
		self.game.start_episode(self.current_position[0]) #set the start position
		#print(self.current_position)
		self.target_transform = self.scene.player_start_spots[self.current_position[1]]
		self.planner = Planner(self.scene.map_name)

		self.success = False
		#skip the  car fall to sence frame
		for i in range(self.speed_up_steps): 
			self.control = VehicleControl()
			self.control.steer = 0
			self.control.throttle = 0.025*i
			self.control.brake = 0
			self.control.hand_brake = False
			self.control.reverse = False
			time.sleep(0.05)
			send_success = self.send_control(self.control)
			if not send_success:
				return None
			self.game.send_control(self.control)

		#measurements, sensor_data = self.game.read_data() #measurements,sensor
		measurements, sensor_data = self.get_data()
		self.initial_timestamp = measurements.game_timestamp
		self.initial_distance = self.distance = self.get_distance(measurements, self.target_transform)
		self.control_ls, self.measure_ls, self.reward_ls = [], [], []
		self.current_step = 0

		directions =self.get_directions(measurements,self.target_transform,self.planner)
		if directions is None or measurements is None:
			return None
		# self.measure_ls.append(measurements)
		state,_,_=self.get_experience(measurements,sensor_data,directions)
		return state

	def step(self,action):
		#take action ,update state 
		#return: observation, reward,done
		self.control = VehicleControl()
		self.control.steer = np.clip(action[0], -1.0, 1.0)
		self.control.throttle = np.clip(action[1], 0.0, 1.0)
		self.control.brake = np.abs(np.clip(action[2], 0.0, 1.0))
		self.control.hand_brake = False
		self.control.reverse = False
		send_success = self.send_control(self.control)
		if not send_success:
				return None,None,None, None
		self.control_ls.append(self.control)
		#recive  new data 
		#measurements, sensor_data = self.game.read_data() #measurements,sensor
		measurements, sensor_data = self.get_data()
		directions =self.get_directions(measurements,self.target_transform,self.planner)
		if measurements is  None or directions is None:
			return None,None,None, None
		self.measure_ls.append(measurements)
		state,reward,done=self.get_experience(measurements,sensor_data,directions)
		self.current_step+=1
		self.reward_ls.append(reward)
		return state,reward,done, measurements.game_timestamp

	def send_control(self, control):
		send_success = False
		try:
			self.game.send_control(control)
			send_success = True
		except Exception:
			print("Send Control error")
		return send_success

	def get_data(self):
		measurements=None
		sensor_data=None
		try:
			measurements, sensor_data = self.game.read_data()
		except Exception:
			return None,None
		return measurements,sensor_data

	def get_directions(self,measurements, target_transform, planner):
		""" Function to get the high level commands and the waypoints.
			The waypoints correspond to the local planning, the near path the car has to follow.
		"""
		# Get the current position from the measurements
		current_point = measurements.player_measurements.transform
		try:
			directions = planner.get_next_command(
				(current_point.location.x,
					current_point.location.y, 0.22),
				(current_point.orientation.x,
					current_point.orientation.y,
					current_point.orientation.z),
				(target_transform.location.x, target_transform.location.y, 0.22),
				(target_transform.orientation.x, target_transform.orientation.y,
					target_transform.orientation.z)
			)
		except Exception:
			print("Route plan error ")
			directions = None
		return directions

	def get_distance(self, measurements, target_transform):
		current_point = measurements.player_measurements.transform
		dist =  sldist([current_point.location.x, current_point.location.y], [target_transform.location.x, target_transform.location.y])
		return dist
		
	#comute new state,reward,and is done
	def get_experience(self,measurements,sensor_data,directions):
		self.reward = 0
		done = False 
		img_feature = self.Image_agent.compute_feature(sensor_data)  #shape = (512,)
		speed = measurements.player_measurements.forward_speed # m/s
		intersection_offroad = measurements.player_measurements.intersection_offroad
		intersection_otherlane = measurements.player_measurements.intersection_otherlane
		collision_vehicles = measurements.player_measurements.collision_vehicles
		collision_pedestrians = measurements.player_measurements.collision_pedestrians
		collision_other = measurements.player_measurements.collision_other

		self.reward = self.get_reward(directions, speed, intersection_offroad, intersection_otherlane,
									  collision_vehicles, collision_pedestrians, collision_other)

		done = self.get_done(intersection_offroad, intersection_otherlane, collision_vehicles,
							 collision_pedestrians, collision_other)

		if speed*3.6<=1.0:
			self.nospeed_times+=1
			if self.nospeed_times>100:
				done=True
			self.reward-=1
		else:
			self.nospeed_times=0
		# compute  state  512+2
		speed = min(1,speed/10.0)#max_speed=36km/h

		
		return  np.concatenate((img_feature, (speed,directions))),self.reward,done 

	def get_reward(self, directions, speed, intersection_offroad, intersection_otherlane, collision_vehicles,
				   collision_pedestrians, collision_other):
		reward = 0
		# reward for steer
		if directions == 5:  # go  straight
			if abs(self.control.steer) > 0.2:
				reward -= 20
			reward += min(35, speed * 3.6)
		elif directions == 2:  # follow  lane
			reward += min(25, speed * 3.6)
		elif directions == 3:  # turn  left ,steer should be negtive
			if self.control.steer > 0:
				reward -= 15
			if speed * 3.6 <= 20:
				reward += speed * 3.6
			else:
				reward += 40 - speed * 3.6
		elif directions == 4:  # turn  right
			if self.control.steer < 0:
				reward -= 15
			if speed * 3.6 <= 20:
				reward += speed * 3.6
			else:
				reward += 40 - speed * 3.6

		# reward  for  offroad  and  collision
		if intersection_offroad > 0:
			reward -= 100
		if intersection_otherlane > 0:
			reward -= 100
		elif collision_vehicles > 0:
			reward -= 100
		elif collision_pedestrians > 0:
			reward -= 100
		elif collision_other > 0:
			reward -= 50

		return reward

	def get_done(self, measurements, directions, intersection_offroad, intersection_otherlane, collision_vehicles, collision_pedestrians,
				 collision_other):
		# teminal  state
		done = False
		if collision_pedestrians > 0 or collision_vehicles > 0 or collision_other > 0:
			done = True
			print("Collision~~~~~")
		if intersection_offroad > 0.2 or intersection_otherlane > 0.2:
			done = True
			print("Offroad~~~~~")

		self.distance = self.get_distance(measurements, self.target_transform)
		if self.distance < self._distance_for_success or directions == 0:
			done = True
			self.success = True

		return done

	def random_sample(self):

		steer = random.uniform(-1.0, 1, 0)
		throttle = random.uniform(0.0, 1.0)
		brake = random.uniform(0.0, 1.0)

		return [np.array([steer, throttle, brake])]


		# if __name__=="__main__":
#     with tf.Session() as  sess:
#         env = Env("./log","./data",sess)
#         #env.setup_client_and_server()
#         env.reset()
	
   
