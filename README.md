# Pedestrian Interaction Learning for Autonomous Vehicles
Codebase for my Bachelor thesis: Autonomous Driving Control for Interaction with Pedestrains based on Imitation Learing

Language: Python

The following components are included:
- A simulator for the interactions between the autonomous vehicle and pedestrians at the traffic lights, which is built on CARLA.
- A data collector for generating expert driving data based on PID control.
- Implementations of SOTA imitation learning algorithms: Behavioral Cloning, GAIL, and AIRL, for learning behaviors based on the collected expert driving data.
- Implementation of SOTA reinforcement learning algorithms: TRPO and PPO, for learning driving behaviors, which are used as baselines.
- A deep multi-head CNN and corresponding pre-trained checkpoint for vision/image-based self-driving.
