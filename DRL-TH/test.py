#!/usr/bin/env python

# -*- coding: UTF-8 -*-



""" main script """

import random
import time
import rospy
import copy
import numpy as np
import cv2
import scipy
import sys
import math
import os
import logging
import socket
import matplotlib.pyplot as plt
import tf
import carla
import torch
import torch.nn as nn
from env2 import CarlaEnv1

from torch.optim import Adam

from collections import deque

from network.net import CNNPolicy

from network.ppo import ppo_update_stage1, generate_train_data

from network.ppo import generate_action, generate_action_no_sampling

from network.ppo import transform_buffer

from network.erfnet import erfnet

from network.vlnet import VLnet

from tensorboardX import SummaryWriter
writer = SummaryWriter('./log_test')

def setup_seed(seed):

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

setup_seed(20)

MAX_EPISODES = 10000

LASER_BEAM = 512

HORIZON = 256

GAMMA = 0.99

LAMDA = 0.95

BATCH_SIZE = 256

EPOCH = 4

COEFF_ENTROPY = 0.01

CLIP_VALUE = 0.1

OBS_SIZE = 512

ACT_SIZE = 2

LEARNING_RATE = 3e-4

NUM_ENV = 1



def run(env, lidar_policy ,visual_policy ,vl_policy , policy_path, action_bound, optimizer):

    global_update = 0
    global_step = 0
    lidar_policy.eval()
    visual_policy.eval()
    vl_policy.eval()

    for id in range(MAX_EPISODES):
        env.reset_pose(id)
        # env.gen_barrier()
        terminal = False
        ep_reward = 0
        step = 1
        # obs_lidar = env.get_laser_observation()
        obs_lidar = env.get_cad_bev()
        obs_seg = env.get_image_observation_rgb2seg(global_step)
        # obs_lidar_stack = deque([obs_lidar, obs_lidar, obs_lidar])
        # goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        goal = np.asarray(env.get_waypoint_polar())

        if speed[0] < 0:
            speed[0] = 0

        state = [obs_lidar, obs_seg , goal]
        while not terminal and not rospy.is_shutdown():
            # env.det_barrier()
            state_list = []
            state_list.append(state)
            #测试选的动作为网络输出的均值，而不是分布进行采样
            _,_,mean, scaled_action,scaled_action1 = generate_action_no_sampling(env=env, state_list=state_list, lidar_policy=lidar_policy,

                                                           visual_policy=visual_policy, vl_policy=vl_policy, action_bound=action_bound)
            # execute actions
            env.control_vel(scaled_action1[0])
            #print(scaled_action[0])
            #time.sleep(1)
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step, id)
            r= (r+3.55)/7.05
            ep_reward += r
            global_step += 1
            # get next state

            obs_lidar = env.get_cad_bev()
            # lidar_next = env.get_laser_observation()
            # left = obs_lidar_stack.popleft()
            # obs_lidar_stack.append(lidar_next)
            obs_seg = env.get_image_observation_rgb2seg(global_step)
            #stop = time.time()
            #print(stop - start)

            # goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            goal_next = np.asarray(env.get_waypoint_polar())

            if speed[0] < 0:
                speed[0] = 0
            state_next = [obs_lidar, obs_seg, goal_next]
            step += 1
            state = state_next

        # writer.add_scalar('episode reward', ep_reward,id)
        # env.dis_barrier(terminal)
        print('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, global_steps %d, Reward %-5.4f,  %s' % \

              (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, global_update, ep_reward,  result))

# if __name__ == '__main__':

#     reward = None
#     action_bound = [[0, -1], [1, 1]]

#     # lidar_policy_path = 'lidar_models'

#     # visual_policy_path = 'visual_models'

#     # vl_policy_path = '/home/xyd/project/1.RL/RP/test/hxq/21/m1_1117' # 'vl_models_no_car_and_person_best'-->1360       'vl_models'-->1200

#     # lidar_policy = CNNPolicy(action_space=2)

#     # lidar_policy.cuda()

#     # n_classes = 2

#     # visual_policy = erfnet(n_classes=n_classes)

#     # visual_policy.cuda()

#     # vl_policy = VLnet(input_channel=3, action_space=2)

#     # vl_policy.cuda()

#     # opt = Adam(vl_policy.parameters(), lr=LEARNING_RATE)



#     lidar_policy_path = 'lidar_models'
#     visual_policy_path = 'visual_models'
#     vl_policy_path = 'vl_models_complex1_test' 

#     policy_path = 'p1/m1/'
#     actor_policy_path = 'p1/m1/'


#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     lidar_policy = CNNPolicy(action_space=2).to(device)


#     n_classes = 2
#     visual_policy = erfnet(n_classes=n_classes)
#     visual_policy.cuda()
    

#     vl_policy = VLnet(input_channel=3, action_space=2)
#     vl_policy.cuda()

#     opt = Adam(vl_policy.parameters(), lr=LEARNING_RATE)

#     # lidar_file = lidar_policy_path + '/Stage1_6000'
#     visual_file = visual_policy_path + '/erfnet_CityScapes_class_2_496.pt'







#     # lidar_file = lidar_policy_path + '/Stage1_6000'

#     # visual_file = visual_policy_path + '/erfnet_CityScapes_class_2_496.pt'

#     vl_file = vl_policy_path + '/Stage2_7403 episode_5253reward_1012.4866'    #'/Stage1_7500'   #1460  1200 1040 660 620 600 560 500

#     if os.path.exists(lidar_file) and os.path.exists(visual_file) :

#         print('####################################')

#         print('############Loading_Model###########')

#         print('####################################')

#         state_dict = torch.load(lidar_file, map_location='cuda:0')

#         lidar_policy.load_state_dict(state_dict)

#         state_dict = torch.load(visual_file, map_location='cuda:0')

#         visual_policy.load_state_dict(state_dict)

#     else:
#         visual_policy =None

#         lidar_policy = None

#     if os.path.exists(vl_file):

#         print('####################################')

#         print('############Loading_VLModel###########')

#         print('####################################')

#         state_dict = torch.load(vl_file, map_location='cuda:0')

#         vl_policy.load_state_dict(state_dict)

#     else:
#         print('#####################################')
#         print('############Start Training###########')
#         print('#####################################')

#     env = CarlaEnv1(index = 0,visual_policy = visual_policy)
#     try:

#         run(env=env, lidar_policy=lidar_policy, visual_policy=visual_policy, vl_policy = vl_policy,
#                policy_path=vl_policy_path, action_bound=action_bound, optimizer=opt)

#     except KeyboardInterrupt:
#         pass


if __name__ == '__main__':
    reward = None
    action_bound = [[0, -1], [1, 1]]

    lidar_policy_path = 'lidar_models'
    visual_policy_path = 'visual_models'
    vl_policy_path = 'vl_models_complex1_test'

    policy_path = 'p1/m1/'
    actor_policy_path = 'p1/m1/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lidar_policy = CNNPolicy(action_space=2).to(device)

    n_classes = 2
    visual_policy = erfnet(n_classes=n_classes)
    visual_policy.cuda()

    vl_policy = VLnet(input_channel=3, action_space=2)
    vl_policy.cuda()

    opt = Adam(vl_policy.parameters(), lr=LEARNING_RATE)

    visual_file = visual_policy_path + '/erfnet_CityScapes_class_2_496.pt'
    vl_file = vl_policy_path + '/Stage2_4701 episode_4903reward_491.3908'

    if os.path.exists(visual_file):
        print('####################################')
        print('############Loading_Model###########')
        print('####################################')
        state_dict = torch.load(visual_file, map_location='cuda:0')
        visual_policy.load_state_dict(state_dict)
    else:
        visual_policy = None

    if os.path.exists(vl_file):
        print('####################################')
        print('############Loading_VLModel###########')
        print('####################################')
        state_dict = torch.load(vl_file, map_location='cuda:0')
        vl_policy.load_state_dict(state_dict)
    else:
        print('#####################################')
        print('############Start Testing###########')
        print('#####################################')

    env = CarlaEnv1(index=0, visual_policy=visual_policy)
    try:
        run(env=env, lidar_policy=lidar_policy, visual_policy=visual_policy, vl_policy=vl_policy,
            policy_path=vl_policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass