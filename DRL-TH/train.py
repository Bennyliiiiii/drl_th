

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
# import subprocess

import torch
import torch.nn as nn

import os
import signal


def setup_seed(seed):

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    #torch.backends.cudnn.deterministic = True

setup_seed(20)


from env2 import CarlaEnv1
from torch.optim import Adam
from collections import deque
from network.net import CNNPolicy
from network.ppo import ppo_update_stage1, generate_train_data
from network.ppo import generate_action
from network.ppo import transform_buffer
from network.erfnet import erfnet
from network.vlnet import VLnet
from tensorboardX import SummaryWriter
writer = SummaryWriter('./p1/log')


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
LEARNING_RATE = 1e-5  
NUM_ENV = 1         



def run(env, lidar_policy ,visual_policy ,vl_policy , policy_path, action_bound, optimizer, start_episode = 0):
    # rate = rospy.Rate(5)


    buff = []  
    global_update = 0 
    every_update_average_reward = 0  
    global_step = 0  
    best = 0  
    lidar_policy.eval()  
    visual_policy.eval() 
    vl_policy.train()    
    '''
    if env.index == 0:
        env.reset_world()
    '''
    for id in range(start_episode, MAX_EPISODES):
        env.reset_pose(id) 
        # if id % 10 !=0:
        #     os.kill(573058,signal.SIGUSR1)
        env.gen_barrier()
        # ch_process = subprocess.Popen(["python3","./spawn_barrier/spawn_scenes_town02_3.py"])
        terminal = False    
        ep_reward = 0  
        step = 1  


     
        obs_lidar = env.get_cad_bev()
        obs_seg = env.get_image_observation_rgb2seg(global_step)        
        
        # obs_lidar_stack = deque([obs_lidar, obs_lidar, obs_lidar])   
        # goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        goal = np.asarray(env.get_waypoint_polar())
        if speed[0] < 0:
            speed[0] = 0                       
        state = [obs_lidar, obs_seg, goal]

        # ch_process = subprocess.Popen(["python3","./spawn_barrier/spawn_scenes_town02_3.py"])

        while not terminal and not rospy.is_shutdown():
            # cad_data = env.get_cad_bev()
            state_list = []
            state_list.append(state)

            # print(state_list)

            # generate actions at rank==0
            v, a, logprob, scaled_action = generate_action(env=env, state_list=state_list, lidar_policy=lidar_policy,
                                                           visual_policy=visual_policy, vl_policy=vl_policy, action_bound=action_bound)
            # execute actions
            env.control_vel(scaled_action[0])
            #print(scaled_action[0])
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step, id)
            # r= (r+4.05)/7.55                    
            ep_reward += r
            global_step += 1

            # get next state
            # lidar_next = env.get_laser_observation()
            # # left = obs_lidar.popleft()
            # obs_lidar.append(lidar_next)
            obs_lidar = env.get_cad_bev()
            obs_seg = env.get_image_observation_rgb2seg(global_step)
            #stop = time.time()
            #print(stop - start)

            # goal_next = np.asarray(env.get_local_goal())
            # speed_next = np.asarray(env.get_self_speed())
            goal_next = np.asarray(env.get_waypoint_polar())
            if speed[0] < 0:
                speed[0] = 0
            state_next = [obs_lidar, obs_seg, goal_next]
           
            if global_step % HORIZON == 0:
                state_next_list = []
                state_next_list.append(state_next)
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list,lidar_policy=lidar_policy,
                                                visual_policy=visual_policy, vl_policy=vl_policy,action_bound=action_bound)
            # add transitons in buff and update policy
            r_list = []
            r_list.append(r)
            terminal_list = []
            terminal_list.append(terminal)
            
            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                every_update_average_reward += np.mean(r_list)
                if len(buff) > HORIZON - 1:
                    lidar_batch, rgb_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (lidar_batch, rgb_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    
                    ppo_update_stage1(lidar_policy=lidar_policy, visual_policy=visual_policy, vl_policy=vl_policy,
                                      optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                      epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                      num_env=NUM_ENV, frames=3 ,obs_size=OBS_SIZE,rgb_size = (84,84) ,act_size=ACT_SIZE)

                    buff = []   
                    global_update += 1   
                    writer.add_scalar('global_update_reward', every_update_average_reward, global_update)

                    every_update_average_reward = 0
                    

            step += 1
            state = state_next   
        env.dis_barrier()
        # if id % 10 !=0:
        #     os.kill(573058,signal.SIGUSR2)
        # breakpoint()
        # ch_process.terminate()
        
        if env.check_timeout():
            if global_update != 0 and global_update < 500 and step >700:
                if ep_reward > best:
                    best = ep_reward        
                    torch.save(vl_policy.state_dict(), policy_path + '/Stage2_{} '.format(global_update)+'episode_{}'.format(id + 1)+'reward_{:.4f}'.format(ep_reward)) 
            elif global_update != 0 and  global_update >= 500 and step >700:
                torch.save(vl_policy.state_dict(), policy_path + '/Stage2_{} '.format(global_update)+'episode_{}'.format(id + 1)+'reward_{:.4f}'.format(ep_reward))
            torch.save({
                "episode_id": id,
                "episode_reward": ep_reward,
                "actor_optimizer": optimizer.state_dict(),
                "actor_state_dict": vl_policy.state_dict()
            }, policy_path + 'last.pth')

            writer.add_scalar('episode reward', ep_reward,id)
        else:
            print("The odometer data is lose")
        print('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, global_steps %d, Reward %-5.4f,  %s' % \
              (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, global_update, ep_reward,  result))
        


if __name__ == '__main__':

    reward = None
    action_bound = [[0, -1], [1, 1]]
    # g = [100, 100]
    # env.reset()

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

    # lidar_file = lidar_policy_path + '/Stage1_6000'
    visual_file = visual_policy_path + '/erfnet_CityScapes_class_2_496.pt'
    # vl_file = vl_policy_path + '/Stage2_2802 episode_4324reward'    

    if os.path.exists(visual_file) :
        print('####################################')
        print('############Loading_Model###########')
        print('####################################')
        # state_dict = torch.load(lidar_file, map_location='cuda:0')
        # lidar_policy.load_state_dict(state_dict)
        # lidar_policy = CNNPolicy(2)
        state_dict = torch.load(visual_file, map_location='cuda:0')
        visual_policy.load_state_dict(state_dict)
    else:
        visual_policy =None
        # lidar_policy = None

    last_file = actor_policy_path + 'last.pth'
    if os.path.exists(last_file):
        print('####################################')
        print('############recovery_from_last###########')
        print('####################################')
        resume_checkpoint = torch.load(last_file)
        start_episode = resume_checkpoint["episode_id"]
        actor_state_dict = resume_checkpoint["actor_state_dict"]
        vl_policy.load_state_dict(actor_state_dict)
        vl_policy.cuda()
        opt.load_state_dict(resume_checkpoint["actor_optimizer"])
    else:
        print('################x-1011-2cnn#####################')
        print('############learning rate 1e-4###########')
        print('#####################################')
        start_episode = 0

    env = CarlaEnv1(index = 0,visual_policy = visual_policy)


    try:
        run(env=env, lidar_policy=lidar_policy, visual_policy=visual_policy, vl_policy = vl_policy,
               policy_path=policy_path, action_bound=action_bound, optimizer=opt, start_episode=start_episode)
    except KeyboardInterrupt:
        pass


