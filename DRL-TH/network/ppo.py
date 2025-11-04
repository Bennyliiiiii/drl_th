import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

# 将buff拆开，分成各个batch
def transform_buffer(buff):
    lidar_batch, rgb_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], []
    lidar_temp, rgb_temp, speed_temp = [], [], []

    for e in buff:
        for state in e[0]:
            lidar_temp.append(state[0])
            rgb_temp.append(state[1])
            speed_temp.append(state[2])
        lidar_batch.append(lidar_temp)
        rgb_batch.append(rgb_temp)
        speed_batch.append(speed_temp)
        lidar_temp = []
        rgb_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    lidar_batch = np.asarray(lidar_batch)
    rgb_batch = np.asarray(rgb_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return lidar_batch, rgb_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch


def generate_action(env, state_list, lidar_policy, visual_policy, vl_policy, action_bound):
    if env.index == 0:
        lidar_list, rgb_list, goal_list = [], [], []
        for i in state_list:
            lidar_list.append(i[0])
            rgb_list.append(i[1])
            goal_list.append(i[2])

        lidar_list = np.asarray(lidar_list)
        rgb_list = np.asarray(rgb_list)
        goal_list = np.asarray(goal_list)

        lidar_list = Variable(torch.from_numpy(lidar_list)).float().cuda()
        rgb_list = Variable(torch.from_numpy(rgb_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()

        # print(lidar_list.shape) #torch.Size([1, 1, 128, 128])
        # print(rgb_list.shape) #torch.Size([1, 3, 84,84])
        # print(speed_list.shape) #torch.Size([1, 2])


        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU 或 CPU
        # lidar_policy = lidar_policy.to(device)  # 确保模型在正确设备上
        # lidar_list = lidar_list.to(device)  # 确保输入数据也在同一设备

        # lidar_feature = lidar_policy(lidar_list)
        # print("Lidar feature before feeding into VLnet:", lidar_feature.shape)
        #seg = visual_policy(rgb_list)
        cad = lidar_list
        seg = rgb_list
        # print("seg ", seg.shape)
        v, a, logprob, mean = vl_policy(seg, cad, goal_list)
        
        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])

    else:
        v = None
        a = None
        scaled_action = None
        logprob = None

    return v, a, logprob, scaled_action

def generate_action_no_sampling(env, state_list, lidar_policy, visual_policy, vl_policy, action_bound):
    if env.index == 0:
        lidar_list, rgb_list, goal_list = [], [], []
        for i in state_list:
            lidar_list.append(i[0])
            rgb_list.append(i[1])
            goal_list.append(i[2])

        lidar_list = np.asarray(lidar_list)
        rgb_list = np.asarray(rgb_list)
        goal_list = np.asarray(goal_list)

        lidar_list = Variable(torch.from_numpy(lidar_list)).float().cuda()
        rgb_list = Variable(torch.from_numpy(rgb_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()

        # lidar_feature = lidar_policy(lidar_list)
        # print("Lidar feature before feeding into VLnet:", lidar_feature.shape)
        # seg = visual_policy(rgb_list)
        cad = lidar_list
        seg = rgb_list

        v, a, logprob, mean = vl_policy(seg, cad, goal_list)

        v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])

        mean = mean.data.cpu().numpy()
        scaled_action1 = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
    else:
        mean = None
        scaled_action = None

    return v, a, logprob, scaled_action, scaled_action1



def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs



def ppo_update_stage1(lidar_policy, visual_policy, vl_policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, rgb_size = (84,84),act_size=4):
    lidar, rgb, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    # print(lidar.shape) # (256, 1, 3, 512)



    lidar = lidar.reshape((num_step*num_env, 1, 128, 128))
    #print(rgb.shape)



    rgb = rgb.reshape((num_step*num_env, 3, rgb_size[0], rgb_size[1]))
    # print("rgbshape###########################",rgb.shape)

    speeds = speeds.reshape((num_step*num_env, 2))

    actions = actions.reshape(num_step*num_env, act_size)

    logprobs = logprobs.reshape(num_step*num_env, 1)

    advs = advs.reshape(num_step*num_env, 1)

    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            # print("sampled_lidar.shape@@@@@@@@@@@@@@@@@",sampled_lidar.shape)
            # print("type(lidar), type(rgb), type(speeds)",type(lidar), type(rgb), type(speeds))
            if isinstance(lidar, torch.Tensor):
                lidar = lidar.cpu().numpy()
            sampled_lidar = Variable(torch.from_numpy(lidar[index])).float().cuda()
            sampled_rgb = Variable(torch.from_numpy(rgb[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()

            # print(sampled_lidar.shape) # torch.Size([256, 3, 512])

            lidar = sampled_lidar
            #seg = visual_policy(sampled_rgb)
            seg = sampled_rgb
            
            new_value, new_logprob, dist_entropy = vl_policy.evaluate_actions(seg, lidar, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))

    print('update')


def ppo_update_stage2(policy, optimizer, batch_size, memory, filter_index, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    obss = np.delete(obss, filter_index, 0)
    goals = np.delete(goals, filter_index, 0)
    speeds = np.delete(speeds, filter_index, 0)
    actions = np.delete(actions, filter_index, 0)
    logprobs  = np.delete(logprobs, filter_index, 0)
    advs = np.delete(advs, filter_index, 0)
    targets = np.delete(targets, filter_index, 0)


    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=True)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))



    print('filter {} transitions; update'.format(len(filter_index)))

