import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from .utils import log_normal_density

class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)


class CNNPolicy(nn.Module):
    def __init__(self, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        
        self.act_fea_cv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)  # [1, 1, 128, 128] -> [1, 32, 64, 64]
        self.act_fea_cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # [1, 32, 64, 64] -> [1, 64, 32, 32]
        self.act_fea_cv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # [1, 64, 32, 32] -> [1, 128, 16, 16]

        self.act_fc1 = nn.Linear(128 * 16 * 16, 256)  
        self.act_fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_space)

        # （Critic）
        self.crt_fea_cv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.crt_fea_cv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)

        self.crt_fc1 = nn.Linear(128 * 16 * 16, 256)
        self.crt_fc2 = nn.Linear(256, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        """返回价值估计、动作"""
      
        a = F.relu(self.act_fea_cv1(x)) 
        a = F.relu(self.act_fea_cv2(a))  
        a = F.relu(self.act_fea_cv3(a))  
        a = a.view(a.shape[0], -1)  
        a = F.relu(self.act_fc1(a))
        # a = F.relu(self.act_fc2(a))
        # action = self.actor(a)

        return a


if __name__ == '__main__':
    net = CNNPolicy(2)  
    net.cuda() 
    observation = torch.randn(1, 1, 128, 128).cuda()  
    output = net(observation)
    print(output.shape)  



# if __name__ == '__main__':

#     net = CNNPolicy(3, 2)
#     net.cuda() 
#     lidar_policy_path = '/home/dh/visual_lidar_collision_avoidance/lidar_models/'
#     lidar_file = lidar_policy_path + 'stage2.pth'
#     state_dict = torch.load(lidar_file)
#     net.load_state_dict(state_dict)
#     net.eval()
#     observation = torch.randn(1,3, 512).cuda()
#     v = net(observation)
#     print(v.shape)


