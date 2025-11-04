import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from network.utils import log_normal_density

class VLnet(nn.Module):
    def __init__(self, input_channel , action_space):
        super(VLnet, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4) #, padding=1
        self.act_fea_cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.act_fea_cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,stride=1)  

        self.act_lidar_conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=4) 
        self.act_lidar_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.act_lidar_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.act_lidar_fc1 = nn.Linear(5408, 1024)  
        self.act_lidar_fc2 = nn.Linear(1024, 512)  
        self.act_lidar_fc3 = nn.Linear(512, 256)  
 
        self.act_fc1 = nn.Linear(7*7*64, 512)
        self.act_fc11 = nn.Linear(512, 256)
        self.act_goal_fc = nn.Linear(2, 16)
        self.act_goal_fc2 = nn.Linear(16, 64)
        self.act_goal_fc3 = nn.Linear(64,128)   
        self.act_fc2 = nn.Linear(256+256+128, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        # self.crt_fea_cv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4)
        # self.crt_fea_cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # self.crt_fea_cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # self.crt_fc1 = nn.Linear(7*7*64, 512)
        # self.crt_fc2 = nn.Linear(512+128, 128)
        self.cri_fea_cv1 = nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=8, stride=4) #, padding=1
        self.cri_fea_cv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.cri_fea_cv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,stride=1)  

        self.cri_lidar_conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=4)  
        self.cri_lidar_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.cri_lidar_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.cri_lidar_fc1 = nn.Linear(5408, 1024)  
        self.cri_lidar_fc2 = nn.Linear(1024, 512)   
        self.cri_lidar_fc3 = nn.Linear(512, 256) 
 
        self.cri_fc1 = nn.Linear(7*7*64, 512)
        self.cri_fc11 = nn.Linear(512, 256)
        self.cri_goal_fc = nn.Linear(2, 16)
        self.cri_goal_fc2 = nn.Linear(16, 64)
        self.cri_goal_fc3 = nn.Linear(64,128)   
        self.cri_fc2 = nn.Linear(256+256+128, 128)
        self.critic = nn.Linear(128, 1)



    def forward(self, visual_input ,lidar_input, goal):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        # print("Lidar input shape:", lidar_input.shape)
        # print("goal input shape:", goal.shape)
        # print("visual_input shape:", visual_input.shape)
        a = F.relu(self.act_fea_cv1(visual_input))
        a = F.relu(self.act_fea_cv2(a))
        a = F.relu(self.act_fea_cv3(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = F.relu(self.act_fc11(a))
        # print("a input shape:", a.shape)
        goal_out = F.relu(self.act_goal_fc(goal))
        goal_out = F.relu(self.act_goal_fc2(goal_out))
        goal_out = F.relu(self.act_goal_fc3(goal_out))
        # print("goal_out shape:", goal_out.shape)

        conv1_out = F.relu(self.act_lidar_conv1(lidar_input))
        conv2_out = F.relu(self.act_lidar_conv2(conv1_out))
        conv3_out = F.relu(self.act_lidar_conv3(conv2_out))
        # print("Conv1 output shape:", conv1_out.shape)
        # print("Conv2 output shape:", conv2_out.shape)
        # print("Conv3 output shape:", conv3_out.shape)
        lidar_out = conv3_out.view(conv3_out.shape[0], -1)  
        # print("lidar_out shape before fc:", lidar_out.shape)
        lidar_out = F.relu(self.act_lidar_fc1(lidar_out)) 
        lidar_out = F.relu(self.act_lidar_fc2(lidar_out)) 
        lidar_out = F.relu(self.act_lidar_fc3(lidar_out)) 
        # lidar_out = F.relu(self.lidar_fc3(lidar_out)) 
        # print("lidar_out shape after fc:", lidar_out.shape)
        a = torch.cat((a, lidar_out, goal_out), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))    
        mean = torch.cat((mean1, mean2), dim=-1)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        #print(mean,std)
        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)




        b = F.relu(self.cri_fea_cv1(visual_input))
        b = F.relu(self.cri_fea_cv2(b))
        b = F.relu(self.cri_fea_cv3(b))
        b = b.view(b.shape[0], -1)
        b = F.relu(self.cri_fc1(b))
        b = F.relu(self.cri_fc11(b))
        # print("a input shape:", a.shape)
        goal_out_cri = F.relu(self.cri_goal_fc(goal))
        goal_out_cri = F.relu(self.cri_goal_fc2(goal_out_cri))
        goal_out_cri = F.relu(self.cri_goal_fc3(goal_out_cri))
        # print("goal_out shape:", goal_out.shape)

        conv1_out_cri = F.relu(self.cri_lidar_conv1(lidar_input))
        conv2_out_cri = F.relu(self.cri_lidar_conv2(conv1_out_cri))
        conv3_out_cri = F.relu(self.cri_lidar_conv3(conv2_out_cri))
        # print("Conv1 output shape:", conv1_out.shape)
        # print("Conv2 output shape:", conv2_out.shape)
        # print("Conv3 output shape:", conv3_out.shape)
        lidar_out_cri = conv3_out.view(conv3_out_cri.shape[0], -1)  
        # print("lidar_out shape before fc:", lidar_out.shape)
        lidar_out_cri = F.relu(self.cri_lidar_fc1(lidar_out_cri)) 
        lidar_out_cri = F.relu(self.cri_lidar_fc2(lidar_out_cri))
        lidar_out_cri = F.relu(self.cri_lidar_fc3(lidar_out_cri)) 
        # lidar_out = F.relu(self.lidar_fc3(lidar_out)) 
        # print("lidar_out shape after fc:", lidar_out.shape)


        b = torch.cat((b, lidar_out_cri, goal_out_cri), dim=-1)
        b = F.relu(self.cri_fc2(b))

        v = self.critic(b)

        return v, action, logprob, mean

    def evaluate_actions(self, visual_input ,lidar_input , goal_input, action):
        v, _, _, mean = self.forward(visual_input, lidar_input, goal_input)
        # logstd = self.logstd.expand_as(mean)
        logstd = self.logstd.view(1, -1).expand_as(mean)  
        
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy
