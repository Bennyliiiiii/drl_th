#!/usr/bin/env python
# -*- coding: UTF-8 -*-

""" main script """

import time
import rospy
import copy
import numpy as np
import cv2
import sys
import math
import os
import logging
import socket
import matplotlib.pyplot as plt
import tf
import carla
import random
# from mpi4py import MPI
from torch.optim import Adam
from torch.autograd import Variable
from collections import deque
import torch
import torch.nn as nn
import scipy.misc as m
from network.erfnet import erfnet

# sys.path.append("/home/rzh/UnrealEngine_4.22/carla/Unreal/CarlaUE4/Content/Carla/carla-ros-bridge/ros-bridge/")
# sys.path.append("..")

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import NavSatFix, Image, PointCloud2, LaserScan
from carla_msgs.msg import CarlaCollisionEvent, CarlaEgoVehicleInfo, CarlaLaneInvasionEvent
from cv_bridge import CvBridge, CvBridgeError
from rosgraph_msgs.msg import Clock
import sensor_msgs.point_cloud2 as pc2
import pickle
from visualization_msgs.msg import Marker

bridge = CvBridge()
road = [[15, 14, 13, 9, 8, 7, 1, 0], [12, 10, 11], [4, 5, 6, 18, 19]]


class CarlaEnv1():
    def __init__(self, index, visual_policy):
        rospy.init_node('CarlaEnv2', anonymous=None)
        self.visual_policy = visual_policy
        self.state3 = np.zeros((512, 3))
        self.image_state = np.zeros((88, 200, 3))
        self.beam_mum = 512
        self.pointclouds = None
        self.img = None
        self.segmentation = None
        self.scan = None
        self.scan_tmp = np.zeros(17)
        self.cad_points = []
        self.cad_ranges = None

        self.is_crashed = None
        self.crossed_lane_markings = None
        self.speed = None
        self.index = index
        self.car_id = None
        self.lane = None
        # self.off_road_time = deque([0,0,0])
        self.last_road_id = 0
        self.last_right_flag = 0
        self.host = rospy.get_param('/carla/host', '127.0.0.1')
        self.port = rospy.get_param('/carla/port', '2000')
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(2000.0)
        self.world = client.get_world()
        colors = [[128, 64, 128],
                  [232, 35, 244], ]
        self.label_colours = dict(zip(range(2), colors))
        self.valid_classes = [0, 70, 190, 250, 220, 153, 157, 128, 244, 107, 102]
        self.our_classes = [128, 157, 244]  # road    road_line     other

        self.bev_img_size = [512, 512]
        self.bev_range = [-10, -20, 30, 20]
        x_range = self.bev_range[2] - self.bev_range[0]
        y_range = self.bev_range[3] - self.bev_range[1]
        self.ego_anchor_pixel = [int(self.bev_img_size[1] * self.bev_range[3] / y_range - 1),
                                 int(self.bev_img_size[0] * self.bev_range[3] / x_range - 1)] # origin self.bev_range[2]
        self.voxel_size = [(self.bev_range[2]-self.bev_range[0])/self.bev_img_size[0], (self.bev_range[3]-self.bev_range[1])/self.bev_img_size[1]]
        self.goal_point = [157, 240]
        self.last_message_time = rospy.get_time()
        self.current_index = 2
        with open("./path/pointlrttown02_1.pickle", "rb") as f:
        # with open("./path/new_path10.pickle", "rb") as f:
            self.path = pickle.load(f)
        self.current_waypoint = self.path[self.current_index]
        self.num_points = len(self.path)
        self.old_distance = 10.1

        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0
        self.max_velocity = 10

        node_name = 'CarlaEnv_' + str(index)

        # -----------Publisher and Subscriber-------------
        self.filtered_seg_pub = rospy.Publisher("/filtered_road", Image, queue_size=1)

        # cmd_vel_topic = '/carla/ego' + str(index) + '/twist_cmd'
        cmd_vel_topic = '/carla/ego_vehicle/twist'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)


        image_state_topic = '/carla/ego_vehicle/rgb_front/image'  # front
        self.image_state_sub = rospy.Subscriber(image_state_topic, Image, self.image_callback)


        segmentation_state_topic = '/carla/ego_vehicle/semantic_segmentation_front_fenge/image'
        self.segmentation_state_sub = rospy.Subscriber(segmentation_state_topic, Image, self.segmentation_callback)


        location_state_topic = '/carla/ego_vehicle/gnss'
        self.gps_state_sub = rospy.Subscriber(location_state_topic, NavSatFix, self.location_callback)


        odom_topic = '/carla/ego_vehicle/odometry'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)


        laser_topic = '/carla/ego_vehicle/lidar'
        self.laser_sub = rospy.Subscriber(laser_topic, PointCloud2, self.laser_scan_callback)


        crash_topic = '/carla/ego_vehicle/collision'
        self.check_crash = rospy.Subscriber(crash_topic, CarlaCollisionEvent, self.crash_callback)


        lane_topic = '/carla/ego_vehicle/lane_invasion'
        self.lane_crash = rospy.Subscriber(lane_topic, CarlaLaneInvasionEvent, self.lane_callback, queue_size=1)


        info_topic = '/carla/ego_vehicle/vehicle_info'
        self.check_crash = rospy.Subscriber(info_topic, CarlaEgoVehicleInfo, self.vehicle_info_callback)


        cad_topic = '/cad_carla_gt'
        self.cad_sub = rospy.Subscriber(cad_topic, LaserScan, self.cad_callback)
        self.img_pub = rospy.Publisher("cad_img", Image, queue_size=10)

        waypoint_topic = '/sub_waypoint'
        self.waypoint_pub = rospy.Publisher(waypoint_topic, Marker, queue_size=10)

        self.sim_clock = rospy.Subscriber('/clock', Clock, self.sim_clock_callback)
        self.bridge = CvBridge()

        while self.car_id is None or self.speed is None or self.odom_location is None:
            pass

        rospy.sleep(1)

    def cad_callback(self, cad_data):
        self.cad_ranges = cad_data.ranges

        angle_min = cad_data.angle_min
        angle_max = cad_data.angle_max
        angle_increment = cad_data.angle_increment
        ranges = cad_data.ranges
        angles = np.arange(angle_min, angle_max, angle_increment)
        new_lidar_points = []
        for r, angle in zip(ranges, angles):
            new_lidar_points.append([r * np.cos(angle), r * np.sin(angle)])
        self.cad_points = new_lidar_points

        # 183-200
        # self.scan_tmp = np.zeros(17)
        scan_data = np.array(cad_data.ranges, dtype=np.float32)
        self.scan_tmp = np.min(scan_data[183:200])
        # self.scan_tmp = np.min(scan_data[186:197])

    def get_cad_bev(self):
        if self.cad_ranges is None:
            rospy.loginfo("No CAD perception !")
            return

        bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype=np.uint8)
        # bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1]), dtype=np.uint8)

        cad_points = []

        for idx in range(len(self.cad_ranges)):
            r = self.cad_ranges[idx] / self.voxel_size[0]
            theta = (0.5 + idx) * (2 * np.pi / 384) - np.pi
            cad_points.append([-r * np.sin(theta), -r * np.cos(theta)])

        cad_points = np.array(cad_points, dtype=np.int32)

        cad_points[:, 0] = cad_points[:, 0] + self.ego_anchor_pixel[0]
        cad_points[:, 1] = cad_points[:, 1] + self.ego_anchor_pixel[1]

        cv2.drawContours(bev_img, [cad_points], -1, (255, 255, 255), -1)
        gray_bev_img = cv2.cvtColor(bev_img, cv2.COLOR_RGB2GRAY)
        bev_resized = cv2.resize(gray_bev_img, (128, 128))
        bev_resized = bev_resized / 255.0
        # print("bev_resized shape", bev_resized.shape)
        bev_resized = np.expand_dims(bev_resized, axis=-1)  # 变成 (H, W, 1)
        bev_resized = np.transpose(bev_resized, (2, 0, 1))

        cad_bev_img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
        # cv2.drawContours(bev_img, [cad_points], -1, 255, -1)
        # cad_bev_img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="mono8")
        self.img_pub.publish(cad_bev_img_msg)

        return bev_resized


    def image_callback(self, image_data):

        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError as e:
            print(e)
        self.img = np.asarray(image_data, dtype=np.float32)

    def segmentation_callback(self, image_data):
        # print(image_data.shape)
        try:
            image_type = image_data.encoding
            image_data = bridge.imgmsg_to_cv2(image_data, image_type)
        except CvBridgeError as e:
            print(e)
        self.segmentation = np.asarray(image_data, dtype=np.float32)  # float32

    def laser_scan_callback(self, scan):
        # print(scan.width,scan.height)
        lidar = pc2.read_points(scan, field_names=("x", "y", "z"))  # ,field_names=("x", "y", "z"),skip_nans=False
        self.lidar_points = np.array(list(lidar))
        # points = points[np.where([zmin <= point[2] <= zmax for point in points])]

    def odometry_callback(self, odometry):
        Quaternious = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.odom_location = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

        self.last_message_time = rospy.get_time()

    def location_callback(self, NavSatFix):
        pass

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.other_actor_id

    def lane_callback(self, data):
        self.crossed_lane_markings = data.crossed_lane_markings

    def vehicle_info_callback(self, info):
        self.car_id = info.id

    def get_image_observation(self, count):
        img = copy.deepcopy(self.img)
        # image_name = './picture/'+str(count) + '.png'
        # cv2.imwrite(image_name, img)

        img = img[:, :, 0:3]
        img = np.array(img, dtype=np.uint8)
        img = m.imresize(img, (84, 84))
        img = img.astype(np.float64)
        img = img.astype(float) / 255.0

        '''
        f, axarr = plt.subplots(1, 1)
        axarr.imshow(img)
        plt.show()'''
        img = np.transpose(img, (2, 0, 1))
        # print(image_data.shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def decode_segmap1(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, 2):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)
        # rgb[:, :, 2] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 0] = b / 255.0
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

    def get_image_observation_rgb2seg(self, count):
        # filtered segmentation img
        seg_img = self.segmentation
        road_bgr = (128, 64, 128)
        lane = (50, 234, 157)
        background = (232, 35, 244)

        mask_road = cv2.inRange(seg_img[:, :, :3], road_bgr, road_bgr)
        mask_lane = cv2.inRange(seg_img[:, :, :3], lane, lane)

        mask_total = np.logical_or(mask_road == 255, mask_lane == 255)

        mask = np.zeros((240, 320), dtype=np.float32)

        mask_b = mask.copy()
        mask_g = mask.copy()
        mask_r = mask.copy()

        mask_b[mask_total] = road_bgr[0]
        mask_g[mask_total] = road_bgr[1]
        mask_r[mask_total] = road_bgr[2]

        mask_b[mask_total == False] = background[0]
        mask_g[mask_total == False] = background[1]
        mask_r[mask_total == False] = background[2]
        # print(mask_r.shape)
        bgr = np.zeros((seg_img.shape[0], seg_img.shape[1], 3), dtype=np.float32)
        # print(bgr.shape)
        bgr[:, :, 2] = mask_r
        bgr[:, :, 1] = mask_g
        bgr[:, :, 0] = mask_b

        published_bgr = bgr.copy()
        img_msg = bridge.cv2_to_imgmsg(published_bgr.astype(np.uint8), "bgr8")
        self.filtered_seg_pub.publish(img_msg)
        # print("publishing seg fitered ...")

        # bridge.cv2_to_imgmsg()
        seg_resized = cv2.resize(bgr, (84, 84))  # resized the segmentation image into (84, 84)
        seg_resized.astype(np.float32) / 255.0
        seg = np.transpose(seg_resized, (2, 0, 1))
        return seg

    def get_image_observation_rgb2seg_backup(self, count):  # seg = visual_policy(rgb_list)
        img = copy.deepcopy(self.img)
        # image_name = './picture/'+str(count) + '.png'
        # cv2.imwrite(image_name, img)
        # print(img.shape)
        img = img[:, :, 0:3]
        img = np.array(img, dtype=np.uint8)
        # img = m.imresize(img, (84, 84))
        img = cv2.resize(img, (84, 84))
        img = img.astype(np.float64)

        img = img.astype(float) / 255.0
        '''
        f, axarr = plt.subplots(1, 1)
        axarr.imshow(img)
        plt.show()'''

        img = np.transpose(img, (2, 0, 1))
        # print(image_data.shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # rgb -->> seg
        # print(img.shape)
        img = torch.from_numpy(img).float().unsqueeze(0).cuda()
        pred = self.visual_policy(img)

        seg_pred = pred.cpu().data.max(1)[1].numpy().squeeze(0)
        seg = self.decode_segmap1(seg_pred)  # bgr
        image_name = './picture/' + str(count) + '.png'
        # print(f"seg img shape {seg.shape}, {seg[:, 0]}")

        cv2.imwrite(image_name, seg)
        '''f, axarr = plt.subplots(1, 1)
        axarr.imshow(seg)
        plt.show()'''
        seg = np.transpose(seg, (2, 0, 1))
        return seg

    def get_laser_observation1(self):
        scan = copy.deepcopy(self.lidar_points)
        return scan

    def get_laser_observation(self):
        scan = copy.deepcopy(self.lidar_points)
        scan[np.isnan(scan)] = 20.0
        scan[np.isinf(scan)] = 20.0
        scan1 = 20 * np.ones((16000), dtype=np.float32)
        scan2 = np.zeros((8001), dtype=np.float32)
        theta = np.zeros((512), dtype=np.float32)
        for i in range(scan.shape[0]):
            m = math.sqrt(math.pow(scan[i][0], 2) + math.pow(scan[i][1], 2))
            n = math.atan(scan[i][1] / (scan[i][0] + 1e-10))
            n = n * 180 / math.pi
            if (scan[i][0] < 0) and (scan[i][1] > 0):
                n = n + 180
            if (scan[i][0] < 0) and (scan[i][1] < 0):
                n = n + 180
            if (scan[i][0] > 0) and (scan[i][1] < 0):
                n = n + 360
            p = int(round(n / 0.0225))
            if p == 16000:
                p = 0
            scan1[p] = m
        for i in range(8001):
            scan2[i] = scan1[i - 4000]

        for j in range(512):
            theta[j] = np.pi - np.pi / 512 * j
        r = scan2

        scan2[np.isnan(scan2)] = 6.0
        scan2[np.isinf(scan2)] = 6.0
        scan2 = np.array(scan2)
        scan2[scan2 > 6] = 6.0

        raw_beam_num = len(scan2)  # 8001
        sparse_beam_num = self.beam_mum  # 512
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan2[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan2[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        '''
        ax = plt.subplot(111, projection='polar')
        c = ax.scatter(theta, scan_sparse)
        plt.show()'''

        return scan_sparse / 6.0 - 0.5

    def get_self_speed(self):
        return self.speed

    def get_self_odom_location(self):
        return self.odom_location

    def get_crash_state(self):
        if self.is_crashed != None:
            print('self.is_crashed:', self.is_crashed)
            self.is_crashed = None
            return True
        else:
            return False

    def get_lane_state(self):

        if self.crossed_lane_markings != None:
            if self.crossed_lane_markings[0] is CarlaLaneInvasionEvent.LANE_MARKING_OTHER or self.crossed_lane_markings[
                0] is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                self.crossed_lane_markings = None
                return 1
            elif self.crossed_lane_markings[0] is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                self.crossed_lane_markings = None
                return 2
            else:
                self.crossed_lane_markings = None
                return 3
        else:
            return 1

    def get_goal(self, alpha):
        [x, y, theta] = self.get_self_odom_location()
        goal = [x + 3 * np.cos(theta + alpha), y + 3 * np.sin(theta + alpha)]
        return goal

    # def generate_goal_point(self):
    #     [x_g, y_g] = self.generate_random_goal()
    #     self.goal_point = [x_g, y_g]
    #     [x, y] = self.get_local_goal()

    #     self.pre_distance = np.sqrt(x ** 2 + y ** 2)
    #     self.distance = copy.deepcopy(self.pre_distance)

    # 将世界坐标系转化为机器人坐标系
    def get_local_goal(self):
        [x, y, theta] = self.get_self_odom_location()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def get_waypoint_next(self):
        # self.current_waypoint = self.path[self.current_index]
        [x, y, theta] = self.get_self_odom_location()
        pose1 = self.path[self.current_index]
        # print(pose)
        pose1_x = pose1.pose.position.x
        pose1_y = pose1.pose.position.y
        if self.current_index == 9:
            if y > -197.5:
                self.current_index += 1
                self.old_distance = 10.1
        elif self.current_index < self.num_points - 1:
            pose2 = self.path[self.current_index + 1]
            pose2_x = pose2.pose.position.x
            pose2_y = pose2.pose.position.y
            if abs(pose1_x - pose2_x) >= abs(pose1_y - pose2_y):
                if pose1_x < x < pose2_x or pose2_x < x < pose1_x:
                    self.current_index += 1
                    self.old_distance = 10.1
            else:
                if pose1_y < y < pose2_y or pose2_y < y < pose1_y:
                    self.current_index += 1
                    self.old_distance = 10.1

        # if angle <= -1.5 or angle >= 1.5:
        #     self.current_index += 1
        self.current_waypoint = self.path[self.current_index]
        # print(self.current_waypoint)
        pub_waypoint = Marker()
        # pub_waypoint = self.current_waypoint
        # pub_waypoint.header.seq = 0
        pub_waypoint.header.stamp = rospy.Time.now()
        pub_waypoint.header.frame_id = "map"

        pub_waypoint.ns = "pose_marker"
        pub_waypoint.id = 0
        pub_waypoint.type = Marker.SPHERE
        pub_waypoint.action = Marker.ADD

        pub_waypoint.pose.position = self.current_waypoint.pose.position
        pub_waypoint.pose.orientation = self.current_waypoint.pose.orientation
        pub_waypoint.scale.x = 0.5  # Size of the point
        pub_waypoint.scale.y = 0.5
        pub_waypoint.scale.z = 0.5
        pub_waypoint.color.r = 1.0
        pub_waypoint.color.g = 0.0
        pub_waypoint.color.b = 0.0
        pub_waypoint.color.a = 1.0
        self.waypoint_pub.publish(pub_waypoint)
        return self.current_waypoint

    def get_waypoint_polar(self):
        [x, y, theta] = self.get_self_odom_location()
        pose = self.get_waypoint_next()
        # print(pose)
        ahead_x = pose.pose.position.x
        ahead_y = pose.pose.position.y
        dx = ahead_x - x
        dy = ahead_y - y

        local_x = dx * np.cos(-theta) - dy * np.sin(-theta)
        local_y = dx * np.sin(-theta) + dy * np.cos(-theta)
        # print(local_x, local_y)

        max_x = 10.1
        min_x = 0
        n_x = 2 * (local_x - min_x) / (max_x - min_x) - 1
        max_y = 3
        min_y = -3
        n_y = 2 * (local_y - min_y) / (max_y - min_y) - 1

        return n_x, n_y

    def get_dis_and_angle(self):
        # self.current_waypoint = self.path[self.current_index]
        [x, y, theta] = self.get_self_odom_location()
        # pose = self.current_waypoint
        pose = self.get_waypoint_next()
        # print(pose)
        ahead_x = pose.pose.position.x
        ahead_y = pose.pose.position.y
        dx = ahead_x - x
        dy = ahead_y - y
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        x_vehicle = cos_theta * dx + sin_theta * dy
        y_vehicle = -sin_theta * dx + cos_theta * dy
        # 计算极坐标
        distance = math.sqrt(x_vehicle ** 2 + y_vehicle ** 2)
        angle = math.atan2(y_vehicle, x_vehicle)
        return distance, angle

    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0] * 2  # action[0] * 2
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)
        # print(f"action: {action}, angular.z: {move_cmd.angular.z}")
        rospy.sleep(1e-5)

    def get_reward_and_terminate(self, t, id):
        done = False  
        collsion = False  
        collsion = self.get_crash_state()   
        lane = 1  
        lane = self.get_lane_state()        
        # print('collision:',collsion)
        [x, y, theta] = self.get_self_odom_location() 
        [v, w] = self.get_self_speed() 
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2) 
        reward_time = -0.1  
        reward_c = 0  
        reward_c1 = 0  
        reward_c2 = 0  
        reward_v = 0  
        reward_w = 0  
        result = 0  
        right_flag = 0  

        # if 157 - x < 1:
        #     done = True
        #     result = 'success'

        if lane == 3 and t < 8: print('reset is on Sidewalk line:', lane)
        if lane == 2:
            result = 'Cross center line'
            # reward_c1 = -0.5
            # print(result)
        elif lane == 3:
            done = True
            reward_c1 = -5.0
            result = 'Cross Sidewalk line'
        '''
        left = self.off_road_time.popleft()
        if lane ==3 and t >= 8:
            self.off_road_time.append(1)
        else :
            self.off_road_time.append(0)
        if np.sum(self.off_road_time) > 2:
            done = True
            reward_c1 = -10
            result = 'Cross Sidewalk line'
        '''

        a = self.world.get_actor(self.car_id)
        waypoint = self.world.get_map().get_waypoint(a.get_location())
        road_id = waypoint.road_id
        if road_id in road[0]:
            if theta > -np.pi and theta < 0:
                right_flag = 1
            else:
                right_flag = -1
        if road_id in road[1]:
            if theta > -0.5 * np.pi and theta < 0.5 * np.pi:
                right_flag = 1
            else:
                right_flag = -1
        if road_id in road[2]:
            if theta > -0.5 * np.pi and theta < 0.5 * np.pi:
                right_flag = -1
            else:
                right_flag = 1

        if waypoint.lane_id == right_flag or right_flag == 0:
            # print ("right")
            pass
        else:
            reward_c2 = -0.01    # 0.1
            # reward_c2 = -1 * (self.distance / 10) 
            # print ("left")

        if collsion:
            done = True
            reward_c = -5
            result = 'Crashed'

        reward_v = max(0, v - 0.5)
        self.reward1 += 0.1 * reward_v


        # reward_w = -min(0.5, 0.5 * w ** 2)

        reward_w = -0.3 * abs(w) if abs(w) < 1.0 else -1.0 

        self.reward2 += reward_w

        # print(f"road_id: {road_id}, theta: {theta}, right_flag: {right_flag}")


        if x < 120 and y > -243:
            if y < -242 or y > -237.5:
                done = True
                result = 'over boundary1'
        elif y < -235.5 and x > 132.5:  # x > 140
            done = True
            result = 'over boundary2'
        elif y > -196.4 and x < 132.5:
            done = True
            result = 'over boundary3'
        elif y < -192.5 and x > 186:  # y < -193.7
            done = True
            result = 'over boundary4'
        elif x > 195.0:
            done = True
            result = 'over boundary5'

        r_waypoint1 = 0
        distance, angel = self.get_dis_and_angle()
        if distance < 1.0:           # 0.5
            r_waypoint1 = 3         # 6
        self.reward3 += r_waypoint1
        r_waypoint2 = 0
        if t % 10 == 0:
            self.old_distance = distance
        r_waypoint2 = 3 * (self.old_distance - distance)    # 1
        # print(r_waypoint2)
        if abs(r_waypoint2) > 2:            # 0.9
            r_waypoint2 = 0
        self.reward4 += r_waypoint2

        if t > 950:
            done = True
            result = 'Time out'

        if np.sqrt((192.5 - x) ** 2 + (-152.1 - y) ** 2) < 5:
            done = True
            result = 'success'
        if self.current_index == 19 and y > -160:
            done = True
            result = 'success1111'

        if self.last_road_id == road_id and self.last_right_flag != right_flag and self.last_right_flag != 0 and t > 3:
            done = True
            result = 'reverse'
            reward_c2 = -2.0  
        self.last_right_flag = right_flag
        self.last_road_id = road_id

        reward = reward_c + reward_c1 + reward_c2 + reward_v* 0.1 + reward_w + r_waypoint1 + r_waypoint2

        if done is True:
            if id % 2 == 0:
                self.current_index = 2
            else:
                self.current_index = 9
            self.old_distance = 10.1
            print("reward1:", self.reward1, "reward2:", self.reward2, "reward3:", self.reward3, "reward4:", self.reward4)
            self.reward1 = 0
            self.reward2 = 0
            self.reward3 = 0
            self.reward4 = 0

        return reward, done, result
    





    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)

    def reset_pose_two(self, id):
        # self.lane = False
        self.control_vel([0, 0])
        rospy.sleep(0.15)
        a = self.world.get_actor(self.car_id)
        if id % 2 == 0:
            location = carla.Location(x=92, y=240, z=0)  # 94
            rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
            self.current_index = 2
        else:
            self.current_index = 9
            location = carla.Location(x=135.4, y=205, z=0)
            rotation = carla.Rotation(pitch=0, yaw=-90, roll=0)

        spawn_point = carla.Transform(location, rotation)
        a.set_transform(spawn_point)
        rospy.sleep(0.05)
        b = a.get_transform()

        print('---start reset---')
        start = time.time()
        # 判断是否复位成功
        while np.abs(b.location.x - spawn_point.location.x) > 0.2 or np.abs(
                b.location.y - spawn_point.location.y) > 0.2 or b.location.z > 0.3:
            stop = time.time()
            if stop - start > 3:
                # self.reset_pose()
                print('break while')
                break
            b = a.get_transform()
        rospy.sleep(0.5)
        for i in range(10): collsion = self.get_crash_state()
        for i in range(10): lane = self.get_lane_state()
        print('---achieve reset---')
        rospy.sleep(0.5)


    def reset_pose(self, id):
        # self.lane = False
        self.control_vel([0, 0])
        rospy.sleep(0.15)
        a = self.world.get_actor(self.car_id)
        

        location = carla.Location(x=63, y=240, z=0)  
        rotation = carla.Rotation(pitch=0, yaw=0, roll=0)  
        self.current_index = 2  

        spawn_point = carla.Transform(location, rotation)
        a.set_transform(spawn_point)
        rospy.sleep(0.05)
        b = a.get_transform()

        print('---start reset---')
        start = time.time()

        while np.abs(b.location.x - spawn_point.location.x) > 0.2 or np.abs(
                b.location.y - spawn_point.location.y) > 0.2 or b.location.z > 0.3:
            stop = time.time()
            if stop - start > 3:
                print('break while')
                break
            b = a.get_transform()
        rospy.sleep(0.5)
        for i in range(10): collsion = self.get_crash_state()
        for i in range(10): lane = self.get_lane_state()
        print('---achieve reset---')
        rospy.sleep(0.5)

    def check_timeout(self):
        current_time = rospy.get_time()
        if (current_time - self.last_message_time) > 0.3:
            return False
        else:
            return True

    def get_distance(self, loc1, loc2):
        return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)
    def gen_barrier(self):
        ped_blueprints1 = self.world.get_blueprint_library().find("walker.pedestrian.0002")
        p_transform1 = carla.Transform(carla.Location(x=68, y=240, z=1),
                                       carla.Rotation(roll=0, pitch=0, yaw=180)) # 150
        self.ped1 = self.world.spawn_actor(ped_blueprints1, p_transform1)

        ped_blueprints2 = self.world.get_blueprint_library().find("walker.pedestrian.0004")
        p_transform2 = carla.Transform(carla.Location(x=70, y=240.5, z=1),
                                       carla.Rotation(roll=0, pitch=0, yaw=180)) # 150
        self.ped2 = self.world.spawn_actor(ped_blueprints2, p_transform2)

        ped_blueprints3 = self.world.get_blueprint_library().find("walker.pedestrian.0007")
        p_transform3 = carla.Transform(carla.Location(x= 80, y=240.6, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=180))
        self.ped3 = self.world.spawn_actor(ped_blueprints3, p_transform3)

        ped_blueprints4 = self.world.get_blueprint_library().find("walker.pedestrian.0005")
        p_transform4 = carla.Transform(carla.Location(x= 85, y=239.5, z=1), # y=190.7
                                    carla.Rotation(roll=0, pitch=0, yaw=180))
        self.ped4 = self.world.spawn_actor(ped_blueprints4, p_transform4)

        p_transform5 = carla.Transform(carla.Location(x= 95, y=240.6, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=180))
        self.ped5 = self.world.spawn_actor(ped_blueprints2, p_transform5)

        ped_blueprints6 = self.world.get_blueprint_library().find("walker.pedestrian.0006")
        p_transform6 = carla.Transform(carla.Location(x= 102, y=240.2, z=1), # y=191.8, 235.2
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped6 = self.world.spawn_actor(ped_blueprints6, p_transform6)

        p_transform7 = carla.Transform(carla.Location(x= 110, y=240, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped7 = self.world.spawn_actor(ped_blueprints3, p_transform7)

        ped_blueprints8 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform8 = carla.Transform(carla.Location(x=120, y=242, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped8 = self.world.spawn_actor(ped_blueprints8, p_transform8)

        ped_blueprints9 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform9 = carla.Transform(carla.Location(x=125, y=240.6, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped9 = self.world.spawn_actor(ped_blueprints9, p_transform9)

        ped_blueprints10 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform10 = carla.Transform(carla.Location(x=135, y=242, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped10 = self.world.spawn_actor(ped_blueprints10, p_transform10)

        ped_blueprints11 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform11 = carla.Transform(carla.Location(x=143, y=240.6, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped11 = self.world.spawn_actor(ped_blueprints11, p_transform11)


        ped_blueprints12 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform12 = carla.Transform(carla.Location(x=150, y=242, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped12 = self.world.spawn_actor(ped_blueprints12, p_transform12)

        ped_blueprints13 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform13 = carla.Transform(carla.Location(x=157, y=242, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped13 = self.world.spawn_actor(ped_blueprints13, p_transform13)

        ped_blueprints14 = self.world.get_blueprint_library().find("walker.pedestrian.0008")
        p_transform14 = carla.Transform(carla.Location(x=164, y=240.2, z=1),
                                    carla.Rotation(roll=0, pitch=0, yaw=90))
        self.ped14 = self.world.spawn_actor(ped_blueprints14, p_transform14)




        ve_blueprints1 = self.world.get_blueprint_library().find("vehicle.micro.microlino")
        ve_blueprints2 = self.world.get_blueprint_library().find("vehicle.tesla.model3")
        ve_blueprints4 = self.world.get_blueprint_library().find("vehicle.citroen.c3")

        v_transform1 = carla.Transform(carla.Location(x= 80, y=236.5, z=0.5),
                                    carla.Rotation(roll=-0.001, pitch=0.350, yaw=90)) # 175
        self.ve1 = self.world.spawn_actor(ve_blueprints1, v_transform1)

        v_transform2 = carla.Transform(carla.Location(x= 100, y=236.3, z=0.5),
                            carla.Rotation(roll=-0.001, pitch=0.350, yaw=180))
        self.ve2 = self.world.spawn_actor(ve_blueprints4, v_transform2)

        v_transform3 = carla.Transform(carla.Location(x= 120, y=235, z=0.5),
                            carla.Rotation(roll=-0.001, pitch=0.350, yaw=-90))
        self.ve3 = self.world.spawn_actor(ve_blueprints2, v_transform3)


    def det_barrier(self):
        ped1_location = self.ped1.get_location()
        ped2_location = self.ped2.get_location()
        ped3_location = self.ped3.get_location()
        ped4_location = self.ped4.get_location()
        ped5_location = self.ped5.get_location()
        ped6_location = self.ped6.get_location()
        ped7_location = self.ped7.get_location()
        ped8_location = self.ped8.get_location()
        # ve1_location = self.ve1.get_location()
        ve2_location = self.ve2.get_location()
        ve3_location = self.ve3.get_location()
        # ve4_location = self.ve4.get_location()

        vehicle = self.world.get_actor(135)
        vehicle_location = vehicle.get_location()

        pedestrain_control1 = carla.WalkerControl()
        pedestrain_control1.speed = -1.3
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control1.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control2 = carla.WalkerControl()
        pedestrain_control2.speed = -1.0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control2.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control3 = carla.WalkerControl()
        pedestrain_control3.speed = -1.2
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control3.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control4 = carla.WalkerControl()
        pedestrain_control4.speed = -1.2
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control4.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control5 = carla.WalkerControl()
        pedestrain_control5.speed = -1.2
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control5.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control6 = carla.WalkerControl()
        pedestrain_control6.speed = -1.05
        pedestrain_rotation = carla.Rotation(0, -90, 0)
        pedestrain_control6.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control7 = carla.WalkerControl()
        pedestrain_control7.speed = -1.05
        pedestrain_rotation = carla.Rotation(0, -90, 0)
        pedestrain_control7.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control8 = carla.WalkerControl()
        pedestrain_control8.speed = -1.05
        pedestrain_rotation = carla.Rotation(0, -90, 0)
        pedestrain_control8.direction = pedestrain_rotation.get_forward_vector()

        # target_velocity = carla.Vector3D(x=0.0, y=2.5, z=0.0)
        target_velocity2 = carla.Vector3D(x=-2.0, y=0.0, z=0.0)
        target_velocity3 = carla.Vector3D(x=0.0, y=-1.2, z=0.0)
        # target_velocity4 = carla.Vector3D(x=0.0, y=-1.5, z=0.0)

        if vehicle_location.x > 0 and ped1_location.x > 0 and ped2_location.x > 0 and ped3_location.x > 0 and ped4_location.x > 0 and ped5_location.x > 0 and ped6_location.x > 0 and ped7_location.x > 0 and ped8_location.x > 0 and ve2_location.x > 0 and ve3_location.x > 0:
            if self.get_distance(vehicle_location, ped1_location) < 15:
                self.ped1.apply_control(pedestrain_control1)
                self.ped2.apply_control(pedestrain_control2)
            if self.get_distance(vehicle_location, ped3_location) < 15:
                self.ped3.apply_control(pedestrain_control3)
                self.ped4.apply_control(pedestrain_control4)
                self.ped5.apply_control(pedestrain_control5)
            if self.get_distance(vehicle_location, ped6_location) < 10:
                self.ped6.apply_control(pedestrain_control6)
                self.ped7.apply_control(pedestrain_control7)
                self.ped8.apply_control(pedestrain_control8)

            # if self.get_distance(vehicle_location, ve1_location) < 16:
            #     self.ve1.set_target_velocity(target_velocity)
            if self.get_distance(vehicle_location, ve2_location) < 20:
                self.ve2.set_target_velocity(target_velocity2)
            if self.get_distance(vehicle_location, ve3_location) < 10:
                self.ve3.set_target_velocity(target_velocity3)
            # if self.get_distance(vehicle_location, ve4_location) < 6:
            #     self.ve4.set_target_velocity(target_velocity4)
    def v0_barrier(self):
        ped1_location = self.ped1.get_location()
        ped2_location = self.ped2.get_location()
        ped3_location = self.ped3.get_location()
        ped4_location = self.ped4.get_location()
        ped5_location = self.ped5.get_location()
        ped6_location = self.ped6.get_location()
        ped7_location = self.ped7.get_location()
        ped8_location = self.ped8.get_location()
        # ve1_location = self.ve1.get_location()
        ve2_location = self.ve2.get_location()
        ve3_location = self.ve3.get_location()
        # ve4_location = self.ve4.get_location()

        vehicle = self.world.get_actor(135)
        vehicle_location = vehicle.get_location()

        pedestrain_control1 = carla.WalkerControl()
        pedestrain_control1.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control1.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control2 = carla.WalkerControl()
        pedestrain_control2.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control2.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control3 = carla.WalkerControl()
        pedestrain_control3.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control3.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control4 = carla.WalkerControl()
        pedestrain_control4.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control4.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control5 = carla.WalkerControl()
        pedestrain_control5.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control5.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control6 = carla.WalkerControl()
        pedestrain_control6.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control6.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control7 = carla.WalkerControl()
        pedestrain_control7.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control7.direction = pedestrain_rotation.get_forward_vector()
        pedestrain_control8 = carla.WalkerControl()
        pedestrain_control8.speed = 0
        pedestrain_rotation = carla.Rotation(0, 0, 0)
        pedestrain_control8.direction = pedestrain_rotation.get_forward_vector()

        target_velocity = carla.Vector3D(x=0.0, y=0.0, z=0.0)

        if vehicle_location.x > 0 and ped1_location.x > 0 and ped2_location.x > 0 and ped3_location.x > 0 and ped4_location.x > 0 and ped5_location.x > 0 and ped6_location.x > 0 and ped7_location.x > 0 and ped8_location.x > 0 and ve2_location.x > 0 and ve3_location.x > 0:
            if self.get_distance(vehicle_location, ped1_location) < 15:
                self.ped1.apply_control(pedestrain_control1)
                self.ped2.apply_control(pedestrain_control2)
            if self.get_distance(vehicle_location, ped3_location) < 15:
                self.ped3.apply_control(pedestrain_control3)
                self.ped4.apply_control(pedestrain_control4)
                self.ped5.apply_control(pedestrain_control5)
            if self.get_distance(vehicle_location, ped6_location) < 10:
                self.ped6.apply_control(pedestrain_control6)
                self.ped7.apply_control(pedestrain_control7)
                self.ped8.apply_control(pedestrain_control8)
            # if self.get_distance(vehicle_location, ve1_location) < 16:
            #     self.ve1.set_target_velocity(target_velocity)
            if self.get_distance(vehicle_location, ve2_location) < 20:
                self.ve2.set_target_velocity(target_velocity)
            if self.get_distance(vehicle_location, ve3_location) < 10:
                self.ve3.set_target_velocity(target_velocity)
            # if self.get_distance(vehicle_location, ve4_location) < 6:
            #     self.ve4.set_target_velocity(target_velocity)

    def dis_barrier(self):
        print("################done#############")
        self.ped1.destroy()
        self.ped2.destroy()
        self.ped3.destroy()
        self.ped4.destroy()
        self.ped5.destroy()
        self.ped6.destroy()
        self.ped7.destroy()
        self.ped8.destroy()
        self.ped9.destroy()
        self.ped10.destroy()
        self.ped11.destroy()
        self.ped12.destroy()
        self.ped13.destroy()
        self.ped14.destroy()
        self.ve1.destroy()
        self.ve2.destroy()
        self.ve3.destroy()


if __name__ == '__main__':
    n_classes = 2
    visual_policy = erfnet(n_classes=n_classes)
    visual_policy.cuda()
    visual_policy_path = 'visual_models'
    visual_file = visual_policy_path + '/erfnet_CityScapes_class_2_496.pt'
    state_dict = torch.load(visual_file)
    visual_policy.load_state_dict(state_dict)
    # visual_policy = None
    env = CarlaEnv1(0, visual_policy)

    env.reset_pose()
    # env.get_image_observation_rgb2seg()
    print("start")
    i = 0
    while True:
        env.get_reward_and_terminate(i)
        time.sleep(0.2)
        # print(i)
        i += 1




    # g = [100, 100]
    # env.reset()
    '''while True:

        env.get_crash_state()
        env.control_vel([1,0])'''
    # b = env.get_laser_observation()
    '''
    theta = np.zeros((512), dtype=np.float32)
    for j in range(512):
        theta[j] = np.pi - np.pi / 512 * j
    ax = plt.subplot(211, projection='polar')
    c = ax.scatter(theta, a)
    ax1 = plt.subplot(212, projection='polar')
    d = ax1.scatter(theta, b)
    plt.show()
    '''
    # while True: env.step([[0, 1]], g)

    '''
    i = 0
    while True:
        env.control_vel(1,0)
        print(env.get_self_speed())
        coll = env.get_crash_state()
        if coll:
            print(i)
            i+=1

    img = env.get_image_observation()
    cv2.imwrite('a.png', img)
    crash = env.get_crash_state()
    print('crash :', crash)


    print(env.get_self_odom_location())

    scan_sparse = env.get_laser_observation()
    print(scan_sparse.shape)

    points = env.get_laser_observation1()
    point = 20 * np.ones((16000), dtype=np.float32)
    theta = np.zeros((16000), dtype=np.float32)
    np.savetxt("filename1.txt", points)
    for i in range(points.shape[0]):
        m = math.sqrt(math.pow(points[i][0], 2) + math.pow(points[i][1], 2))
        n = math.atan(points[i][1] / points[i][0])
        n = n * 180 / math.pi
        if (points[i][0] < 0) and (points[i][1] > 0):
            n = n + 180
        if (points[i][0] < 0) and (points[i][1] < 0):
            n = n + 180
        if (points[i][0] > 0) and (points[i][1] < 0):
            n = n + 360

        p = int(round(n / 0.0225))
        point[p] = m
        points[i][0] = m
        points[i][1] = n

    print(point)
    np.savetxt("filename.txt", point)

    for j in range(point.shape[0]):
        theta[j] = (2 * np.pi / 16000 * j + np.pi / 2) % (2 * np.pi)
    r = point
    ax = plt.subplot(111, projection='polar')

    c = ax.scatter(theta, r)

    plt.show()
    '''

