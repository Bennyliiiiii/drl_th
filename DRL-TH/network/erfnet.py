#!/usr/bin/python
# -*- coding: UTF-8 -*-

import time
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import scipy.misc as m
import matplotlib.pyplot as plt

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super(non_bottleneck_1d, self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(16, 0.03, 1))

        self.layers.append(DownsamplerBlock(16, 64))

        self.layers.append(non_bottleneck_1d(64, 0.3, 2))
        self.layers.append(non_bottleneck_1d(64, 0.3, 4))
        self.layers.append(non_bottleneck_1d(64, 0.3, 8))
        self.layers.append(non_bottleneck_1d(64, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(64, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output

class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class erfnet(nn.Module):
    def __init__(self, n_classes, encoder=None, pretrained=False):  # use encoder to pass pretrained encoder
        super(erfnet, self).__init__()

        if (encoder == None):
            self.encoder = Encoder(n_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(n_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)



def transform(img ,img_size):
    img = m.imresize(img, (img_size[0], img_size[1]))  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img = img.astype(float) / 255.0
        # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()

    return img
colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
label_colours = dict(zip(range(19), colors))
def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    #rgb[:, :, 0] = r
    #rgb[:, :, 1] = g
    #rgb[:, :, 2] = b
    return rgb





colors1 = [  # [  0,   0,   0],
        [128, 64, 128],
        [107, 142, 35],
    ]
label_colours1 = dict(zip(range(2), colors1))

def decode_segmap1(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours1[l][0]
        g[temp == l] = label_colours1[l][1]
        b[temp == l] = label_colours1[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    #rgb[:, :, 0] = r
    #rgb[:, :, 1] = g
    #rgb[:, :, 2] = b
    return rgb


def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.
    
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
        
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    print(img.shape[0] - crop_heigth)
    img = img[img.shape[0] - crop_heigth: img.shape[0],
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2), :]
    return img

def tou(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    a = temp.copy()
    for l in range(0, 2):
        r[temp == l] = 255
        g[temp == l] = 252
        b[temp == l] = 153
        a[temp == l] = 120

    rgb = np.zeros((temp.shape[0], temp.shape[1], 4), dtype=np.float32)
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    rgb[:, :, 3] = a / 255.0
    #rgb[:, :, 0] = r
    #rgb[:, :, 1] = g
    #rgb[:, :, 2] = b
    return rgb




if __name__ == '__main__':
    
    n_classes = 2
    model = erfnet(n_classes=n_classes)
    model.cuda(device=0) 
    model = torch.nn.DataParallel(model,device_ids = [0])
    model_path ="/home/dh/visual_lidar_collision_avoidance/visual_models/erfnet_CityScapes_class_2_200.pt"
    # model.init_vgg16()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    observation = torch.randn(1,3, 88,200).cuda()
    v = model(observation)
    print(v.shape)
    #img_path ="/home/dh/Documents/semseg/dataset/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
    '''
    img_path ="/home/dh/Documents/semseg/real_img/IMG_20190926_150639.jpg"
    img_0 = m.imread(img_path)
    img_1 = m.imresize(img_0 , (88,200))
    img = np.array(img_0, dtype=np.uint8)
    img = transform(img,(88,200))
    x = img.unsqueeze(0).cuda()

    # ---------------------------fcn32s模型运行时间-----------------------
    start = time.time()
    pred = model(x)
    
    train_pred = pred.cpu().data.max(1)[1].numpy().squeeze(0)
    stop = time.time()
    print("time is : ",stop - start)
    print(train_pred.shape)
    out = decode_segmap1(train_pred)
    #print(out)
    f, axarr = plt.subplots(2, 1)
    axarr[0].imshow(img_1)
    axarr[1].imshow(out)
    plt.show()
    time.sleep(1)
    '''
