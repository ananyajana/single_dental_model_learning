#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
https://github.com/AnTao97/dgcnn.pytorch/tree/97785863ff7a82da8e2abe0945cf8d12c9cc6c18
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_cluster import fps
from torch_cluster import knn as torch_cluster_knn
from helper_tool import ConfigTooth as cfg


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_knn2(x, k=24, idx=None, dim9=True, concat=None, first_run=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            # we let the knn disregard the last 3 entries because those are the coordinates we keep for the
            # purpose of lsam and dists calculation and not directly for knn
            idx = knn(x[:, :-3], k=k)   # (batch_size, num_points, k)
        elif dim9 == True and first_run==False:
            #idx = knn(x[:, 6:], k=k)
            idx = knn(x[:, :], k=k)
        elif dim9 == True and first_run==True:
            #idx = knn(x[:, 6:], k=k)
            #idx = knn(x[:, :12], k=k) # for TSGCN we will use all the 4 coordinates in the first run
            idx = knn(x[:, 9:12], k=k)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    # we concatenate the normal channels also alongwith the coordinate channels
    # because we want same neighborhood to be formed for the n-stream branch as well
    if concat is not None:
        x = torch.cat((x, concat), dim=1) # concatenate them along the channel dim
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    #feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #feature = torch.cat((feature-x, x, feature), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)

def get_knn(x, k=24, idx=None, dim9=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            # we let the knn disregard the last 3 entries because those are the coordinates we keep for the
            # purpose of lsam and dists calculation and not directly for knn
            idx = knn(x[:, :-3], k=k)   # (batch_size, num_points, k)
        else:
            #idx = knn(x[:, 6:], k=k)
            idx = knn(x[:, 9:12], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    #feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    feature = torch.cat((feature-x, x, feature), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            #idx = knn(x[:, 6:], k=k)
            idx = knn(x[:, 9:12], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x


class DGCNN_semseg(nn.Module):
    #def __init__(self, args):
    def __init__(self, num_classes, num_channels, k, emb_dims, dropout):
        super(DGCNN_semseg, self).__init__()
        self.num_classes = num_classes
        self.k = k
        self.emb_dims = emb_dims
        self.num_channels = num_channels
        self.dropout = dropout
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        #self.conv1 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=1, bias=False),
        #                           self.bn1,
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv2d(self.num_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
        #                           self.bn6,
        #                           nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        #self.dp1 = nn.Dropout(p=args.dropout)
        self.dp1 = nn.Dropout(p=self.dropout)
        
        #self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)
        self.conv9 = nn.Conv1d(256, self.num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        #x = x.transpose(2, 1)
        x = x.transpose(2, 1).contiguous()
        x = torch.nn.Softmax(dim=-1)(x.view(-1,self.k))
        x = x.view(batch_size, num_points, self.num_classes)
        
        return x


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
#from torchsummary import summary

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split([3, D-3],dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) #up-sampling
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        
class GAC_encoder(nn.Module):
    def __init__(self, channel=3, channel2=64, channel3=128, channel4=256):
        super(GAC_encoder, self).__init__()
        #self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        # the gac part is somewhat similar to dgcnn and the other encoder is similar to pointnet
        #self.conv1 = torch.nn.Conv1d(3*channel, 64, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        coord_count = 3 # the number of channels for LSAM is 3 times the number of coordinates
        self.k = 24
        
        # GAC 1
        self.conv1 = torch.nn.Conv1d(3*coord_count, 16, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        #self.conv2 = torch.nn.Conv1d(2*channel+16, 64, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.conv2 = torch.nn.Conv1d(2*channel+16, channel2, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.pdist = torch.nn.PairwiseDistance(p=2)
        #self.conv3 = torch.nn.Conv1d(channel*2 + 1, 64, 1) # attention weight calculation function
        self.conv3 = torch.nn.Conv1d(channel*2 + 1, channel2, 1) # attention weight calculation function

        # GAC 2
        self.conv12 = torch.nn.Conv1d(3*coord_count, 16, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        #self.conv22 = torch.nn.Conv1d(2*channel+16, 128, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        #self.conv22 = torch.nn.Conv1d(2*64+16, 128, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.conv22 = torch.nn.Conv1d(2*channel2+16, channel3, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        #self.pdist = torch.nn.PairwiseDistance(p=2)
        #self.conv32 = torch.nn.Conv1d(64*2 + 1, 128, 1) # attention weight calculation function
        self.conv32 = torch.nn.Conv1d(channel2*2 + 1, channel3, 1) # attention weight calculation function

        # GAC 3
        self.conv13 = torch.nn.Conv1d(3*coord_count, 16, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        #self.conv23 = torch.nn.Conv1d(2*128+16, 256, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.conv23 = torch.nn.Conv1d(2*channel3+16, channel4, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        #self.pdist = torch.nn.PairwiseDistance(p=2)
        self.conv33 = torch.nn.Conv1d(channel3*2 + 1, channel4, 1) # attention weight calculation function
        #self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.conv4 = torch.nn.Conv1d(448, 256, 1)
        self.conv4 = torch.nn.Conv1d(channel2 + channel3 + channel4, 256, 1)

        #self.conv5 = torch.nn.Conv1d(64, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.bn12 = nn.BatchNorm1d(16)
        self.bn22 = nn.BatchNorm1d(128)
        self.bn32 = nn.BatchNorm1d(128)
        
        self.bn13 = nn.BatchNorm1d(16)
        self.bn23 = nn.BatchNorm1d(256)
        self.bn33 = nn.BatchNorm1d(256)
        
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        #print("################################# GAC layer 1 starts #####################################")
        y = copy.deepcopy(x)
        B, D, N = x.size()
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x = get_knn(x, k = self.k)

        #print("---------LSAM 1-----------")
        # the x now contains the (feature-x, x, feature) concatenated along the 3rd dim which is permuted to be dim 1
        lsam_ip = torch.cat((x[:, 9:12, :, :], x[:, 24+9:24+12, :, :],  x[:, 48+9:48+12, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        #print('lsam_ip size: ', lsam_ip.size())
        _, c, p, _ = lsam_ip.size()
        lsam_ip = lsam_ip.view(-1, c, p)
        #print('lsam_ip size: ', lsam_ip.size())

        #print("***********euclidean dist of neighbors 1***************")
        coord_i = x[:, 24+9:24+12, :, :] # x is the second 24 dims in the x
        coord_j = x[:, 48+9:48+12, :, :] # feature is the last 24 dims in the x
        #print('coord_i size: ', coord_i.size())
        #print('coord_j size: ', coord_j.size())
        dists = self.pdist(coord_j, coord_i)
        #print('dists size before view: ', dists.size())
        _,p,_ = dists.size()
        dists = dists.view(-1, 1, p)
        #print('dists size after view: ', dists.size())


        B, num_dims, num_points, k = x.size()
        #print('B:{}. num_dims:{}, num_points:{}, k:{}'.format(B, num_dims, num_points, k))
        r = F.leaky_relu(self.bn1(self.conv1(lsam_ip)))
        #print('r size: {}', r.size())
        z = x[:, 24:, :, :].contiguous()
        #print('z size before view: ', z.size())
        _, c, p, _ = z.size()
        z = z.view(-1, c, p)
        #print('z size after view: ', z.size())
        gac_ip = torch.cat((z, r), dim=1)
        #gac_ip = torch.cat((x[:, 24:, :, :].view(-1, 48, 12000, 24), r), dim=1)
        #print('r size: {}, gac_ip size: {}'.format(r.size(), gac_ip.size()))
        f_cap = F.leaky_relu(self.bn2(self.conv2(gac_ip)))
        #print('f_cap after view: ', f_cap.size())
        a1 = x[:, :24, :, :].contiguous().view(-1, 24, 12000)
        a2 = x[:, 48:, :, :].contiguous().view(-1, 24, 12000)
        attn_wts_ip = torch.cat((a1, a2, dists), dim=1)
        #print('attn_wts_ip size : ', attn_wts_ip.size())
        attn_wts = F.leaky_relu(self.bn3(self.conv3(attn_wts_ip)))
        #print('attn_wts size : ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k, -1, N))
        #print('attn_wts size : ', attn_wts.size())
        res = torch.mul(attn_wts, f_cap) # res should have the size BxNx64x1
        #print('res size : ', res.size())
        res = res.view(B, self.k, 64, N)
        #print('res size : ', res.size())
        res = torch.sum(res, dim=1)
        #print('res size after sum: ', res.size())
        #res_f = copy.deepcopy(res)
        



        #print("################################# GAC layer 2 starts #####################################")
        B, D, N = res.size()
        # concat the barycenter coordinates at the end of the features
        res = torch.cat((res, y[:, 9:12, :]), dim=1)
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x2 = get_knn(res, k=self.k, dim9=False)

        #print("---------LSAM 2-----------")
        # the x now contains the (feature-x, x, feature) concatenated along the 3rd dim which is permuted to be dim 1
        # we may not need to do this processing step again as this state is kind of static, every time lsam takes as
        # input the coordinates of the points
        # we will need to calculate the lsam based on the new neighbors found in the previous step i.e. knn
        lsam_ip2 = torch.cat((x2[:, 64:64+3, :, :], x2[:, 128+3:128+6, :, :],  x2[:, 192+6:192+9, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        #print('lsam_ip2 size: ', lsam_ip2.size())
        _, c, p, _ = lsam_ip2.size()
        lsam_ip2 = lsam_ip2.view(-1, c, p)
        #print('lsam_ip2 size: ', lsam_ip2.size())

        #print("***********euclidean dist of neighbors 2***************")
        # we may not need to calculate the dists again as they are already calculated in the GAC layer 1
        # we will need to calculate the dists variable based on the new neighbor
        coord_i2 = x2[:, 128+3:128+6, :, :] # x is the second 67 dims in the x
        coord_j2 = x2[:, 192+6:192+9, :, :] # feature is the last 67 dims in the x
        #print('coord_i2 size: ', coord_i2.size())
        #print('coord_j2 size: ', coord_j2.size())
        dists2 = self.pdist(coord_j2, coord_i2)
        #print('dists2 size before view: ', dists2.size())
        _,p,_ = dists2.size()
        dists2 = dists2.view(-1, 1, p)
        #print('dists2 size after view: ', dists2.size())

        x2= torch.cat((x2[:, :64, :, :], x2[:, 64+3:128+3, :, :],  x2[:, 128+6:192+6, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        B, num_dims, num_points, k = x2.size()
        #print('B:{}. num_dims:{}, num_points:{}, k:{}'.format(B, num_dims, num_points, k))
        r2 = F.leaky_relu(self.bn12(self.conv12(lsam_ip2)))
        #print('r2 size: {}', r2.size())
        #z = x[:, 24:, :, :].contiguous()
        z2 = x2[:, 64:, :, :].contiguous()
        #print('z2 size before view: ', z2.size())
        _, c, p, _ = z2.size()
        z2 = z2.view(-1, c, p)
        #print('z2 size after view: ', z2.size())
        gac_ip2 = torch.cat((z2, r2), dim=1)
        #gac_ip = torch.cat((x[:, 24:, :, :].view(-1, 48, 12000, 24), r), dim=1)
        #print('r2 size: {}, gac_ip2 size: {}'.format(r2.size(), gac_ip2.size()))
        f_cap2 = F.leaky_relu(self.bn22(self.conv22(gac_ip2)))
        #print('f_cap2 after view: ', f_cap2.size())
        a1 = x2[:, :64, :, :].contiguous().view(-1, 64, 12000)
        a2 = x2[:, 128:, :, :].contiguous().view(-1, 64, 12000)
        attn_wts_ip2 = torch.cat((a1, a2, dists2), dim=1)
        #print('attn_wts_ip2 size : ', attn_wts_ip2.size())
        attn_wts2 = F.leaky_relu(self.bn32(self.conv32(attn_wts_ip2)))
        #print('attn_wts2 size : ', attn_wts2.size())
        attn_wts2 = torch.nn.Softmax(dim=1)(attn_wts2.view(B*self.k, -1, N))
        #print('attn_wts2 size : ', attn_wts2.size())
        res2 = torch.mul(attn_wts2, f_cap2) # res should have the size BxNx64x1
        #print('res2 size : ', res2.size())
        res2 = res2.view(B, self.k, 128, N)
        #print('res2 size : ', res2.size())
        res2 = torch.sum(res2, dim=1)
        #print('res2 size after sum: ', res2.size())
        #res2_f = copy.deepcopy(res2)


        #print("################################# GAC layer 3 starts #####################################")
        B, D, N = res2.size()
        # concat the barycenter coordinates at the end of the features
        res2 = torch.cat((res2, y[:, 9:12, :]), dim=1)
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x3 = get_knn(res2, k=self.k, dim9=False)

        #print("---------LSAM 3-----------")
        # the x now contains the (feature-x, x, feature) concatenated along the 3rd dim which is permuted to be dim 1
        # we may not need to do this processing step again as this state is kind of static, every time lsam takes as
        # input the coordinates of the points
        # we will need to calculate the lsam based on the new neighbors found in the previous step i.e. knn
        lsam_ip3 = torch.cat((x3[:, 128:128+3, :, :], x3[:, 256+3:256+6, :, :],  x3[:, (128*3)+6:(128*3)+9, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        #print('lsam_ip3 size: ', lsam_ip3.size())
        _, c, p, _ = lsam_ip3.size()
        lsam_ip3 = lsam_ip3.view(-1, c, p)
        #print('lsam_ip3 size: ', lsam_ip3.size())

        #print("***********euclidean dist of neighbors 3***************")
        # we may not need to calculate the dists again as they are already calculated in the GAC layer 1
        # we will need to calculate the dists variable based on the new neighbor
        coord_i3 = x3[:, (128*2)+3:(128*2)+6, :, :] # x is the second 67 dims in the x
        coord_j3 = x3[:, (128*3)+6:(128*3)+9, :, :] # feature is the last 67 dims in the x
        #print('coord_i3 size: ', coord_i3.size())
        #print('coord_j3 size: ', coord_j3.size())
        dists3 = self.pdist(coord_j3, coord_i3)
        #print('dists3 size before view: ', dists3.size())
        _,p,_ = dists3.size()
        dists3 = dists3.view(-1, 1, p)
        #print('dists3 size after view: ', dists3.size())

        x3 = torch.cat((x3[:, :128, :, :], x3[:, 128+3:256+3, :, :],  x3[:, 256+6:(128*3)+6, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        B, num_dims, num_points, k = x3.size()
        #print('B:{}. num_dims:{}, num_points:{}, k:{}'.format(B, num_dims, num_points, k))
        r3 = F.leaky_relu(self.bn13(self.conv13(lsam_ip3)))
        #print('r3 size: {}', r3.size())
        #z = x[:, 24:, :, :].contiguous()
        z3 = x3[:, 128:, :, :].contiguous()
        #print('z3 size before view: ', z3.size())
        _, c, p, _ = z3.size()
        z3 = z3.view(-1, c, p)
        #print('z3 size after view: ', z3.size())
        gac_ip3 = torch.cat((z3, r3), dim=1)
        #gac_ip = torch.cat((x[:, 24:, :, :].view(-1, 48, 12000, 24), r), dim=1)
        #print('r3 size: {}, gac_ip3 size: {}'.format(r3.size(), gac_ip3.size()))
        f_cap3 = F.leaky_relu(self.bn23(self.conv23(gac_ip3)))
        #print('f_cap3 after view: ', f_cap3.size())
        a1 = x3[:, :128, :, :].contiguous().view(-1, 128, 12000)
        a2 = x3[:, 256:, :, :].contiguous().view(-1, 128, 12000)
        attn_wts_ip3 = torch.cat((a1, a2, dists3), dim=1)
        #print('attn_wts_ip3 size : ', attn_wts_ip3.size())
        attn_wts3 = F.leaky_relu(self.bn33(self.conv33(attn_wts_ip3)))
        #print('attn_wts3 size : ', attn_wts3.size())
        attn_wts3 = torch.nn.Softmax(dim=1)(attn_wts3.view(B*self.k, -1, N))
        #print('attn_wts3 size : ', attn_wts3.size())
        res3 = torch.mul(attn_wts3, f_cap3) # res should have the size BxNx64x1
        #print('res3 size : ', res3.size())
        res3 = res3.view(B, self.k, 256, N)
        #print('res3 size : ', res3.size())
        res3 = torch.sum(res3, dim=1)
        #print('res3 size after sum: ', res3.size())


        # final concatenation of features from the 3 GAC layers
        #print('res size: {}, res2 size: {}, res3 size: {}'.format(res.size(), res2.size(), res3.size()))
        #print('res_f size: {}, res2_f size: {}, res3 size: {}'.format(res_f.size(), res2_f.size(), res3.size()))
        # the last 3 values are excluded for res and res2 because we appened the barycenters manually at the end of them
        res_cat = torch.cat((res[:, :-3, :], res2[:, :-3, :], res3), dim=1)
        #res_cat = torch.cat((res_f, res2_f, res3), dim=1)
        res_final = self.conv4(res_cat)

        return res_final

class pointnet_encoder(nn.Module):
    def __init__(self, channel=3):
        super(pointnet_encoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        B, D, N = x.size()
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        return x.view(-1, 256, 1).repeat(1, 1, N) #up-sampling

class GAC_seg(nn.Module):
    def __init__(self, num_classes=15, num_channels=15):
        super(GAC_seg, self).__init__()
        self.k = num_classes
        self.feat = pointnet_encoder(channel=num_channels)
        self.feat2 = GAC_encoder(channel=num_channels)
        #self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        #self.conv2 = torch.nn.Conv1d(512, 256, 1)
        #self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        #self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        #x, trans, trans_feat = self.feat(x)
        x1 = self.feat(x)
        x2 = self.feat2(x)
        x = torch.cat((x1, x2), dim=1)
        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = torch.nn.Softmax(dim=-1)(x.view(-1,self.k))
        x = x.view(batchsize, n_pts, self.k)
#        return x, trans_feat
        return x

class STN3d_new(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 9)
        self.fc3 = nn.Linear(256, channel)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class bilateral_context_block(nn.Module):
    def __init__(self, in_c=32, in_c2=3, cfg=None, idx=0):
        super(bilateral_context_block, self).__init__()

        self.config = cfg
        self.d_out = torch.Tensor(cfg.d_out)
        self.k_n = cfg.k_n


        #self.in_c = 32
        #self.in_c2 = 3
        self.d_out0 = int(self.d_out[idx].item())
        self.in_c = in_c
        self.in_c2 = in_c2
        #self.d_out0 = self.d_cout
        #print('self.in_c: ', self.in_c)
        #print('self.in_c2: ', self.in_c2)
        #print('self.d_out0: ', self.d_out0)

        self.conv1 = nn.Conv2d(self.in_c, self.d_out0//2, 1)
        self.conv2 = nn.Conv2d(self.d_out0, self.in_c2, 1)
        self.conv3 = nn.Conv2d(self.in_c2*3, self.d_out0//2, 1)
        self.conv4 = nn.Conv2d(self.in_c2*3, self.d_out0//2, 1)
        self.conv5 = nn.Conv2d(int(3*self.d_out0/2), self.d_out0//2, 1)
        self.conv6 = nn.Conv2d(self.d_out0, self.d_out0, 1)
        self.conv7 = nn.Conv2d(2*self.d_out0, self.d_out0, 1)
        self.conv8 = nn.Conv2d(self.d_out0, 2*self.d_out0, 1)

        self.bn1 = nn.BatchNorm2d(self.d_out0//2)
        self.bn2 = nn.BatchNorm2d(self.in_c2)
        self.bn3 = nn.BatchNorm2d(self.d_out0//2)
        self.bn4 = nn.BatchNorm2d(self.d_out0//2)
        self.bn5 = nn.BatchNorm2d(self.d_out0//2)

        self.bn7 = nn.BatchNorm2d(self.d_out0)
        self.bn8 = nn.BatchNorm2d(2*self.d_out0)


    def forward(self, feature, input_xyz, input_neigh_idx):
        # bilateral context encoding starts here
        ## f_encoder_i, new_xyz = self.bilateral_context_block(feature, input_xyz, input_neigh_idx, d_out[i],
        #batch_size = input_xyz.size()[0]
        num_points = input_xyz.size()[1]

        #print('batch size: ', batch_size)
        #print('num_points: ', num_points)
        

        #print('1 feature size: ', feature.size())
        # input encoding
        # make this conversion for tensorflow to torch tensor conversion
        #feature = feature.permute(2,1,0).unsqueeze(0)
        feature = F.relu(self.bn1(self.conv1(feature))) # B * N * 1 * d_out/2 | torch version: B, d_out/2, 1, N
        #print('2 feature size: ', feature.size())
        #feature = feature.permute(3,1,0,2) # N, d_out/2, 1, 1 in torch
        feature = feature.permute(0,3,2,1) # N, d_out/2, 1, 1 in torch
        #print('3 feature size: ', feature.size())
        #print('input_xyz size: ', input_xyz.size())
        #print('input_neigh_idx size: ', input_neigh_idx.size())


        # bilateral augmentation
        #print('input_xyz size: ', input_xyz.size())
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        input_neigh_idx = input_neigh_idx.unsqueeze(2)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        a, b, _ = input_neigh_idx.size()
        _, _, c = input_xyz.size()
        input_neigh_idx1 = input_neigh_idx.expand((a, b, c))
        neigh_xyz = torch.gather(input_xyz, 1, input_neigh_idx1) 
        #print('neigh_xyz size: ', neigh_xyz.size())
        #print('neigh_xyz[0]: ', neigh_xyz[0])
        _, _, c, d = feature.size()
        input_neigh_idx2 = input_neigh_idx.expand((a, b, c))
        input_neigh_idx2 = input_neigh_idx2.unsqueeze(3)
        input_neigh_idx2 = input_neigh_idx2.expand((a, b, c, d))
        neigh_feat = torch.gather(feature, 1, input_neigh_idx2) 
        #print('4 neigh_feat size: ', neigh_feat.size())
        ################33
        #neigh_feat = neigh_feat.reshape(-1, neigh_feat.size()[0], neigh_feat.size()[2])
        #neigh_feat = neigh_feat.reshape(1, -1, self.k_n, neigh_feat.size()[2])
        #print('neigh_feat size: ', neigh_feat.size())
        
        #raise ValueError("Exit!")
        # bilateral augmentation

        tile_feat = feature.tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        #tile_xyz = input_xyz.tile(1, 1, self.k_n, 1) # B * N * k * 3
        #print('xyz size: ', input_xyz.size())
        tile_xyz = torch.unsqueeze(input_xyz, 2).tile(1, 1, self.k_n, 1) # B * N * k * 3

        #print('tile_xyz size: ', tile_xyz.size())
        #print('tile_feat size: ', tile_feat.size())


        # we are making every dim in pytorch convention
        tile_xyz = tile_xyz.permute(0, 3, 2, 1) # B, 3, k, N
        tile_feat = tile_feat.permute(0, 3, 2, 1) # B, d_out/2, k, N
        #print('after permute tile_xyz size: ', tile_xyz.size())
        #print('after permute tile_feat size: ', tile_feat.size())

        a, b, c = neigh_xyz.size() 
        neigh_xyz = neigh_xyz.reshape(a, c, -1,  num_points) # B, 3, k, N
        a, b, c, d = neigh_feat.size() 
        neigh_feat = neigh_feat.reshape(a, d, -1, num_points)  # B, d_out/2, k, N
        #print('after permute neigh_xyz size: ', neigh_xyz.size())
        #print('after permute neigh_feat size: ', neigh_feat.size())

        feat_info = torch.cat((neigh_feat - tile_feat, tile_feat), dim=1) # B, d_out, k, N
        #print('feat_info size : ', feat_info.size())
        neigh_xyz_offsets = F.relu(self.bn2(self.conv2(feat_info))) # B, 3, k, N
        #print('neigh_xyz_offsets size: ', neigh_xyz_offsets.size())
        #raise ValueError("Exit!")
        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets # B * N * k * 3
        #print('shifted_neigh_xyz size: ', shifted_neigh_xyz.size())


        xyz_info = torch.cat((neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz), dim=1) # B, 9, k, N
        #print('xyz_info size : ', xyz_info.size())
        neigh_feat_offsets = F.relu(self.bn3(self.conv3(xyz_info))) # B, d_out/2,  k, N 
        #print('neigh_feat_offsets size: ', neigh_feat_offsets.size())
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets # B, D_out/2, k, N
        #print('shifted_neigh_feat size: ', shifted_neigh_feat.size())

        xyz_encoding = F.relu(self.bn4(self.conv4(xyz_info))) # B, d_out/2, k, N
        #print('xyz_encoding size : ', xyz_encoding.size())
        feat_info = torch.cat((shifted_neigh_feat, feat_info), dim=1) # B, 3/2 * d_out, k, N
        #print('feat_info size : ', feat_info.size())
        feat_encoding = F.relu(self.bn5(self.conv5(feat_info))) # B, d_out/2, k, N
        #print('feat_encoding size : ', feat_encoding.size())
        #raise ValueError("Exit!")

        # Mixed Local Aggregation
        overall_info = torch.cat((xyz_encoding, feat_encoding), dim=1) # B, d_out, k, N
        #print('overall_info size : ', overall_info.size())
        k_weights = self.conv6(overall_info) # B, d_out, k, N
        #print('k_weights size : ', k_weights.size())
        k_weights= torch.nn.Softmax(dim=2)(k_weights) # B, d_out, k, N
        #print('k_weights size : ', k_weights.size())
        overall_info_weighted_sum = torch.sum(overall_info * k_weights, dim=2, keepdim=True) # B, d_out, 1, N
        #print('overall_info_weighted_sum size : ', overall_info_weighted_sum.size())
        overall_info_max = torch.max(overall_info, dim=2, keepdim=True)[0] # B, d_out, 1, N
        #overall_info_max = torch.max(overall_info, dim=2, keepdim=True) # B, d_out, 1, N
        #print('overall_info_max size : ', overall_info_max.size())
        #print('overall_info_max : ', overall_info_max)
        overall_encoding = torch.cat((overall_info_max, overall_info_weighted_sum), dim=1) # B, 2*d_out, 1, N
        #print('overall_encoding size : ', overall_encoding.size())

        overall_encoding = F.relu(self.bn7(self.conv7(overall_encoding))) # B, d_out, 1, N
        #print('overall_encoding size : ', overall_encoding.size())
        #raise ValueError("Exit!")
        output_feat = F.leaky_relu(self.bn8(self.conv8(overall_encoding))) # B, 2*d_out, 1, N
        #print('output_feat size : ', output_feat.size())

        return output_feat, shifted_neigh_xyz

 # we will design a two layer module for the begininning
class MMNet_seg(nn.Module):
    def __init__(self, num_classes=15, num_channels=15, cfg=None):
        super(MMNet_seg, self).__init__()


        self.config = cfg
        self.d_out = torch.Tensor(cfg.d_out)
        self.ratio = torch.Tensor(cfg.sub_sampling_ratio).cuda()
        self.k_n = cfg.k_n
        self.num_layers = cfg.num_layers
        self.n_pts = cfg.num_points

        # we have the output as 32 for the linear layer
        # because we have originally 24 dims and we want to
        # take them ti higher dims, whereas originally
        # in the authors code they might have 6 dims
        # which are being transformed to 8 dim
        self.fc0 = nn.Linear(24, 32)
        self.conv0 = nn.Conv2d(24, 32, 1)
        #self.bn0 = nn.BatchNorm1d(32)
        self.bn0 = nn.BatchNorm2d(32)

        self.in_c = 32
        self.in_c2 = 3
        self.d_out0 = int(self.d_out[0].item())
        #print('self.in_c: ', self.in_c)
        #print('self.in_c2: ', self.in_c2)
        #print('self.d_out0: ', self.d_out0)

        idx = 0
        self.bcb_blocks = nn.ModuleList()
        self.bcb1 = bilateral_context_block(self.in_c, self.in_c2, self.config, idx)
        self.bcb_blocks.append(self.bcb1)

        # the convs for decooder
        self.conv1s = nn.ModuleList()
        self.bn1s = nn.ModuleList()
        self.up_conv1s = nn.ModuleList()
        self.up_bn1s = nn.ModuleList()
        self.conv2s = nn.ModuleList()

        # the first few conv layers of decoder
        d_temp = int(self.d_out[self.config.num_layers-1].item())

        self.conv1 = nn.Conv2d(2*d_temp, 2*d_temp, 1)
        self.bn1 = nn.BatchNorm2d(2*d_temp)
        self.up_conv1 = None
        self.up_bn1 = None
        if self.config.num_layers > 1:
            d_temp2 = int(self.d_out[self.config.num_layers-2].item())
            self.up_conv1 = nn.ConvTranspose2d(2*(d_temp+d_temp2), 2*d_temp2, 1)
            self.up_bn1 = nn.BatchNorm2d(2*d_temp2)
        else:
            self.up_conv1 = nn.ConvTranspose2d(4*d_temp, 2*d_temp, 1)
            self.up_bn1 = nn.BatchNorm2d(2*d_temp)
        #self.conv2 = nn.Conv2d(2*d_temp, 1, 1)
        self.conv2 = nn.Conv2d(2*self.d_out0, 1, 1)
        
        # appending the first conv, deconv and bn layers
        self.conv1s.append(self.conv1)
        self.bn1s.append(self.bn1)
        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)
        self.conv2s.append(self.conv2)

        prev_d = int(self.d_out[0].item())

        for idx in range(1, self.config.num_layers):
            self.temp = prev_d * 2
            self.bcb1 = bilateral_context_block(self.temp, self.in_c2, self.config, idx)
            self.bcb_blocks.append(self.bcb1)
            prev_d = int(self.d_out[idx].item())

            # for the decoder th conv layer dims will be in reverse order
            d_temp = int(self.d_out[self.config.num_layers-1-idx].item())
            self.conv1 = nn.Conv2d(2*d_temp, 2*d_temp, 1)
            self.bn1 = nn.BatchNorm2d(2*d_temp)
            '''
            for idx2 in range(self.config.num_layers - idx):
                d_temp2 = int(self.d_out[self.config.num_layers-2-idx2].item())
                self.up_conv1 = nn.ConvTranspose2d(2*(d_temp+d_temp2), 2*d_temp2, 1)
                #self.up_conv1 = nn.ConvTranspose2d(4*d_temp, 2*d_temp, 1)
                self.up_bn1 = nn.BatchNorm2d(2*d_temp2)

                # append the convtranspose and up_bn layers
                self.up_conv1s.append(self.up_conv1)
                self.up_bn1s.append(self.up_bn1)
            '''
            #self.conv2 = nn.Conv2d(2*d_temp, 1, 1)
            self.conv2 = nn.Conv2d(2*self.d_out0, 1, 1)

            self.conv1s.append(self.conv1)
            self.bn1s.append(self.bn1)
            self.conv2s.append(self.conv2)

        self.up_conv1s = nn.ModuleList()
        self.up_bn1s = nn.ModuleList()

        self.up_conv1 = None
        self.up_bn1 = None

        '''
        # TBD: revert the groups of weights and the assign to make the code fully automatic
        d_temps = []
        d_temp1s = []
        #n = self.config.num_layers
        for i in range(self.config.num_layers):
            k = self.config.num_layers - i
            for j in range(self.config.num_layers-k):
                d_temp = int(self.d_out[j].item())
                #d_temp = int(self.d_out[n-j-1].item())
                d_temps.append(d_temp)
                #if n-j-1 >= 1:
                if j >= 1:
                    d_temp1 = int(self.d_out[j-1].item())
                    #d_temp1 = int(self.d_out[n-j-2].item())
                else:
                    d_temp1 = int(self.d_out[j].item())
                    #d_temp1 = int(self.d_out[n-j-1].item())

                d_temp1s.append(d_temp1)

                print('{}, {}'.format(2*(d_temp+d_temp1), 2*d_temp1))
                #print('{}, {}'.format(2*(d_temp+d_temp1), 2*d_temp1))
                #self.up_conv1 = nn.ConvTranspose2d(2*(d_temp+d_temp1), 2*d_temp1, 1)
                #self.up_bn1 = nn.BatchNorm2d(2*d_temp1)
                
        '''

        #raise ValueError("Exit!")
        #self.up_conv1 = nn.ConvTranspose2d(2*(d_temp+d_temp2), 2*d_temp2, 1)
        #self.up_bn1 = nn.BatchNorm2d(2*d_temp2)
        # D1
        self.up_conv1 = nn.ConvTranspose2d(256+128, 128, 1)
        self.up_bn1 = nn.BatchNorm2d(128)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)

        # D2
        self.up_conv1 = nn.ConvTranspose2d(128+32, 32, 1)
        self.up_bn1 = nn.BatchNorm2d(32)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)

        # D3
        self.up_conv1 = nn.ConvTranspose2d(32+32, 32, 1)
        self.up_bn1 = nn.BatchNorm2d(32)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)

        # D4
        self.up_conv1 = nn.ConvTranspose2d(128+32, 32, 1)
        self.up_bn1 = nn.BatchNorm2d(32)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)

        # D5
        self.up_conv1 = nn.ConvTranspose2d(32+32, 32, 1)
        self.up_bn1 = nn.BatchNorm2d(32)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)

        # D6
        self.up_conv1 = nn.ConvTranspose2d(32+32, 32, 1)
        self.up_bn1 = nn.BatchNorm2d(32)

        self.up_conv1s.append(self.up_conv1)
        self.up_bn1s.append(self.up_bn1)
        '''
        self.conv1 = nn.Conv2d(2*self.d_out0, 2*self.d_out0, 1)
        self.up_conv1 = nn.ConvTranspose2d(4*self.d_out0, 2*self.d_out0, 1)
        self.conv2 = nn.Conv2d(2*self.d_out0, 1, 1)
        '''
        #self.conv1 = nn.Conv2d(2*self.d_out0, 2*self.d_out0, 1)
        #self.up_conv1 = nn.ConvTranspose2d(4*self.d_out0, 2*self.d_out0, 1)
        #self.conv2 = nn.Conv2d(2*self.d_out0, 1, 1)
        #self.conv3 = nn.Conv2d(2*self.d_out0, 4*self.d_out0, 1)
        self.conv3 = nn.Conv2d(2*self.d_out0, 64, 1)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.conv5 = nn.Conv2d(32, self.config.num_classes, 1)

        #self.bn1 = nn.BatchNorm2d(2*self.d_out0)
        #self.up_bn1 = nn.BatchNorm2d(2*self.d_out0)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        #self.bn3 = nn.BatchNorm2d(4*self.d_out0)
        #self.bn4 = nn.BatchNorm2d(2*self.d_out0)

    def forward(self, x):
        #self.feature = inputs['features']
        # X is B x D x N
        # interestingly in baafnet code, the feature is using all the features
        # including the coordinates as well , strange!
        #feature = x.clone()
        feature = x
        og_xyz = x[:, 9:12, :] # B*3*N we treat the barycenters as the pointcloud
        batch_sz, dims, num_pts = og_xyz.size()
        # everything else is considered feature includeing the vertex coordinates for now
        #og_xyz = feature[:, :, :3]
        #feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature_orig = feature.clone() # B, D, N
        #print('1111 feature size: ', feature.size())
        #feature = feature.reshape(-1, feature.shape[1]) # B*N, D
        feature = feature.unsqueeze(2) # B*N, D
        #print('2222 feature size: ', feature.size())
        #feature = F.leaky_relu(self.bn0(self.fc0(feature))) # B*N, D -> B*N, D'
        feature = F.leaky_relu(self.bn0(self.conv0(feature))) # B*N, D -> B*N, D'
        #print('3333 feature size: ', feature.size())
        #feature = torch.unsqueeze(feature, axis=1) # B*N, 1, D'
        #print('4444 feature size: ', feature.size())
        #raise ValueError("Exit!")
        #feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        #feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        f_encoder_list = []
        #print('og_xyz size: ', og_xyz.size()) # B,3,N

        input_xyz = og_xyz.reshape(batch_sz, num_pts, dims)
        #print('input_xyz size: ', input_xyz.size()) # N*3

        input_up_samples = []
        new_xyz_list = []
        xyz_list = []
        #print('5555 feature size: ', feature.size())
        #feature = feature.permute(0, 2, 1).view(batch_sz, -1, 1, num_pts) 
        #print('6666 feature size: ', feature.size())
        

        i = 0
        for i in range(self.config.num_layers):
            #input_xyz = og_xyz.view(-1, og_xyz.size()[1])
            #input_xyz = og_xyz.reshape(-1, og_xyz.size()[1])
            
            # we will deal with the batch size using loops because anyway the batch size is not going
            # to be huge soon, TBD: we should replace this function with the batch operation
            # sometime later
            input_neigh_idx = self.batch_knn(input_xyz, input_xyz, self.k_n)
            #print('input_neigh_idx size: ', input_neigh_idx.size())

            # re arrange the neighbor indices in tensor format
            #print("##################################################################################")
            #print("i: ", i)
            n_pts = self.n_pts // self.ratio[i] # N / r
            ratio = (1./self.ratio[i]).cuda() # N / r

            # calculate batchwise fps instead of individual fps
            inputs_sub_idx = self.batch_fps(input_xyz, ratio)
            #print('inputs_sub_idx size: ', inputs_sub_idx.size())
            

            # perform gather operation
            #inputs_sub_idx = inputs_sub_idx.unsqueeze(-1)
            a, b = inputs_sub_idx.size()
            #a, b, _ = inputs_sub_idx_idx.size()
            _, _, c = input_xyz.size()
            #pool_idx = pool_idx.expand((1, 50, 32))
            #interp_idx = interp_idx.expand((a, b, c))

            sub_xyz = torch.gather(input_xyz, 1, inputs_sub_idx.unsqueeze(-1).expand((a, b, c)))
            #sub_xyz = input_xyz[inputs_sub_idx] # (N / r) * 3 (3 for the three coordinates)
            #print('sub_xyz size: ', sub_xyz.size())


            # get the sub_xyz from the original input_xyz with the help of the inputs_sub_idx
            #inputs_interp_idx = torch_cluster_knn(input_xyz, sub_xyz, 1) # 2 * (N/r)
            # for every point in the original input_xya find the nearest neighbor
            # in the  sub_xyz, this will be used while upsampling
            inputs_interp_idx = self.batch_knn(sub_xyz, input_xyz, 1) # 2 * (N/r)
            #print('inputs_interp_idx: ', inputs_interp_idx)
            #print('inputs_interp_idx size: ', inputs_interp_idx.size())
            input_up_samples.append(inputs_interp_idx)


            #print('before bcb: input_xyz size: ', input_xyz.size()) # N*3
            #print('before bcb: feature size: ', feature.size()) # N*3
            #print('before bcb: input_neigh_idx size: ', input_neigh_idx.size()) # N*3
            # bilateral context encoding starts here
            bcb = self.bcb_blocks[i]
            f_encoder_i, new_xyz = bcb(feature, input_xyz, input_neigh_idx)
            #f_encoder_i, new_xyz = self.bcb1(feature, input_xyz, input_neigh_idx)
            #print('f_encoder_i size: ', f_encoder_i.size())
            #print('inputs_sub_idx size: ', inputs_sub_idx.size())

            # get the feature vector from the indices which we initially subsampled
            # so that feature correspond to the points are chosen
            # but why do we take the max of those
            f_sampled_i = self.random_sample(f_encoder_i, inputs_sub_idx) # B, d_out*2, (N/r), 1
            #print('f_sampled_i size: ', f_sampled_i.size())
            f_sampled_i = f_sampled_i.permute(0, 1, 3, 2)
            feature = f_sampled_i
            #feature = feature.permute(0, 1, 3, 2)
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            xyz_list.append(input_xyz)
            new_xyz_list.append(new_xyz)
            input_xyz = sub_xyz
            #print('loop end: input_xyz size: ', input_xyz.size()) # N*3

            #print('len of f_encoder_list: ', len(f_encoder_list))
            #for m in range(len(f_encoder_list)):
            #    print('f_encoder_list[{}] size: {} '.format(m, f_encoder_list[m].size()))

            #print('len of input_up_samples: ', len(input_up_samples))
            #for m in range(len(input_up_samples)):
            #    print('input_up_samples[{}] size: {} '.format(m, input_up_samples[m].size()))

        ######################### Encoder ########################

        ######################### Decoder ########################
        # Adaptive fusion module

        f_multi_decoder = [] # full sized feature maps
        f_weights_decoders = [] # point-wise adaptive fusion weights

        n = 0
        d = 0 # counter for the decoder
        for n in range(self.config.num_layers):
            #print('============================================================================')
            #print("n: ", n)
            feature = f_encoder_list[-1 - n]
            #print('decoder: feature size: ', feature.size())
            
            conv1 = self.conv1s[n]
            bn1 = self.bn1s[n]
            conv2 = self.conv2s[n]

            #feature = F.leaky_relu(self.bn1(self.conv1(feature)))
            feature = F.leaky_relu(bn1(conv1(feature)))
            #print('decoder: feature size after conv: ', feature.size())

            f_decoder_list = []
            j = 0
            for j in range(self.config.num_layers - n):
                # there are multiple decoders for each n depending on j
                up_conv1 = self.up_conv1s[d]
                up_bn1 = self.up_bn1s[d]
                d = d + 1
                #print("-------------------------------------------------------------------------")
                #print("n: ", n)
                #print("j: ", j)
                f_interp_i = self.nearest_interpolation(feature, input_up_samples[- j - 1 - n])
                #print('before cat: f_interp_i size: ', f_interp_i.size())
                #print('before cat: f_encoder_list[{}] size: {}'.format(-j-2-n, f_encoder_list[-j-2-n].size()))

                ip = torch.cat([f_encoder_list[-j - 2 -n], f_interp_i], dim=1)
                #print('ip size: ', ip.size())
                #f_decoder_i = F.leaky_relu(self.up_bn1(self.up_conv1(ip)))
                f_decoder_i = F.leaky_relu(up_bn1(up_conv1(ip)))
                feature = f_decoder_i
                f_decoder_list.append(f_decoder_i)
                #print('feature size: ', feature.size())
                
            # collect full-sized feature maps which are upsampled from multiple resolu
            f_multi_decoder.append(f_decoder_list[-1])
            
            # checking the decoder sizes
            #for m in range(len(f_decoder_list)):
            #    print('f_decoder_list[{}] size: {} '.format(m, f_decoder_list[m].size()))

            # summarize the point level information
            #curr_weight = self.conv2(f_decoder_list[-1])
            curr_weight = conv2(f_decoder_list[-1])
            #print('curr weight size: ', curr_weight.size())
            f_weights_decoders.append(curr_weight)

        # regress the fusion parameters
        f_weights = torch.cat(f_weights_decoders, dim=1) # the concatenation should be along channel
        #print('f_weights size: ', f_weights.size())
        f_weights = F.softmax(f_weights, dim=1)
        #print('f_weights size: ', f_weights.size())

        # adaptively fuse them by calculating a weighted sum
        f_decoder_final = torch.zeros_like(f_multi_decoder[-1])
        for i in range(len(f_multi_decoder)):
            #f_decoder_final = f_decoder_final + torch.tile(tf.expand_dims(f_weights[:,i,:,:], dim=1), [1, 1, 1, f_multi_decoder[i].get_shape()[-1].value]) * f_multi_decoder[i]
            f_decoder_final = f_decoder_final + torch.tile(torch.unsqueeze(f_weights[:,i,:,:], dim=1), [1, f_multi_decoder[i].size()[1], 1, 1]) * f_multi_decoder[i]
            #print('i: {} f_decoder_final size: {}'.format(i, f_decoder_final.size()))


        f_layer_fc1 = F.leaky_relu(self.bn3(self.conv3(f_decoder_final)))    
        #print('f_layer_fc1 size: ', f_layer_fc1.size())
        f_layer_fc2 = F.leaky_relu(self.bn4(self.conv4(f_layer_fc1)))    
        #print('f_layer_fc2 size: ', f_layer_fc2.size())
        f_layer_drop = F.dropout(f_layer_fc2, p=0.5)
        #print('f_layer_drop size: ', f_layer_drop.size())
        f_layer_fc3 = self.conv5(f_layer_drop)    
        #print('f_layer_fc3 size: ', f_layer_fc3.size())

        f_out = torch.squeeze(f_layer_fc3, dim=2)
        #print('f_out size: ', f_out.size())

        x = F.log_softmax(f_out, dim=1)
        x = x.permute(0, 2, 1)

        #return x
        return x, new_xyz_list, xyz_list
        #raise ValueError("Exit!")
        #return f_out, new_xyz_list, xyz_list


    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        
        #feature = feature.squeeze(2)
        #print('interp_idx size ', interp_idx.size())
        #feature = feature.squeeze(3)
        feature = feature.squeeze(2)
        #print('nearest interpolation: feature size: ', feature.size())
        batch_size = feature.size()[0]
        #batch_size = interp_idx.size()[0]
        #print('nearest interpolation: batch size: ', batch_size)
        #up_num_points = interp_idx.size()[1]
        up_num_points = interp_idx.size()[1]
        #print('nearest interpolation: up_num_points: ', up_num_points)
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        #print('nearest interpolation: interp_idx size: ', interp_idx.size())

        feature = feature.permute(0, 2, 1)
        #pool_idx = pool_idx.unsqueeze(-1)
        #a, b, _ = pool_idx.size()
        interp_idx = interp_idx.unsqueeze(-1)
        a, b, _ = interp_idx.size()
        _, _, c = feature.size()
        #pool_idx = pool_idx.expand((1, 50, 32))
        interp_idx = interp_idx.expand((a, b, c))

        interpolated_features = torch.gather(feature, 1, interp_idx)
        #print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        interpolated_features = interpolated_features.unsqueeze(dim=2)
        #print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        #interpolated_features = interpolated_features.permute(0, 3, 1, 2)
        interpolated_features = interpolated_features.permute(0, 3, 2, 1)
        #print('nearest interpolation: interpolated feature size: ', interpolated_features.size())
        return interpolated_features


    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = torch.squeeze(feature, dim=2)
        #pool_idx = inputs_sub_idx
        num_neigh = pool_idx.size()[1]
        batch_size = feature.size()[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        #print('feature size: ', feature.size())
        #print('num_neigh : ', num_neigh)
        #print('batch_size: ', batch_size)
        #print('pool_idx size: ', pool_idx.size())

        feature = feature.permute(0, 2, 1)
        pool_idx = pool_idx.unsqueeze(-1)
        a, b, _ = pool_idx.size()
        _, _, c = feature.size()
        #pool_idx = pool_idx.expand((1, 50, 32))
        pool_idx = pool_idx.expand((a, b, c))
        #pool_idx = pool_idx.unsqueeze(-1).expand_as(f_encoder_i)
        #print('feature size: ', feature.size())
        #print('pool_idx size: ', pool_idx.size())
        pool_features = torch.gather(feature, 1, pool_idx) 
        #pool_features = 
        #print('pool features size: ', pool_features.size())
        pool_features = pool_features.reshape(batch_size, c, num_neigh, -1)
        #print('after reshape pool features size: ', pool_features.size())
        #pool_features = torch.max(pool_features, dim=2, keepdims=True)
        #print('after torch max pool features size: ', pool_features.size())
        #print('after torch max pool features: ', pool_features)
        # I do not know why should we take max here, commenting it out for now
        # is the purpose to reduce the number of channels?
        # the only max tht makes sense to me is along the feature dimension
        '''
        pool_features = torch.max(pool_features, dim=2, keepdims=True)[0]
        print('after torch max pool features size: ', pool_features.size())
        '''
        return pool_features
        
    #@staticmethod
    #def batch_fps(x, r):
    def batch_fps(self, x, r):
        batch_sz, _, __ = x.size()
        all_tensors = []
        for i in range(batch_sz):
            cur_tensor = x[i]
            #print('cur_tensor size: ', cur_tensor.size())
            cur_sub_idx = fps(cur_tensor, ratio = r) # N / r
            #print('cur_sub_idx size: ', cur_sub_idx.size())
            # append all neighbors for a particular sample
            all_tensors.append(cur_sub_idx)

        # re arrange the neighbor indices in tensor format
        inputs_sub_idx = torch.stack(all_tensors, dim=0)
        #print('inputs_sub_idx size: ', inputs_sub_idx.size())
        return inputs_sub_idx

    #@staticmethod
    #def batch_knn(x, y, k):
    def batch_knn(self, x, y, k):
        batch_sz, _, _ = x.size()
        all_tensors = []
        for i in range(batch_sz):
            cur_tensor = x[i]
            cur_tensor2 = y[i]
            #print('cur_tensor size: ', cur_tensor.size())
            cur_neighbor = torch_cluster_knn(cur_tensor, cur_tensor2, k)[1]
            #print('cur_neighbor size: ', cur_neighbor.size())
            # append all neighbors for a particular sample
            all_tensors.append(cur_neighbor)

        # re arrange the neighbor indices in tensor format
        input_neigh_idx = torch.stack(all_tensors, dim=0)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        return input_neigh_idx
