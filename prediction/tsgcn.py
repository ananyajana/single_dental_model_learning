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
from torch_cluster import knn as torch_cluster_knn


def knn(x, k):
    #print('knn x size: ', x.size())
    #raise ValueError("Exit!")
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_knn_idx(x, k=24):
    batch_size = x.size(0)
    #num_points = x.size(2)
    num_points = x.size(1)
    x = x.contiguous().view(batch_size, -1, num_points)
    idx = knn(x, k=k)   # (batch_size, num_points, k)
    idx = idx.view(batch_size, -1)
    return idx

def get_knn2(x, k=24, idx=None, dim9=True, concat=None, first_run=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    #print('get_knn2: x size: ', x.size())
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

class tsgcn_attention_block(nn.Module):
    def __init__(self, in_c=32, out_c=3, k=24):
        super(tsgcn_attention_block, self).__init__()

        self.channel = in_c
        self.channel2 = out_c
        self.k_n = k
        
        # all the operations for c_stream 1
        self.conv0 = torch.nn.Conv1d(2*self.channel, self.channel2, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn0 = nn.BatchNorm1d(self.channel2)

        self.conv1 = torch.nn.Conv1d(2*self.channel, self.channel2, 1) # attention weight calculation function
        self.bn1 = nn.BatchNorm1d(self.channel2)

        # all the operations for n_stream 1
        self.conv2 = torch.nn.Conv1d(2*self.channel, self.channel2, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn2 = nn.BatchNorm1d(self.channel2)


    def forward(self, xc, xn):
        B, D, N = xc.size()
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        xc = xc.permute(0, 2, 1)
        #print('xc size before knn: ', xc.size())
        input_neigh_idx = get_knn_idx(xc, k=self.k_n)
        #input_neigh_idx = self.batch_knn(xc, xc, self.k_n)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        input_neigh_idx = input_neigh_idx.unsqueeze(2)
        #print('input_neigh_idx size: ', input_neigh_idx.size())

        num_points = xc.size()[1]
        #print('num_points: ', num_points)

        # bilateral augmentation
        a, b, _ = input_neigh_idx.size()
        _, _, c = xc.size()
        input_neigh_idx1 = input_neigh_idx.expand((a, b, c))
        neigh_xyz = torch.gather(xc, 1, input_neigh_idx1) 
        #print('neigh_xyz size: ', neigh_xyz.size())

        xn = xn.permute(0, 2, 1)
        _, _, c= xn.size()
        input_neigh_idx2 = input_neigh_idx.expand((a, b, c))
        neigh_feat = torch.gather(xn, 1, input_neigh_idx2) 
        #print('4 neigh_feat size: ', neigh_feat.size())

        #tile_feat = feature.tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        tile_feat = torch.unsqueeze(xn, 2).tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        #tile_xyz = input_xyz.tile(1, 1, self.k_n, 1) # B * N * k * 3
        #print('xyz size: ', input_xyz.size())
        tile_xyz = torch.unsqueeze(xc, 2).tile(1, 1, self.k_n, 1) # B * N * k * 3
        # we are making every dim in pytorch convention
        tile_feat = tile_feat.permute(0, 3, 2, 1) # B, 3, k, N
        tile_xyz = tile_xyz.permute(0, 3, 2, 1) # B, 3, k, N
        #print('tile_xyz size: ', tile_xyz.size())
        #print('tile_feat size: ', tile_feat.size())

        a, b, c, d = tile_xyz.size()
        tile_xyz = tile_xyz.contiguous().view(a, b, -1)
        a, b, c, d = tile_feat.size()
        tile_feat = tile_feat.contiguous().view(a, b, -1)
        #print('tile_xyz size after fixing dim: ', tile_xyz.size())
        #print('tile_feat size after fixing dim: ', tile_feat.size())
        neigh_xyz = neigh_xyz.permute(0, 2, 1)
        neigh_feat = neigh_feat.permute(0, 2, 1)
        #print('neigh_xyz size after fixing dim: ', neigh_xyz.size())
        #print('neigh_feat size after fixing dim: ', neigh_feat.size())

        xc01 = torch.cat((tile_xyz, neigh_xyz), dim=1)
        xn01 = torch.cat((tile_feat, neigh_feat), dim=1)

        #print('xc01 size: ', xc01.size())
        #print('xn01 size: ', xn01.size())
        ################################## c stream #################################
        # do the c_stream on the graph obtained
        f_cap01 = F.leaky_relu(self.bn0(self.conv0(xc01))) # f_ij_l cap
        #print('f_cap01 size: ', f_cap01.size())
        f_cap01 = f_cap01.view(B*self.k_n, -1, N)
        #print('f_cap01 after view size: ', f_cap01.size())

        a1 = tile_feat
        a2 = neigh_feat
        a3 = tile_feat - neigh_feat
        #print('a1: {}, a2: {}, a3: {} '.format(a1.size(), a2.size(), a3.size()))
        attn_wts_ip = torch.cat((a1, a3), dim=1)
        #print('attn wts_ip size: ', attn_wts_ip.size())
        attn_wts = F.leaky_relu(self.bn1(self.conv1(attn_wts_ip)))
        #print('attn wts size: ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k_n, -1, N))
        #print('attn wts after softmax size: ', attn_wts.size())
        res = torch.mul(attn_wts, f_cap01) # res should have the size BxNx64x1
        #print('res size: ', res.size())
        res = res.view(B, self.k_n, self.channel2, N)
        #print('res size: ', res.size())
        xc1 = torch.sum(res, dim=1)
        #print('xc1 size: ', xc1.size())

        ################################## n stream #################################
        #a, b, c, d = xn01.size()
        #xn01 = xn01.reshape(a, b, -1)
        f_cap11 = F.leaky_relu(self.bn2(self.conv2(xn01))) # f_ij_l cap
        f_cap11= f_cap11.view(B, self.k_n, self.channel2, N)
        xn1 = torch.max(f_cap11, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('xn1 size: ', xn1.size())
        # do input transform for both c_stream and n_stream
        return xc1, xn1

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

class TSGCN_seg2(nn.Module):
    def __init__(self, num_classes=15, num_channels=15):
        super(TSGCN_seg2, self).__init__()
        self.c = num_classes
        self.k = 32 # we will use 32 neighbors for the knn
        self.channels = [12, 64, 128, 256]

        # get all the blocks in a list
        self.tsgcn_blocks = nn.ModuleList()
        '''
        for i in range(3):
            self.gac1 = gac_block(cooord_count=coord_count, in_c=self.channels[i], out_c=self.channels[i+1], idx=1):
            self.gac_blocks.append(self.gac1)
        '''
        # TBD: convert this to a loop
        self.tsgcn1 = tsgcn_attention_block(in_c=self.channels[0], out_c=self.channels[1],  k=self.k)
        self.tsgcn_blocks.append(self.tsgcn1)
        self.tsgcn1 = tsgcn_attention_block(in_c=self.channels[1], out_c=self.channels[2],  k=self.k)
        self.tsgcn_blocks.append(self.tsgcn1)
        self.tsgcn1 = tsgcn_attention_block(in_c=self.channels[2], out_c=self.channels[3],  k=self.k)
        self.tsgcn_blocks.append(self.tsgcn1)

        # from here starts feature combination from individual streams and both streams
        #self.conv9 = torch.nn.Conv1d(448, 512, 1)
        self.conv9 = torch.nn.Conv1d(self.channels[1]+self.channels[2]+self.channels[3], 512, 1)
        #self.conv10 = torch.nn.Conv1d(448, 512, 1)
        self.conv10 = torch.nn.Conv1d(self.channels[1]+self.channels[2]+self.channels[3], 512, 1)
        self.conv11 = torch.nn.Conv1d(1024, 512, 1)
        self.conv12 = torch.nn.Conv1d(512, 256, 1)
        self.conv13 = torch.nn.Conv1d(256, 128, 1)
        self.conv14 = torch.nn.Conv1d(128, self.c, 1)

        self.bn9 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(512)
        self.bn11 = nn.BatchNorm1d(512)
        self.bn12 = nn.BatchNorm1d(256)
        self.bn13 = nn.BatchNorm1d(128)

        #self.num_half = 12 # the number of half channels
        self.k = 32 # we will use 32 neighbors for the knn
        # we use two separate networks because the way
        # canonicalization happens may be different for
        # coordinates and the normals
        #self.stn1 = STN3d_1(self.channels//2)
        #self.stn2 = STN3d_2(self.channels//2)
        #self.maxpool = nn.MaxPool1d(64)

    def forward(self, x):
        B, D, N = x.size()
        xc01 = x[:, :12, :]
        xn01 = x[:, 12:, :]

        i=0
        xc1, xn1 = self.tsgcn_blocks[i](xc01, xn01)
        i += 1

        xc2, xn2 = self.tsgcn_blocks[i](xc1, xn1)
        i += 1

        xc3, xn3 = self.tsgcn_blocks[i](xc2, xn2)
        i += 1
        ################################## c stream layer 2 #################################

        # concatenate the features from all the layers to
        # enable multiscale learning in c_stream
        res_c = torch.cat((xc1, xc2, xc3), dim=1)
        #print('res_c size: ', res_c.size())
        feat_c = F.leaky_relu(self.bn9(self.conv9(res_c)))
        #print('feat_c size: ', feat_c.size())

        # concatenate the features from all the layers to
        # enable multiscale learning in n_stream
        res_n = torch.cat((xn1, xn2, xn3), dim=1)
        #print('res_n size: ', res_n.size())
        feat_n = F.leaky_relu(self.bn10(self.conv10(res_n)))
        #print('feat_n size: ', feat_n.size())

        # feature normalization, this is also for the TMI paper, not the CVPR one

        # feature fusion with attention layer, this is also for the TMI paper, not the CVPR one
        res_final = torch.cat((feat_c, feat_n), dim=1)
        #print('res_final size: ', res_final.size())
        #res_final = torch.cat((xc1, xn1), dim=1)


        # final segmentation prediction
        x = self.conv11(res_final)
        #print('x size after conv11: ', x.size())
        x = self.conv12(x)
        #print('x size after conv12 : ', x.size())
        x = self.conv13(x)
        #print('x size after conv13: ', x.size())
        x = self.conv14(x)
        #print('x size after conv14: ', x.size())
        x = x.transpose(2,1).contiguous()
        #print('x size after transpose: ', x.size())
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #x = torch.nn.Softmax(dim=-1)(x.view(-1,self.c))
        x = torch.nn.LogSoftmax(dim=-1)(x.view(-1,self.c))
        #print('x size after logsoftmax : ', x.size())
        x = x.view(B, N, self.c)
        #print('x size after view : ', x.size())
#        return x, trans_feat
        return x
    
class TSGCN_seg(nn.Module):
    def __init__(self, num_classes=15, num_channels=15):
        super(TSGCN_seg, self).__init__()
        coord_count = 12
        self.c = num_classes
        self.channels = num_channels

        # all the operations for c_stream 1
        self.conv0 = torch.nn.Conv1d(2*coord_count, 64, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn0 = nn.BatchNorm1d(64)

        self.conv1 = torch.nn.Conv1d(24, 64, 1) # attention weight calculation function
        self.bn1 = nn.BatchNorm1d(64)

        # all the operations for n_stream 1
        self.conv2 = torch.nn.Conv1d(2*coord_count, 64, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn2 = nn.BatchNorm1d(64)


        # all the operations for c_stream 2
        self.channel2 = 64
        #self.conv3 = torch.nn.Conv1d(2*self.channel2, 128, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.conv3 = torch.nn.Conv1d(2*64, 128, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = torch.nn.Conv1d(128, 128, 1) # attention weight calculation function
        self.bn4 = nn.BatchNorm1d(128)

        # all the operations for n_stream 2
        self.conv5 = torch.nn.Conv1d(2*self.channel2, 128, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn5 = nn.BatchNorm1d(128)

        # all the operations for c_stream 3
        self.channel2 = 128
        self.conv6 = torch.nn.Conv1d(2*self.channel2, 256, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn6 = nn.BatchNorm1d(256)

        self.conv7 = torch.nn.Conv1d(256, 256, 1) # attention weight calculation function
        self.bn7 = nn.BatchNorm1d(256)

        # all the operations for n_stream 3
        self.conv8 = torch.nn.Conv1d(2*self.channel2, 256, 1) # this is because we will be dealing with [xi concat xj] in the channel dim
        self.bn8 = nn.BatchNorm1d(256)



        # from here starts feature combination from individual streams and both streams
        self.conv9 = torch.nn.Conv1d(448, 512, 1)
        self.conv10 = torch.nn.Conv1d(448, 512, 1)
        self.conv11 = torch.nn.Conv1d(1024, 512, 1)
        self.conv12 = torch.nn.Conv1d(512, 256, 1)
        self.conv13 = torch.nn.Conv1d(256, 128, 1)
        self.conv14 = torch.nn.Conv1d(128, self.c, 1)

        self.bn9 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(512)
        self.bn11 = nn.BatchNorm1d(512)
        self.bn12 = nn.BatchNorm1d(256)
        self.bn13 = nn.BatchNorm1d(128)

        #self.num_half = 12 # the number of half channels
        self.k = 32 # we will use 32 neighbors for the knn
        # we use two separate networks because the way
        # canonicalization happens may be different for
        # coordinates and the normals
        #self.stn1 = STN3d_1(self.channels//2)
        #self.stn2 = STN3d_2(self.channels//2)
        #self.maxpool = nn.MaxPool1d(64)

    def forward(self, x):
        B, D, N = x.size()


        ################################## c stream layer 1 #################################
        # do input transform for both c_stream and n_stream

        # get knn from the features of the coordinate stream, this graph is shared between both
        # for the first time we do not pass the concat variable because anyway the input x
        # is together with cordi and normals both
        #print('forward x size: ', x.size())
        x = get_knn2(x, k = self.k, first_run=True)
        B, num_dims, num_points, k = x.size()
        #x = get_knn(xc0, k = self.k)
        #xc01, xn01 = x[:, :12, :], x[:, 12:, :]
        # first part of the cat comes from the feature coordinate section
        # second part of the cat comes from the x coordinate section
        # see drawing in c o l notebook
        # x is arranged like [c12, n12, c12, n12] in channel dim where 
        # c stand for the coordinate stream and n stand for the normal
        # stream
        xc01 = torch.cat((x[:, :12, :, :], x[:,24:36, :, :]), dim=1)
        xn01 = torch.cat((x[:, 12:24, :, :], x[:,36:, :, :]), dim=1)

        #print('xc01 size before view: ', xc01.size())
        _, c, p, _ = xc01.size()
        xc01 = xc01.view(-1, c, p)
        # do the c_stream on the graph obtained
        f_cap01 = F.leaky_relu(self.bn0(self.conv0(xc01))) # f_ij_l cap
        # calculate the attention weights with the help of the original features
        # a1 is the features
        # a2 is the centers i.e. x
        # the output of knn is (feature, x), num_points = 12k currently
        a1 = x[:, :12, :, :].contiguous().view(-1, 12, num_points) #f_i_l
        a2 = x[:, 12*2:12*3, :, :].contiguous().view(-1, 12, num_points)
        a3 = (a1 - a2).contiguous().view(-1, 12, num_points) # f_ij_l
        attn_wts_ip = torch.cat((a1, a3), dim=1)
        attn_wts = F.leaky_relu(self.bn1(self.conv1(attn_wts_ip)))
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k, -1, N))
        res = torch.mul(attn_wts, f_cap01) # res should have the size BxNx64x1
        res = res.view(B, self.k, 64, N)
        xc1 = torch.sum(res, dim=1)
        #print('xc1 size: ', xc1.size())


        ################################## n stream layer 1 #################################
        _, c, p, _ = xn01.size()
        xn01 = xn01.view(-1, c, p)
        f_cap11 = F.leaky_relu(self.bn2(self.conv2(xn01))) # f_ij_l cap
        f_cap11= f_cap11.view(B, self.k, 64, N)
        xn1 = torch.max(f_cap11, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('xn1 size: ', xn1.size())


        ################################## c stream layer 2 #################################
        # get knn from the features of the coordinate stream, this graph is shared between both
        # it may seem that we are getting 64*2 for xc02 and xn02 but that is because feature is
        # concatenated with x
        x = get_knn2(xc1, k = self.k, concat=xn1)
        xc02 = torch.cat((x[:, :64, :, :], x[:,64*2:64*3, :, :]), dim=1)
        xn02 = torch.cat((x[:, 64:64*2, :, :], x[:,64*3:, :, :]), dim=1)

        _, c, p, _ = xc02.size()
        xc02 = xc02.view(-1, c, p)
        #print('xc02 size: ', xc02.size())
        # do the c_stream on the graph obtained
        f_cap02 = F.leaky_relu(self.bn3(self.conv3(xc02))) # f_ij_l cap
        # calculate the attention weights with the help of the original features
        # a1 is the features
        # a2 is the centers i.e. x
        # the output of knn is (feature, x), num_points = 12k currently
        a1 = x[:, :64, :, :].contiguous().view(-1, 64, num_points) #f_i_l
        a2 = x[:, 64*2:64*3, :, :].contiguous().view(-1, 64, num_points)
        a3 = (a1 - a2).contiguous().view(-1, 64, num_points) # f_ij_l
        attn_wts_ip = torch.cat((a1, a3), dim=1)
        attn_wts = F.leaky_relu(self.bn4(self.conv4(attn_wts_ip)))
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k, -1, N))
        res = torch.mul(attn_wts, f_cap02) # res should have the size BxNx64x1
        res = res.view(B, self.k, 128, N)
        xc2 = torch.sum(res, dim=1)
        #print('xc2 size: ', xc2.size())


        ################################## n stream layer 2 #################################
        _, c, p, _ = xn02.size()
        xn02 = xn02.view(-1, c, p)
        f_cap12 = F.leaky_relu(self.bn5(self.conv5(xn02))) # f_ij_l cap
        f_cap12= f_cap12.view(B, self.k, 128, N)
        xn2 = torch.max(f_cap12, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('xn2 size: ', xn2.size())


        ################################## c stream layer 3 #################################
        # get knn from the features of the coordinate stream, this graph is shared between both
        # it may seem that we are getting 64*2 for xc02 and xn02 but that is because feature is
        # concatenated with x
        # get knn from the features of the coordinate stream, this graph is shared between both
        x = get_knn2(xc2, k = self.k, concat=xn2)
        xc03 = torch.cat((x[:, :128, :, :], x[:,128*2:128*3, :, :]), dim=1)
        xn03 = torch.cat((x[:, 128:128*2, :, :], x[:,128*3:, :, :]), dim=1)

        _, c, p, _ = xc03.size()
        xc03 = xc03.view(-1, c, p)
        # do the c_stream on the graph obtained
        f_cap03 = F.leaky_relu(self.bn6(self.conv6(xc03))) # f_ij_l cap
        # calculate the attention weights with the help of the original features
        # a1 is the features
        # a2 is the centers i.e. x
        # the output of knn is (feature, x), num_points = 12k currently
        # we could have alternatively used xc03 but then it had to be done
        # before changing the view of xc03
        a1 = x[:, :128, :, :].contiguous().view(-1, 128, num_points) #f_i_l
        a2 = x[:, 128*2:128*3, :, :].contiguous().view(-1, 128, num_points)
        a3 = (a1 - a2).contiguous().view(-1, 128, num_points) # f_ij_l
        attn_wts_ip = torch.cat((a1, a3), dim=1)
        attn_wts = F.leaky_relu(self.bn7(self.conv7(attn_wts_ip)))
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k, -1, N))
        res = torch.mul(attn_wts, f_cap03) # res should have the size BxNx64x1
        res = res.view(B, self.k, 256, N)
        xc3 = torch.sum(res, dim=1)
        #print('xc3 size: ', xc3.size())


        ################################## n stream layer 3 #################################
        #print('xn01 size before view: ', xn01.size())
        _, c, p, _ = xn03.size()
        xn03 = xn03.view(-1, c, p)
        f_cap13 = F.leaky_relu(self.bn8(self.conv8(xn03))) # f_ij_l cap
        f_cap13 = f_cap13.view(B, self.k, 256, N)
        xn3 = torch.max(f_cap13, dim=1)[0] # res should have the size BxNx64x1, the max pooling is along the channel dim
        #print('xn3 size: ', xn3.size())


        # concatenate the features from all the layers to
        # enable multiscale learning in c_stream
        res_c = torch.cat((xc1, xc2, xc3), dim=1)
        feat_c = F.leaky_relu(self.bn9(self.conv9(res_c)))

        # concatenate the features from all the layers to
        # enable multiscale learning in n_stream
        res_n = torch.cat((xn1, xn2, xn3), dim=1)
        feat_n = F.leaky_relu(self.bn10(self.conv10(res_n)))

        # feature normalization, this is also for the TMI paper, not the CVPR one


        # feature fusion with attention layer, this is also for the TMI paper, not the CVPR one
        res_final = torch.cat((feat_c, feat_n), dim=1)
        #res_final = torch.cat((xc1, xn1), dim=1)


        # final segmentation prediction
        x = self.conv11(res_final)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = x.transpose(2,1).contiguous()
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #x = torch.nn.Softmax(dim=-1)(x.view(-1,self.c))
        x = torch.nn.LogSoftmax(dim=-1)(x.view(-1,self.c))
        x = x.view(B, N, self.c)
#        return x, trans_feat
        return x
