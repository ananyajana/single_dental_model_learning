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
    '''
    print('idx: ', idx)
    print('type of idx: ', type(idx))
    raise ValueError("Exit!")
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
    '''
    idx = idx.view(batch_size, -1)
    return idx

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
            #idx = knn(x[:, :12], k=k)
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


class graph_attention_block(nn.Module):
    def __init__(self, coord_count=3, in_c=32, out_c=3, idx=0, k=24):
        super(graph_attention_block, self).__init__()

        self.channel = in_c
        self.channel2 = out_c
        #print('self.channel: {}, self.channel2: {}'.format(self.channel, self.channel2))
        self.layer_num = idx
        self.k_n = k
        self.pdist = torch.nn.PairwiseDistance(p=2)

        self.conv1 = torch.nn.Conv1d(3*coord_count, 16, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        #self.conv2 = torch.nn.Conv1d(2*channel+16, 64, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.conv2 = torch.nn.Conv1d(2*self.channel+16, self.channel2, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.pdist = torch.nn.PairwiseDistance(p=2)
        #self.conv3 = torch.nn.Conv1d(channel*2 + 1, 64, 1) # attention weight calculation function
        self.conv3 = torch.nn.Conv1d(self.channel*2 + 1, self.channel2, 1) # attention weight calculation function

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(self.channel2)
        self.bn3 = nn.BatchNorm1d(self.channel2)
        
    def forward(self, feature, input_xyz, input_neigh_idx):
        #print('feature size: ', feature.size())
        #print('input_xyz size: ', input_xyz.size())
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        input_neigh_idx = input_neigh_idx.unsqueeze(2)

        num_points = input_xyz.size()[1]

        # bilateral augmentation
        #input_neigh_idx = input_neigh_idx.unsqueeze(2)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        a, b, _ = input_neigh_idx.size()
        _, _, c = input_xyz.size()
        input_neigh_idx1 = input_neigh_idx.expand((a, b, c))
        neigh_xyz = torch.gather(input_xyz, 1, input_neigh_idx1) 
        #print('neigh_xyz size: ', neigh_xyz.size())
        #print('neigh_xyz[0]: ', neigh_xyz[0])
        N = num_points
        B = a

        feature = feature.permute(0, 2, 1)
        _, _, c= feature.size()
        input_neigh_idx2 = input_neigh_idx.expand((a, b, c))
        #input_neigh_idx2 = input_neigh_idx2.unsqueeze(3)
        neigh_feat = torch.gather(feature, 1, input_neigh_idx2) 
        #print('4 neigh_feat size: ', neigh_feat.size())

        #tile_feat = feature.tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        tile_feat = torch.unsqueeze(feature, 2).tile(1, 1, self.k_n, 1) # B * N * k * d_out/2
        #tile_xyz = input_xyz.tile(1, 1, self.k_n, 1) # B * N * k * 3
        #print('xyz size: ', input_xyz.size())
        tile_xyz = torch.unsqueeze(input_xyz, 2).tile(1, 1, self.k_n, 1) # B * N * k * 3
        # we are making every dim in pytorch convention
        tile_feat = tile_feat.permute(0, 3, 2, 1) # B, 3, k, N
        tile_xyz = tile_xyz.permute(0, 3, 2, 1) # B, 3, k, N

        #print('tile_xyz size: ', tile_xyz.size())
        #print('tile_feat size: ', tile_feat.size())

        a, b, c = neigh_xyz.size()
        neigh_xyz = neigh_xyz.view(a, c, -1, num_points)
        #print('neigh_xyz size: ', neigh_xyz.size())

        #xyz_info = torch.cat((tile_xyz, neigh_xyz, neigh_xyz - tile_xyz), dim=1) # B, 9, k, N
        lsam_ip= torch.cat((tile_xyz, neigh_xyz, neigh_xyz - tile_xyz), dim=1) # B, 9, k, N
        #print('xyz_info size: ', xyz_info.size())
        #print('lsam_ip size: ', lsam_ip.size())
        a, c, p, _ = lsam_ip.size()
        lsam_ip = lsam_ip.view(a, c, -1)
        #lsam_ip = lsam_ip.view(-1, c, p)
        #print('lsam_ip size: ', lsam_ip.size())

        #print("***********euclidean dist of neighbors 1***************")
        #coord_i = x[:, 24+9:24+12, :, :] # x is the second 24 dims in the x
        #coord_j = x[:, 48+9:48+12, :, :] # feature is the last 24 dims in the x

        coord_i = tile_xyz
        coord_j = neigh_xyz
        #print('coord_i size: ', coord_i.size())
        #print('coord_j size: ', coord_j.size())
        dists = self.pdist(coord_j, coord_i)
        #print('dists size before view: ', dists.size())
        #_,p,_ = dists.size()
        #dists = dists.view(-1, 1, p)
        dists = dists.view(a, 1, -1)
        #print('dists size after view: ', dists.size())

        #B, num_dims, num_points, k = x.size()
        #print('B:{}. num_dims:{}, num_points:{}, k:{}'.format(B, num_dims, num_points, k))
        r = F.leaky_relu(self.bn1(self.conv1(lsam_ip)))
        #print('r size: {}', r.size())

        neigh_feat = torch.gather(feature, 1, input_neigh_idx2) 
        a, b, c = neigh_feat.size()
        neigh_feat = neigh_feat.view(a, c, b)
        #print('5 neigh_feat size: ', neigh_feat.size())
        a, b, c, d = tile_feat.size()
        tile_feat = tile_feat.reshape(a, b, -1)
        #print('tile_feat size: ', tile_feat.size())
        gac_ip = torch.cat((r, tile_feat, neigh_feat), dim=1)
        #print('r size: {}, gac_ip size: {}'.format(r.size(), gac_ip.size()))
        f_cap = F.leaky_relu(self.bn2(self.conv2(gac_ip)))
        #print('f_cap before view: ', f_cap.size())
        f_cap = f_cap.view(B*self.k_n, -1, N)
        #f_cap = f_cap.view(B, -1, N*self.k_n)
        #f_cap = f_cap.view(B*N, -1, self.k_n)
        #print('f_cap after view: ', f_cap.size())
        a1 = neigh_feat - tile_feat
        a2 = neigh_feat
        attn_wts_ip = torch.cat((a1, a2, dists), dim=1)
        #print('attn_wts_ip size : ', attn_wts_ip.size())
        attn_wts = F.leaky_relu(self.bn3(self.conv3(attn_wts_ip)))
        #print('attn_wts size : ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k_n, -1, N))
        #attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B, -1, N*self.k_n))
        #attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*N, -1, self.k_n))
        #print('attn_wts size : ', attn_wts.size())
        res = torch.mul(attn_wts, f_cap) # res should have the size BxNx64x1
        #print('res size : ', res.size())
        res = res.view(B, self.k_n, self.channel2, N)
        #print('res size : ', res.size())
        res = torch.sum(res, dim=1)
        #print('res size after sum: ', res.size())
        return res


class gac_encoder2(nn.Module):
    def __init__(self, channel=3, channel2=64, channel3=128, channel4=256):
        super(gac_encoder2, self).__init__()
        coord_count = 3 # the number of channels for LSAM is 3 times the number of coordinates
        self.k = 24 # num neighbors for knn
        #self.k = 4 # num neighbors for knn

        self.channels = [channel, channel2, channel3, channel4]

        # get all the blocks in a list
        self.gac_blocks = nn.ModuleList()
        '''
        for i in range(3):
            self.gac1 = gac_block(cooord_count=coord_count, in_c=self.channels[i], out_c=self.channels[i+1], idx=1):
            self.gac_blocks.append(self.gac1)
        '''
        # TBD: convert this to a loop
        self.gac1 = graph_attention_block(coord_count=coord_count, in_c=channel, out_c=channel2, idx=0, k=self.k)
        self.gac_blocks.append(self.gac1)
        self.gac1 = graph_attention_block(coord_count=coord_count, in_c=channel2, out_c=channel3, idx=1, k=self.k)
        self.gac_blocks.append(self.gac1)
        self.gac1 = graph_attention_block(coord_count=coord_count, in_c=channel3, out_c=channel4, idx=2, k=self.k)
        self.gac_blocks.append(self.gac1)

        self.conv4 = torch.nn.Conv1d(channel2 + channel3 + channel4, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        #print("################################# GAC layer 1 starts #####################################")
        y = copy.deepcopy(x)
        B, D, N = x.size()
        feature = x
        og_xyz = x[:, 9:12, :] # B*3*N we treat the barycenters as the pointcloud
        batch_sz, dims, num_pts = og_xyz.size()
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        # get the neighbor indices
        x = x.permute(0, 2, 1)
        #print('x size before knn: ', x.size())
        input_neigh_idx = get_knn_idx(x, k = self.k)
        #print('x size after knn: ', x.size())
        #raise ValueError("Exit!")
        #input_neigh_idx = self.batch_knn(x, x, self.k)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        #print('x size after knn: ', x.size())
        input_xyz = og_xyz.reshape(batch_sz, num_pts, dims)
        
        i = 0
        res = self.gac_blocks[i](feature, input_xyz, input_neigh_idx)
        i += 1

        feature = res
        res = res.permute(0, 2, 1)
        #print('res size before knn: ', res.size())
        input_neigh_idx = get_knn_idx(res, k = self.k)
        #input_neigh_idx = self.batch_knn(res, res, self.k)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        #print('x size after knn: ', x.size())
        res2 = self.gac_blocks[i](feature, input_xyz, input_neigh_idx)
        i += 1
        #print("################################# GAC layer 2 starts #####################################")

        feature = res2
        res2 = res2.permute(0, 2, 1)
        #print('res size before knn: ', res.size())
        input_neigh_idx = get_knn_idx(res2, k = self.k)
        #input_neigh_idx = self.batch_knn(res2, res2, self.k)
        #print('input_neigh_idx size: ', input_neigh_idx.size())
        #print('x size after knn: ', x.size())
        res3 = self.gac_blocks[i](feature, input_xyz, input_neigh_idx)
        #print("################################# GAC layer 3 starts #####################################")

        res = res.permute(0, 2, 1)
        res2 = res2.permute(0, 2, 1)

        # final concatenation of features from the 3 GAC layers
        #print('res size: {}, res2 size: {}, res3 size: {}'.format(res.size(), res2.size(), res3.size()))
        #print('res_f size: {}, res2_f size: {}, res3 size: {}'.format(res_f.size(), res2_f.size(), res3.size()))
        # the last 3 values are excluded for res and res2 because we appened the barycenters manually at the end of them
        res_cat = torch.cat((res, res2, res3), dim=1)
        #print('res_cat size: ', res_cat.size())
        #res_cat = torch.cat((res_f, res2_f, res3), dim=1)
        res_final = F.leaky_relu(self.bn4(self.conv4(res_cat)))
        #print('res_final size: ', res_final.size())
        #raise ValueError("Exit!")

        return res_final

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

class GAC_seg3(nn.Module):
    def __init__(self, num_classes=15, num_channels=15):
        super(GAC_seg3, self).__init__()
        self.k = num_classes
        self.feat = pointnet_encoder(channel=num_channels)
        self.feat2 = gac_encoder2(channel=num_channels)
        #self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        #self.conv2 = torch.nn.Conv1d(512, 256, 1)
        #self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.conv1 = torch.nn.Conv1d(512, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
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
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #x = torch.nn.Softmax(dim=-1)(x.view(-1,self.k))
        x = torch.nn.LogSoftmax(dim=-1)(x.view(-1,self.k))
        x = x.view(batchsize, n_pts, self.k)
#        return x, trans_feat
        return x


class gac_block(nn.Module):
    def __init__(self, coord_count=3, in_c=32, out_c=3, idx=0, k=24):
        super(gac_block, self).__init__()

        self.channel = in_c
        self.channel2 = out_c
        self.layer_num = idx
        self.k = k

        self.conv1 = torch.nn.Conv1d(3*coord_count, 16, 1) # this is because we will be dealing with [xi concat xj concat (xj - xi)] in the channel dim
        #self.conv2 = torch.nn.Conv1d(2*channel+16, 64, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.conv2 = torch.nn.Conv1d(2*self.channel+16, self.channel2, 1) # gac1 conv conv1d(64, 64, 1), channel = 24
        self.pdist = torch.nn.PairwiseDistance(p=2)
        #self.conv3 = torch.nn.Conv1d(channel*2 + 1, 64, 1) # attention weight calculation function
        self.conv3 = torch.nn.Conv1d(self.channel*2 + 1, self.channel2, 1) # attention weight calculation function

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(self.channel2)
        self.bn3 = nn.BatchNorm1d(self.channel2)
        
    def forward(self, x):
        _, _, N, _ = x.size()
        #print('layer_num: ', self.layer_num)
        # for the first layer the coordinate is part of the channels
        # but from second layer onwards, the three coordinates are appended to the end of the channels
        idxs = []
        l = 0
        u = 0
        for i in range(3):
            if self.layer_num == 0:
                l = i*self.channel + 9
                u = l + 3
            else:
                l = (i+1)*self.channel + i*3
                u = l + 3

            idxs.append(l)
            idxs.append(u)

        #print('idxs: ', idxs)

        yy1 = x[:, idxs[0]:idxs[1], :, :]
        yy2 = x[:, idxs[2]:idxs[3], :, :]
        yy3 = x[:, idxs[4]:idxs[5], :, :]
        #print('yy1 size: ', yy1.size())
        #print('yy2 size: ', yy2.size())
        #print('yy3 size: ', yy3.size())
        #print("---------LSAM 1-----------")
        # the x now contains the (feature-x, x, feature) concatenated along the 3rd dim which is permuted to be dim 1
        lsam_ip = torch.cat((yy1, yy2, yy3), dim=1) # because LSAM only takes as input the 3D coordinates
        #lsam_ip = torch.cat((x[:, 9:12, :, :], x[:, 24+9:24+12, :, :],  x[:, 48+9:48+12, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        #lsam_ip = torch.cat((x[:, idxs[0]:idxs[1], :, :], x[:, idxs[2]:idxs[3], :, :],  x[:, idxs[3]:idxs[5], :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        #print('lsam_ip size: ', lsam_ip.size())
        _, c, p, _ = lsam_ip.size()
        lsam_ip = lsam_ip.view(-1, c, p)
        #print('lsam_ip size: ', lsam_ip.size())

        #print("***********euclidean dist of neighbors 1***************")
        #coord_i = x[:, 24+9:24+12, :, :] # x is the second 24 dims in the x
        #coord_j = x[:, 48+9:48+12, :, :] # feature is the last 24 dims in the x
        coord_i = x[:, idxs[2]:idxs[3], :, :] # x is the second 24 dims in the x
        coord_j = x[:, idxs[4]:idxs[5], :, :] # feature is the last 24 dims in the x
        #print('coord_i size: ', coord_i.size())
        #print('coord_j size: ', coord_j.size())
        dists = self.pdist(coord_j, coord_i)
        #print('dists size before view: ', dists.size())
        _,p,_ = dists.size()
        dists = dists.view(-1, 1, p)
        #print('dists size after view: ', dists.size())


        # for the layer = 0 we do not need to append the coordinates as they are already present
        # but for the layer 1 onwards we have appended the coordinates as the end and sent it to
        # knn previously, so we need to take care of that here
        if self.layer_num >= 1:
            mm1 = x[:, :self.channel, :, :]
            mm2 = x[:, self.channel+3:(self.channel*2)+3, :, :]
            mm3 = x[:, (self.channel*2)+6:(self.channel*3)+6, :, :]
            x = torch.cat((mm1, mm2, mm3), dim=1)
            #print('x size after getting rid of cordinates: ', x.size())
            #x2= torch.cat((x2[:, :64, :, :], x2[:, 64+3:128+3, :, :],  x2[:, 128+6:192+6, :, :]), dim=1) # because LSAM only takes as input the 3D coordinates
        B, num_dims, num_points, k = x.size()
        #print('B:{}. num_dims:{}, num_points:{}, k:{}'.format(B, num_dims, num_points, k))
        r = F.leaky_relu(self.bn1(self.conv1(lsam_ip)))
        #print('r size: {}', r.size())
        z = x[:, self.channel:, :, :].contiguous()
        #print('z size before view: ', z.size())
        _, c, p, _ = z.size()
        z = z.view(-1, c, p)
        #print('z size after view: ', z.size())
        gac_ip = torch.cat((z, r), dim=1)
        #gac_ip = torch.cat((x[:, 24:, :, :].view(-1, 48, 12000, 24), r), dim=1)
        #print('r size: {}, gac_ip size: {}'.format(r.size(), gac_ip.size()))
        f_cap = F.leaky_relu(self.bn2(self.conv2(gac_ip)))
        #print('f_cap after view: ', f_cap.size())
        a1 = x[:, :self.channel, :, :].contiguous().view(-1, self.channel, num_points)
        a2 = x[:, self.channel*2:, :, :].contiguous().view(-1, self.channel, num_points)
        attn_wts_ip = torch.cat((a1, a2, dists), dim=1)
        #print('attn_wts_ip size : ', attn_wts_ip.size())
        attn_wts = F.leaky_relu(self.bn3(self.conv3(attn_wts_ip)))
        #print('attn_wts size : ', attn_wts.size())
        attn_wts = torch.nn.Softmax(dim=1)(attn_wts.view(B*self.k, -1, N))
        #print('attn_wts size : ', attn_wts.size())
        res = torch.mul(attn_wts, f_cap) # res should have the size BxNx64x1
        #print('res size : ', res.size())
        res = res.view(B, self.k, self.channel2, N)
        #print('res size : ', res.size())
        res = torch.sum(res, dim=1)
        #print('res size after sum: ', res.size())
        #res_f = copy.deepcopy(res)

        return res
        
class gac_encoder(nn.Module):
    def __init__(self, channel=3, channel2=64, channel3=128, channel4=256):
        super(gac_encoder, self).__init__()
        coord_count = 3 # the number of channels for LSAM is 3 times the number of coordinates
        self.k = 24

        self.channels = [channel, channel2, channel3, channel4]

        # get all the blocks in a list
        self.gac_blocks = nn.ModuleList()
        '''
        for i in range(3):
            self.gac1 = gac_block(cooord_count=coord_count, in_c=self.channels[i], out_c=self.channels[i+1], idx=1):
            self.gac_blocks.append(self.gac1)
        '''
        # TBD: convert this to a loop
        self.gac1 = gac_block(coord_count=coord_count, in_c=channel, out_c=channel2, idx=0, k=self.k)
        self.gac_blocks.append(self.gac1)
        self.gac1 = gac_block(coord_count=coord_count, in_c=channel2, out_c=channel3, idx=1, k=self.k)
        self.gac_blocks.append(self.gac1)
        self.gac1 = gac_block(coord_count=coord_count, in_c=channel3, out_c=channel4, idx=2, k=self.k)
        self.gac_blocks.append(self.gac1)

        self.conv4 = torch.nn.Conv1d(channel2 + channel3 + channel4, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)

    def forward(self, x):
        #print("################################# GAC layer 1 starts #####################################")
        y = copy.deepcopy(x)
        B, D, N = x.size()
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x = get_knn(x, k = self.k)
        #print('x size: ', x.size())
        
        i = 0
        res = self.gac_blocks[i](x)
        i += 1

        #print("################################# GAC layer 2 starts #####################################")
        B, D, N = res.size()
        # concat the barycenter coordinates at the end of the features
        res = torch.cat((res, y[:, 9:12, :]), dim=1)
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x2 = get_knn(res, k=self.k, dim9=False)
        res2 = self.gac_blocks[i](x2)
        i += 1 

        #print("################################# GAC layer 3 starts #####################################")
        B, D, N = res2.size()
        # concat the barycenter coordinates at the end of the features
        res2 = torch.cat((res2, y[:, 9:12, :]), dim=1)
        #print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x3 = get_knn(res2, k=self.k, dim9=False)
        res3 = self.gac_blocks[i](x3)

        # final concatenation of features from the 3 GAC layers
        #print('res size: {}, res2 size: {}, res3 size: {}'.format(res.size(), res2.size(), res3.size()))
        #print('res_f size: {}, res2_f size: {}, res3 size: {}'.format(res_f.size(), res2_f.size(), res3.size()))
        # the last 3 values are excluded for res and res2 because we appened the barycenters manually at the end of them
        res_cat = torch.cat((res[:, :-3, :], res2[:, :-3, :], res3), dim=1)
        #res_cat = torch.cat((res_f, res2_f, res3), dim=1)
        res_final = F.leaky_relu(self.bn4(self.conv4(res_cat)))

        return res_final

class GAC_seg2(nn.Module):
    def __init__(self, num_classes=15, num_channels=15):
        super(GAC_seg2, self).__init__()
        self.k = num_classes
        self.feat = pointnet_encoder(channel=num_channels)
        self.feat2 = gac_encoder(channel=num_channels)
        #self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        #self.conv2 = torch.nn.Conv1d(512, 256, 1)
        #self.conv2 = torch.nn.Conv1d(256, 256, 1)
        self.conv1 = torch.nn.Conv1d(512, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
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
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #x = torch.nn.Softmax(dim=-1)(x.view(-1,self.k))
        x = torch.nn.LogSoftmax(dim=-1)(x.view(-1,self.k))
        x = x.view(batchsize, n_pts, self.k)
#        return x, trans_feat
        return x

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
        print('B:{}, D:{}, N:{}'.format(B, D, N))
        #x = get_knn(x)
        #B, _, num_points, k = x.size()
        x = get_knn(x, k = self.k)
        print('x size after knn: ', x.size())
        #raise ValueError("Exit!")

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
        a1 = x[:, :24, :, :].contiguous().view(-1, 24, num_points)
        a2 = x[:, 48:, :, :].contiguous().view(-1, 24, num_points)
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
        a1 = x2[:, :64, :, :].contiguous().view(-1, 64, num_points)
        a2 = x2[:, 128:, :, :].contiguous().view(-1, 64, num_points)
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
        a1 = x3[:, :128, :, :].contiguous().view(-1, 128, num_points)
        a2 = x3[:, 256:, :, :].contiguous().view(-1, 128, num_points)
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
        res_final = F.leaky_relu(self.bn4(self.conv4(res_cat)))

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
        x = F.leaky_relu(self.bn3(self.conv3(x)))
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
        self.conv1 = torch.nn.Conv1d(512, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
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
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        #x = F.log_softmax(x.view(-1,self.k), dim=-1)
        #x = torch.nn.Softmax(dim=-1)(x.view(-1,self.k))
        x = torch.nn.LogSoftmax(dim=-1)(x.view(-1,self.k))
        x = x.view(batchsize, n_pts, self.k)
#        return x, trans_feat
        return x
