# coding=utf-8
# ----tuzixini@gmail.com----
# WIN10 Python3.6.6
# 用途: DRL_Lane Pytorch 实现
# model.py

import torch.nn as nn
import pdb
import torch

class DRL_LANE(nn.Module):
    def __init__(self, cfg):
        super(DRL_LANE,self).__init__()
        # 使用默认的strid和padding
        self.landmark_num = cfg.LANDMARK_NUM
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU())
        
        self.compress = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU())

        qhead_list = []
        for _ in range(self.landmark_num):
            qhead_list.append(QHead(56448))
        self.qhead_list = nn.ModuleList(qhead_list)

    def forward(self, img, states):
        img = self.encoder(img)
        img = self.compress(img)
        img = img.view(img.shape[0],-1)
        qvals = []
        for i in range(self.landmark_num):
            qhead = self.qhead_list[i]
            state = states[:,i,...]
            q = qhead(img, state)
            qvals.append(q)
        qvals = torch.stack(qvals,dim=1)
        return qvals

class QHead(nn.Module):
    def __init__(self, node_num):
        super(QHead, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(node_num, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(545, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 4))

    def forward(self, img, state):
        img = self.fc1(img)
        img = torch.cat((img, state),1)
        img = self.fc2(img)
        img = self.fc3(img)
        return img

def getModel(cfg):
    net = DRL_LANE(cfg)
    return net