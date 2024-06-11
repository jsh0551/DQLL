# coding=utf-8
# ----tuzixini----
# MACOS Python3.6.6
'''
载入 self_lane数据集
'''
import pdb
import collections
from torch.utils import data
from scipy import io as sio
from torch.utils.data import DataLoader
import os.path as osp
import json
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
import cv2

def getData(cfg,tf=None):
    if cfg.DATA.NAME =='SelfLane':
        trainset = SelfLane(cfg.DATA.TRAIN_LIST)
        valset = SelfLane(cfg.DATA.VAL_LIST)
        trainloader = DataLoader(trainset, 
                                num_workers=cfg.DATA.NUM_WORKS,
                                batch_size=cfg.DATA.TRAIN_IMGBS, 
                                shuffle=cfg.DATA.IMGSHUFFLE)
        valloader = DataLoader(valset, 
                            num_workers=cfg.DATA.NUM_WORKS,
                            batch_size=cfg.DATA.VAL_IMGBS, 
                            shuffle=cfg.DATA.IMGSHUFFLE)
        meanImg = sio.loadmat(cfg.DATA.MEAN_IMG_PATH)
        meanImg = meanImg['meanImg']
        return meanImg,trainloader, valloader
    if cfg.DATA.NAME == 'TuSimpleLane':
        trainset = TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.TRAIN_LIST,isTrain=True,cfg=cfg, tf=tf)
        valset =TuSimpleLane(cfg.DATA.ROOT,cfg.DATA.VAL_LIST,isTrain=False, cfg=cfg)
        trainloader =DataLoader(trainset,batch_size=cfg.DATA.TRAIN_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        valloader =DataLoader(valset,batch_size=cfg.DATA.VAL_IMGBS,shuffle=cfg.DATA.IMGSHUFFLE,num_workers=cfg.DATA.NUM_WORKS)
        meanImg =np.load(cfg.DATA.MEAN_IMG_PATH)
        return meanImg,trainloader,valloader

class TuSimpleLane(data.Dataset):
    def __init__(self, dataroot, ListPath, isTrain=True, cfg=None, tf=None):
        self.cfg = cfg
        self.isTrain = isTrain
        if self.isTrain:
            self.root = osp.join(dataroot, 'train')
        else:
            self.root = osp.join(dataroot, 'test')
        # DRL root
        self.root = osp.join(self.root, 'DRL', 'resize')
        with open(ListPath, 'r') as f:
            self.pathList= json.load(f)
        self.tf = tf
        # set points
        self.initY = [11, 31, 51, 71, 91]
        self.p = 0.5
    def __getitem__(self, index):
        # img
        temp = osp.join(self.root, self.pathList[index] + '.png')
        img = np.array(Image.open(temp))
        img = img.astype(np.float32)
        temp = osp.join(self.root, self.pathList[index] + '_mask.png')
        mask = np.array(Image.open(temp))
        mask = mask.astype(np.uint8)
        temp = osp.join(self.root, self.pathList[index] + '.json')
        # transform
        if (self.cfg.TRAIN.tf == True) and (self.isTrain == True):
            tmp_p = np.random.uniform(0,1,2)
            if self.p < tmp_p[0]:
                angle = np.random.uniform(-15, 15)
                img = rotate(img, angle, reshape=False, mode='reflect')
                mask = rotate(mask, angle, reshape=False, mode='reflect')

            # if self.p < tmp_p[1]:
            #     crops = np.random.randint(0,10,4)
            #     x1,y1 = crops[:2]
            #     x2,y2 = 100-crops[2:]
            #     img = img[y1:y2, x1:x2]
            #     mask = mask[y1:y2, x1:x2]
            #     mask = mask.astype(np.uint8)
            #     img = cv2.resize(img,(100,100))
            #     mask = cv2.resize(mask,(100,100))
        initX = []
        for y in self.initY:
            xx = mask[y, :]
            xx = np.where(xx == 1)
            if len(xx[0]) == 0:
                x = -1
            else:
                x = int((np.max(xx)+np.min(xx))/2)
            initX.append(x)
        gt = np.array(initX)
        with open(temp, 'r') as f:
            data = json.load(f)
        cla = np.array(data['class'])
        return cla, img, gt

    def __len__(self):
        return len(self.pathList)


class SelfLane(data.Dataset):
    def __init__(self, pathList, im_tf=None, gt_tf=None):
        self.pathList = pathList
        self.im_tf = im_tf
        self.gt_tf = gt_tf

    def __getitem__(self, index):
        temp = sio.loadmat(self.pathList[index])
        img = temp['img']
        img = np.array(Image.fromarray(img).resize((100,100)))
        img =img.astype(np.float32)
        # fea = temp['fea']
        cl = np.array(int(temp['class_name'][0]))
        gt = np.array(temp['mark'][0])
        return cl, img, gt

    def __len__(self):
        return len(self.pathList)


class bufferLoader(data.Dataset):
    def __init__(self, buffer, tf=None):
        self.buffer = buffer
        self.tf = tf

    def __getitem__(self, index):
        fea, state, Q = self.buffer[index]
        fea = np.array(fea).astype(np.float32)
        state = np.array(state).astype(np.float32)
        Q = np.array(Q).astype(np.float32)
        if self.tf is not None:
            fea = self.tf(fea)
        return fea, state, Q

    def __len__(self):
        return len(self.buffer)
