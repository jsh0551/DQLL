import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from config import cfg
import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
import os
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights,\
MobileNet_V3_Large_Weights, mobilenet_v3_large
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# setting
hitRange = 5
epochs = 150
best_loss = 1e10
points_num = 7
model_path = 'model_trained/p7_scale_mobilenetV3'
os.makedirs(model_path, exist_ok=True)
os.makedirs(os.path.join(model_path, 'val'), exist_ok=True)

class Efficient_V2_reg(nn.Module):
    def __init__(self, points_num=5, size='medium'):
        super(Efficient_V2_reg, self).__init__()
        if size=='small':
            weights = EfficientNet_V2_S_Weights.DEFAULT
            model = efficientnet_v2_s(weights=weights)
        else:
            weights = EfficientNet_V2_M_Weights.DEFAULT
            model = efficientnet_v2_m(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.reg_layer = nn.Sequential(nn.Dropout(0.3),nn.Linear(1280,points_num))
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), -1)
        reg_value = self.reg_layer(img)
        return reg_value
    
class MobilenetV3_reg(nn.Module):
    def __init__(self, points_num=5):
        super(MobilenetV3_reg, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.reg_layer = nn.Sequential(nn.Dropout(0.3),nn.Linear(960,points_num))
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), -1)
        reg_value = self.reg_layer(img)
        return reg_value    
    
class Ghostnet_reg(nn.Module):
    def __init__(self, points_num=5):
        super(Ghostnet_reg, self).__init__()
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.reg_layer = nn.Sequential(nn.Dropout(0.3),nn.Linear(1280,points_num))
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), -1)
        reg_value = self.reg_layer(img)
        return reg_value
    
def draw_point(img, pred, gt):
    img = img.squeeze().permute(1,2,0).cpu().detach().numpy()
    img = (img*STD + MEAN) * 255
    img = np.array(img[:,:,::-1])
    pred = pred.squeeze().cpu().detach().numpy().astype(np.uint16)
    gt = gt.squeeze().cpu().detach().numpy().astype(np.uint16)
    # initY = np.array([11, 31, 51, 71, 91])
    # initY = np.array([0+11*i for i in range(10)])
    initY = np.array([5, 20, 35, 50, 65, 80, 95])
    for x,y in zip(gt, initY):
        cv2.circle(img, (x,y), radius=8, color=(255, 0, 0), thickness=2)

    for x,y in zip(pred, initY):
        cv2.line(img, (x,y), (x,y), (0, 0, 255), 4)
    return img

trainloader, valloader = datasets.getData(cfg)

# model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
model = MobilenetV3_reg(points_num=points_num)
# model = Ghostnet_reg(points_num=points_num)
# model = Efficient_V2_reg(size='small')
model.to('cuda')

reg_loss = torch.nn.MSELoss()
cls_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.TRAIN.LR)

tf = transforms.Compose(
    [
    transforms.ColorJitter(0.1,0.1,0.1,0.2),
    transforms.Normalize(mean=MEAN,std=STD)
    ]
)
norm = transforms.Normalize(mean=MEAN,std=STD)
flip = transforms.RandomVerticalFlip(p=1)


# train
for epoch in range(epochs):
    model.train()
    losses = 0
    gt_counts, counts = 0,0
    for img, gt in tqdm(trainloader):
        img = img.permute(0,3,1,2).to('cuda')
        gt = gt.to('cuda').float()/100
        cls = torch.where(gt<0, torch.zeros_like(gt), torch.ones_like(gt)).float()
        # if random.random() > 0.5:
        #     img = flip(img)
        #     gt = torch.where(gt<0, gt, 1-gt)
        img /= 255.
        img = norm(img)
        optimizer.zero_grad()
        reg_value = model(img)
        loss = reg_loss(reg_value, gt)
        loss.backward()
        losses += np.sum(loss.cpu().detach().numpy())/(len(trainloader.dataset))
        dst = abs(100*gt-100*reg_value)
        count = len(dst[dst<hitRange])
        counts += count
        gt_counts += len(gt)*points_num
        optimizer.step()
    print(f'train epoch : {epoch+1}, train loss : {losses:.6f}, hit rate : {counts/gt_counts:.6f}')
    
    model.eval()
    losses = 0
    gt_counts, counts = 0,0
    for img, gt in tqdm(valloader):
        img = img.permute(0,3,1,2).to('cuda')
        gt = gt.to('cuda')/100
        cls = torch.where(gt<0, torch.zeros_like(gt), torch.ones_like(gt)).float()
        img /= 255.
        img = norm(img)
        # optimizer.zero_grad()
        reg_value = model(img)
        loss = reg_loss(reg_value, gt)
        losses += np.sum(loss.cpu().detach().numpy())/(len(valloader))
        dst = abs(100*gt-100*reg_value)
        count = len(dst[dst<hitRange])
        counts += count
        gt_counts += len(gt)*points_num
    torch.save(model.state_dict(), os.path.join(model_path,f'{epoch+1}_{counts/gt_counts:.6f}.pt'))
    img = draw_point(img, reg_value*100, gt*100)
    cv2.imwrite(os.path.join(model_path,f'val/val_{epoch+1}.jpg'),img)
    if best_loss > losses:
        print('best loss!')
        best_loss = losses
        torch.save(model.state_dict(), os.path.join(model_path,f'best_{epoch+1}_{counts/gt_counts:.6f}.pt'))
    print(f'val loss : {losses:.6f}, hit rate : {counts/gt_counts:.6f}')