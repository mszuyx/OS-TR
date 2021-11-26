import numpy as np
import argparse

import torch
from torch import nn
import torchvision.transforms as transforms

from utils.data import datasets
from utils.model import models

import os
import glob
import cv2
import matplotlib.pyplot as plt

from utils.data.generate_collages import generate_collages
from torch.autograd import Variable

import albumentations as A
from albumentations.pytorch import ToTensorV2

model_path_ = '/home/ros/OS_TR/log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-25-16:42:11_texture.pth'
model = torch.load(model_path_)
model.eval()

transform_A = A.Compose([
    # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
    # A.HorizontalFlip(p=0.5),
    # A.Flip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=4, p=0.5),
    # A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    # A.Affine (scale=(0.8,1.2), translate_percent=0.1, rotate=(-20,20), shear=(-20,20), p=0.2),
    # A.PiecewiseAffine (scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=0.1),
    ToTensorV2()
])

transform_A2 = A.Compose([
    # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
    # A.HorizontalFlip(p=0.5),
    # A.Flip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=4, p=0.5),
    # A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    # A.Affine (scale=(0.8,1.2), translate_percent=0.1, rotate=(-20,20), shear=(-20,20), p=0.2),
    # A.PiecewiseAffine (scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=0.1),
    ToTensorV2()
])

mydataset_embedding = datasets["dtd"]
sampleDataset = mydataset_embedding(split='all', transform = transform_A, transform_ref = transform_A2)
sampleLoader = torch.utils.data.DataLoader(sampleDataset, batch_size=1, shuffle=False)

for i, data in enumerate(sampleLoader):
            _, _, inputs, target, patch, image_class = data[0], data[1], data[2], data[3], data[4], data[5]

            if i == 5: # 44 164 271
                
                # print(inputs[0,0,:,:])
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    target = target.cuda(non_blocking=True) #target = target.cuda(async=True)
                    patch = patch.cuda()

                scores = model(inputs, patch)
                # scores_mean = torch.mean(scores)
                # scores[scores >= 0.35] = 1
                # scores[scores < 0.35] = 0
                seg = scores[0, 0, :, :]#.long()
                # pred = seg.data.cpu().numpy()
                # target = target.cpu().numpy()

                break



pred = seg.data.cpu().numpy()


fig = plt.figure(0)
ax = fig.add_subplot(1, 4, 1)
imgplot = plt.imshow(inputs[0].permute(1, 2, 0).data.cpu().numpy())
ax.set_title('Query')
ax.axis('off')
ax = fig.add_subplot(1, 4, 2)
imgplot = plt.imshow(patch[0].permute(1, 2, 0).data.cpu().numpy())
ax.set_title('Reference')
ax.axis('off')
ax = fig.add_subplot(1, 4, 3)
imgplot = plt.imshow(target[0].data.cpu().numpy())
ax.set_title('Target(GT)')
ax.axis('off')
ax = fig.add_subplot(1, 4, 4)
imgplot = plt.imshow(pred)
ax.set_title('Prediction')
ax.axis('off')

plt.figure(1)
plt.imshow(inputs[0].permute(1, 2, 0).data.cpu().numpy())
plt.imshow(pred, alpha=0.5, cmap=plt.get_cmap("RdBu"))
plt.show()


ref_out = cv2.cvtColor(patch[0].permute(1, 2, 0).data.cpu().numpy()*255, cv2.COLOR_RGB2BGR)
query_out = cv2.cvtColor(inputs[0].permute(1, 2, 0).data.cpu().numpy()*255, cv2.COLOR_RGB2BGR)

cv2.imwrite('/home/ros/OS_TR/ref_1.jpg', ref_out)
cv2.imwrite('/home/ros/OS_TR/query_1.jpg', query_out)

