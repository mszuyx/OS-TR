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
# import onnx
# print(onnx.__version__)


import albumentations as A
# import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2


np.random.seed(233)

class mydataset_embedding(torch.utils.data.Dataset):

    def __init__(self, split='train', transform=None, transform_ref=None, checkpoint=0):
        self.split = split
        self.transform = transform
        self.transform_ref = transform_ref
        self.image_path = []
        self.dir = '/home/ros/OS_TR/dtd/images'
        self.idx_to_class, self.image_path_all = self.load_path(self.dir)
        print(self.idx_to_class)
        self.texture = np.zeros((5, 256, 256, 3))
        self.test = []
        if split == 'train':
            for i in range(5*checkpoint+5, 5*checkpoint+47):
                j = i % 47
                self.image_path.append(self.image_path_all[j])
        elif split == 'all':
            for i in range(0, 47):
                self.test.append(self.idx_to_class[i])
                self.image_path.append(self.image_path_all[i][:])
        else:
            for i in range(5*checkpoint, 5*checkpoint+5):
                self.test.append(self.idx_to_class[i])
                self.image_path.append(self.image_path_all[i])
        self.len = len(self.image_path)
        print("Total extracted classes: " + str(self.len))
        print("Total image path: " + str(self.len*len(self.image_path[0])))

    def load_path(self, path):
        image_path_all = []
        classes = []
        dirs = os.listdir(path)
        dirs.sort()
        for dir in dirs:
            classes.append(dir)
            path_new = os.path.join(path, dir)
            dirs_new = glob.glob(path_new + '/*.jpg')
            image_path_all.append(dirs_new)
        idx_to_class = {i: classes[i] for i in range(len(classes))}

        return idx_to_class, image_path_all

    def load_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        return image

    def __getitem__(self, index):
        index_new = index % self.len
        query_num, support_num = np.random.randint(0, len(self.image_path[index_new]), size=2)
        support_pic = self.load_image(self.image_path[index_new][support_num])

        self.texture[0] = self.load_image(self.image_path[index_new][query_num])
        for i in range(4):
            choice = np.delete(np.arange(self.len), index_new)
            class_other = np.random.choice(choice)
            query_num = np.random.randint(0, len(self.image_path[class_other]))
            self.texture[i+1] = self.load_image(self.image_path[class_other][query_num])

        query_pic, query_target = generate_collages(self.texture)
        query_pic = query_pic.astype(np.uint8)
        if self.transform is not None:
            transformed = self.transform(image=query_pic, mask=query_target)
            if self.transform_ref is not None:
                transformed_ref = self.transform_ref(image=support_pic)

            query_pic_tf = transformed["image"]/255.0
            query_target_tf = transformed["mask"]
            support_pic_tf = transformed_ref["image"]/255.0
        else:
            query_pic_tf = query_pic
            query_target_tf = query_target
            support_pic_tf = support_pic

        return query_pic, support_pic, query_pic_tf, query_target_tf, support_pic_tf, index_new+1

    def __len__(self):
        if self.split == 'train':
            return 5000
        else:
            return 500






model_path_ = '/home/ros/OS_TR/log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-21-17:01:25_texture.pth'
model = torch.load(model_path_)
model.eval()

transform_A = A.Compose([
    # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
    A.HorizontalFlip(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=4, p=0.5),
    A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ToTensorV2()
])

transform_A2 = A.Compose([
    # A.Normalize (mean=(0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667),p=1),
    A.HorizontalFlip(p=0.5),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=4, p=0.5),
    A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RGBShift (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ToTensorV2()
])

transform_zk = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
            # transforms.RandomVerticalFlip(p=1),
            # transforms.RandomHorizontalFlip(p=1)
            # transforms.RandomRotation([-20,20])
        ])

sampleDataset = mydataset_embedding(split='all', transform = transform_A, transform_ref = transform_A2)
sampleLoader = torch.utils.data.DataLoader(sampleDataset, batch_size=1, shuffle=False)

for i, data in enumerate(sampleLoader):
            _, _, inputs, target, patch, image_class = data[0], data[1], data[2], data[3], data[4], data[5]

            if i == 274: # 44 164 271
                
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

