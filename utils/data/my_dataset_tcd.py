import torch
import torch.utils.data
#import torch.utils.data.Dataset as Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
# import pickle

from .generate_collages import generate_collages


# np.random.seed(1)
class tcd_embedding(torch.utils.data.Dataset):
    mean = (136.544425, 123.722431, 113.253360)
    std = (68.155618, 66.077526, 68.080128)

    def __init__(self, split='train', transform=None, transform_ref=None, checkpoint=0):
        self.split = split
        self.transform = transform
        self.transform_ref = transform_ref
        self.image_path = []
        self.dir = '/home/ros/OS_TR/datasets/tcd/images'
        self.idx_to_class, self.image_path_all = self.load_path(self.dir)
        print(self.idx_to_class)
        self.texture = np.zeros((5, 256, 256, 3))
        self.test = []
        if split == 'train':
            for i in range(5*checkpoint+5, 5*checkpoint+64):
                j = i % 64
                self.image_path.append(self.image_path_all[j])
        elif split == 'all':
            for i in range(0, 64):
                self.test.append(self.idx_to_class[i])
                self.image_path.append(self.image_path_all[i][:])
        else:
            for i in range(5*checkpoint, 5*checkpoint+5):
                self.test.append(self.idx_to_class[i])
                self.image_path.append(self.image_path_all[i])
        self.len = len(self.image_path)
        print("Split type: " + split)
        print("Total extracted classes: " + str(self.len))

    def load_path(self, path):
        image_path_all = []
        classes = []
        dirs = os.listdir(path)
        dirs.sort()
        for dir in dirs:
            classes.append(dir)
            path_new = os.path.join(path, dir)
            # dirs_new = os.listdir(path_new)
            dirs_new = glob.glob(path_new + '/*.jpg')
            image_path_all.append(dirs_new)
        # class_to_idx = {classes[i]: i for i in range(len(classes))}
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

        # if self.transform is not None:
        #     query_pic1 = self.transform(query_pic)
        #     support_pic1 = self.transform(support_pic)
        # else:
        #     query_pic1 = query_pic
        #     support_pic1 = support_pic
        # return query_pic, support_pic, query_pic1, query_target, support_pic1, index_new+1

    def __len__(self):
        if self.split == 'train':
            return 5000
        else:
            return 500