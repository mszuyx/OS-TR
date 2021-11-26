import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torchvision import models

from torchsummary import summary

#######################################################################################
################################         OS-TR         ################################
#######################################################################################
model_path_ = '/home/ros//OS_TR/log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-19-14:55:08_texture.pth'
model = torch.load(model_path_)
summary(model, [(3,256,256),(3,256,256)])


#######################################################################################
################################      mobilenet_v2     ################################
#######################################################################################
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
# model = torch.nn.Sequential(*(list(model.children())[:-1]))
# for param in model.parameters():
#     param.requires_grad = False
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# if torch.cuda.is_available():
#     model = model.cuda()
# summary(model, (3,254,254))

# If switch to mobilenet directly:
# self.resnet_layer2 = nn.Sequential(*list(model.children())[5])
# IndexError: list index out of range
# Need to take care of the decoding structure


#######################################################################################
################################        ResNet50       ################################
#######################################################################################

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/ros/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
                                                                                                       # utils/model/resnet50-19c8e357.pth
# model=torch.load('utils/model/resnet50-19c8e357.pth')   
# model = torch.nn.Sequential(*(list(model.children())[:-1]))       
# # model = torch.nn.Sequential(*list(model.children())[0:5])


# for param in model.parameters():
#     param.requires_grad = False
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)
# if torch.cuda.is_available():
#     model = model.cuda()
# summary(model, (3,256,256))


# total resnet50 param in OS-TR: 47016064÷2 = 23508032