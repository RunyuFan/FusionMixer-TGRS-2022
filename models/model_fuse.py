import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate

import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet

import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
# from models.Checkin_net import Checkin_net
# from models.vit import ViT
# from models.transformer_block import Transformer1d
from models.MLPMixer import MMF_MLPMixer
from models.MLPMixerbase import mlp_mixer_s16, mlp_mixer_b16, mlp_mixer_s32
# from models.resnext import resnext50
from torchsummary import summary
# from models.resnet import ResNetFeature

class Mixer_base_img(nn.Module):
    def __init__(self, n_class):
        super(Mixer_base_img,self).__init__()
        self.n_class=n_class

        # self.resnet50_feature = ResNetFeature(128)
        # print('resnet50_feature parameters:', sum(p.numel() for p in self.resnet50_feature.parameters() if p.requires_grad))

        self.mixer = mlp_mixer_s16(num_classes=2, image_size=256, channels = 3)
        # self.fc = nn.Linear(64, self.n_class)

    def forward(self, img, sv):

        # out = self.resnet50_feature(img)
        # out = channel_out + out
        out_feature = self.mixer(img)
        # out_feature = self.ViT(out)
        # out = self.fc(out_feature)
        # print(out.shape)
        return out_feature, out_feature

class Mixer_base_sv(nn.Module):
    def __init__(self, n_class):
        super(Mixer_base_sv,self).__init__()
        self.n_class=n_class

        # self.resnet50_feature = ResNetFeature(128)
        # print('resnet50_feature parameters:', sum(p.numel() for p in self.resnet50_feature.parameters() if p.requires_grad))

        self.mixer = mlp_mixer_s16(num_classes=2, image_size=256, channels = 3)
        # self.fc = nn.Linear(64, self.n_class)

    def forward(self, img, sv):

        # out = self.resnet50_feature(img)
        # out = channel_out + out
        out_feature = self.mixer(sv)
        # out_feature = self.ViT(out)
        # out = self.fc(out_feature)
        # print(out.shape)
        return out_feature, out_feature

class Mixer_base_fuse(nn.Module):
    def __init__(self, n_class):
        super(Mixer_base_fuse,self).__init__()
        self.n_class=n_class

        # self.resnet50_feature = ResNetFeature(128)
        # print('resnet50_feature parameters:', sum(p.numel() for p in self.resnet50_feature.parameters() if p.requires_grad))

        self.mixer_img = mlp_mixer_s16(num_classes=2, image_size=256, channels = 3)
        self.mixer_sv = mlp_mixer_s16(num_classes=2, image_size=256, channels = 3)
        self.fc = nn.Linear(4, self.n_class)

    def forward(self, img, sv):
        img = self.mixer_img(img)
        sv = self.mixer_sv(sv)
        fuse = torch.cat([img, sv], 1)
        out = self.fc(fuse)
        return fuse, out

class MMFMixer(nn.Module):
    def __init__(self, n_class):
        super(MMFMixer,self).__init__()
        self.n_class=n_class

        resnet50 = models.resnet50(pretrained=False)
        print('resnet50 parameters:', sum(p.numel() for p in resnet50.parameters() if p.requires_grad))
        self.resnet50 = list(resnet50.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50 = nn.Sequential(*self.resnet50)
        print(self.resnet50)

        resnet50_sv = models.resnet50(pretrained=False)
        print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv.parameters() if p.requires_grad))
        self.resnet50_sv = list(resnet50_sv.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50_sv = nn.Sequential(*self.resnet50_sv)
        self.mixer = MMF_MLPMixer(num_classes=2, image_size=64, channels = 256)
        # self.mixer_img = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.mixer_sv = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.fc = nn.Linear(128, self.n_class)

    def forward(self, img, sv):
        # print(self.resnet50)
        # print(img.shape)
        img = self.resnet50(img)
        sv = self.resnet50_sv(sv)
        # print(img.shape, sv.shape)
        img = self.mixer(img, sv)
        # img = self.mixer_img(img)
        # sv = self.mixer_sv(sv)
        #
        # fuse_cat = torch.cat([img, sv], 1)
        # out = self.fc(fuse_cat)
        return img, img

class MMFMixer_UV(nn.Module):
    def __init__(self, n_class):
        super(MMFMixer_UV,self).__init__()
        self.n_class=n_class

        resnet50 = models.resnet50(pretrained=False)
        print('resnet50 parameters:', sum(p.numel() for p in resnet50.parameters() if p.requires_grad))
        self.resnet50 = list(resnet50.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50 = nn.Sequential(*self.resnet50)

        resnet50_sv0 = models.resnet50(pretrained=False)
        print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv0.parameters() if p.requires_grad))
        self.resnet50_sv0 = list(resnet50_sv0.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50_sv0 = nn.Sequential(*self.resnet50_sv0)

        resnet50_sv1 = models.resnet50(pretrained=False)
        print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv1.parameters() if p.requires_grad))
        self.resnet50_sv1 = list(resnet50_sv1.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50_sv1 = nn.Sequential(*self.resnet50_sv1)

        resnet50_sv2 = models.resnet50(pretrained=False)
        print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv2.parameters() if p.requires_grad))
        self.resnet50_sv2 = list(resnet50_sv2.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50_sv2 = nn.Sequential(*self.resnet50_sv2)

        resnet50_sv3 = models.resnet50(pretrained=False)
        print('resnet50_sv parameters:', sum(p.numel() for p in resnet50_sv3.parameters() if p.requires_grad))
        self.resnet50_sv3 = list(resnet50_sv3.children())[:-5]
        # self.resnext50.append(nn.AdaptiveAvgPool2d(1))
        self.resnet50_sv3 = nn.Sequential(*self.resnet50_sv3)

        self.conv_block = nn.Sequential(
            # FCViewer(),
            nn.Conv2d(256*4, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True)
            # nn.Conv2d(self.dim, self.dim, kernel_size=(1, 1), stride=(1, 1))
        )

        self.mixer = MMF_MLPMixer(num_classes=2, image_size=64, channels = 256)
        # self.mixer_img = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.mixer_sv = mlp_mixer_s16(num_classes=64, image_size=64, channels = 256)
        # self.fc = nn.Linear(128, self.n_class)

    def forward(self, img, sv0, sv1, sv2, sv3):
        # print(self.resnet50)
        # print(img.shape, sv0.shape, sv1.shape, sv2.shape, sv3.shape)
        img = self.resnet50(img)
        sv0 = self.resnet50_sv0(sv0)
        sv1 = self.resnet50_sv1(sv1)
        sv2 = self.resnet50_sv2(sv2)
        sv3 = self.resnet50_sv3(sv3)
        sv = self.conv_block(torch.cat([sv0, sv1, sv2, sv3], 1))
        # print(img.shape, sv.shape)
        img = self.mixer(img, sv)
        # img = self.mixer_img(img)
        # sv = self.mixer_sv(sv)
        #
        # fuse_cat = torch.cat([img, sv], 1)
        # out = self.fc(fuse_cat)
        return img, img

class ImgNet(nn.Module):
    def __init__(self, n_class):
        super(ImgNet,self).__init__()
        self.n_class=n_class
        img_model = models.resnet50(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d
        # print(img_model)
        self.img_model = list(img_model.children())[:-2]
        self.img_model.append(nn.AdaptiveAvgPool2d(1))
        self.img_model = nn.Sequential(*self.img_model)


        self.fc = nn.Linear(img_model.fc.in_features, 2)
        # self.last_fc = nn.Linear(64, self.n_class)


    def forward(self, img, sv):

        img = self.img_model(img)
        # print(img.shape)
        img = img.view(img.size(0), -1)
        img_fc = self.fc(img)
        # fuse = torch.cat([checkin, img], 1)
        # img_fc = self.last_fc(img)
        return img, img_fc

class SVNet(nn.Module):
    def __init__(self, n_class):
        super(SVNet,self).__init__()
        self.n_class=n_class
        img_model = models.resnet50(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d
        # print(img_model)
        self.img_model = list(img_model.children())[:-2]
        self.img_model.append(nn.AdaptiveAvgPool2d(1))
        self.img_model = nn.Sequential(*self.img_model)


        self.fc = nn.Linear(img_model.fc.in_features, 2)
        # self.last_fc = nn.Linear(64, self.n_class)


    def forward(self, img, sv):

        img = self.img_model(sv)
        # print(img.shape)
        img = img.view(img.size(0), -1)
        img_fc = self.fc(img)
        # fuse = torch.cat([checkin, img], 1)
        # img_fc = self.last_fc(img)
        return img, img_fc

class IMG_SVNet(nn.Module):
    def __init__(self, n_class):
        super(IMG_SVNet,self).__init__()
        self.n_class=n_class
        img_model = models.resnet50(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d
        # print(img_model)
        self.img_model = list(img_model.children())[:-2]
        self.img_model.append(nn.AdaptiveAvgPool2d(1))
        self.img_model = nn.Sequential(*self.img_model)

        sv_model = models.resnet50(pretrained=False)  # resnext50_32x4d  # resnext101_32x8d

        self.sv_model = list(sv_model.children())[:-2]
        self.sv_model.append(nn.AdaptiveAvgPool2d(1))
        self.sv_model = nn.Sequential(*self.sv_model)


        self.fc = nn.Linear(img_model.fc.in_features*2, 2)
        # self.last_fc = nn.Linear(64, self.n_class)


    def forward(self, img, sv):
        # print(img.shape)
        img = self.img_model(img)
        sv = self.sv_model(sv)
        # print(img.shape)
        img = img.view(img.size(0), -1)
        sv = img.view(sv.size(0), -1)

        fuse = torch.cat([img, sv], 1)

        out = self.fc(fuse)
        # fuse = torch.cat([checkin, img], 1)
        # img_fc = self.last_fc(img)
        return fuse, out
