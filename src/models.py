import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from train_params import unet_pretrain_weight, resnet50_pretrain_weight, best_f_hpr, best_f_lpr, image_size
from train_params import dfew_image_fer_weight,uvit_256_weight

import os
os.chdir('/content/U-ViT')
os.environ['PYTHONPATH'] = '/env/python:/content/U-ViT'

import torch
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import libs.autoencoder
from libs.uvit import UViT
import einops
from torchvision.utils import save_image
from PIL import Image
from uvit.uvit import UViT



# 1. UNet parts --> UNet model
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigm(logits)


def get_unet():
    pretrained_model = UNet(n_channels=3, n_classes=3)
    # load the weights
    pretrained_model.load_state_dict(torch.load(unet_pretrain_weight), strict=False)
    return pretrained_model


def get_controller():
    # get resnet50 model from pretrained weights
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(resnet50_pretrain_weight), strict=False)
    num_ftrs = model.fc.in_features
    # change the last layer to 91 classes
    model.fc = nn.Linear(num_ftrs, 91)
    return model

def get_resnet50():
    # get resnet50 model from pretrained weights
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # change the last layer to 7 expression classes
    model.fc = nn.Linear(num_ftrs,7)
    return model

def get_r3d():
    # return r3d model
    model = r3d_18(weights = R3D_18_Weights.KINETICS400_V1)
    num_ftrs = model.fc.in_features
    # change the last layer to 7 expression classes
    model.fc = nn.Linear(num_ftrs,7)
    return model


def get_best_f_hpr():
    pretrained_model = UNet(n_channels=3, n_classes=3)
    # load the weights
    pretrained_model.load_state_dict(torch.load(best_f_hpr), strict=False)
    return pretrained_model

def get_best_f_lpr():
    pretrained_model = UNet(n_channels=3, n_classes=3)
    # load the weights
    pretrained_model.load_state_dict(torch.load(best_f_lpr), strict=False)
    return pretrained_model


def get_uvit():
    img_size = image_size // 8
    nnet = UViT(img_size=img_size, patch_size=2, in_chans=4, embed_dim=1152,depth=28,num_heads=16,num_classes=1001,conv=False)
    nnet.to('cuda')
    nnet.load_state_dict(torch.load(uvit_256_weight), strict=False, map_location='cuda')
    return nnet
    

def get_feature_controller():
    # get resnet50 model from pretrained weights
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(dfew_image_fer_weight), strict=False, map_location='cuda')
    num_ftrs = model.fc.in_features
    # change the last layer to 6 expression classes
    model.fc = nn.Linear(num_ftrs,6)
    return model