import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .conv import conv_1x1_bn, conv_nxn_bn
from .mobileNetV2 import MV2Block


class MobileViT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2)

        self.mv = nn.ModuleList([])
        self.mv.append(MV2Block(16, 16, stride=1))

        self.mv.append(MV2Block(16, 24, stride=2))
        self.mv.append(MV2Block(24, 24, stride=1))
        self.mv.append(MV2Block(24, 24, stride=1))

        self.mv.append(MV2Block(24, 48, stride=1))
        self.mv.append(MV2Block(48, 48, stride=1))

        self.mv.append(MV2Block(48, 64, stride=1))
        self.mv.append(MV2Block(64, 64, stride=1))

        self.mv.append(MV2Block(64, 80, stride=1))
        self.mv.append(MV2Block(80, 80, stride=1))

        self.mv.append(MV2Block(80, 16, stride=1))

    def forward(self, x):
        return x


class MobileViTBlock(pl.LightningModule):
    def __init__(self, dim, channels, kernel_size, stride):
        super().__init__()
        self.conv1 = conv_nxn_bn(channels, channels, kernel_size)
        self.conv2 = conv_1x1_bn(channels, dim)

        self.conv3 = conv_1x1_bn(dim, channels)
        self.conv4 = conv_nxn_bn(2 * channels, channels, kernel_size)

    def forward(self, x):
        # clone input tensor for concatenation in the fusion step
        x_hat = x.clone()

        # Local representation
        x = self.conv1(x)
        x = self.conv2(x)

        # TODO: Global representation

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, x_hat), axis=1)
        x = self.conv4(x)

        return x
