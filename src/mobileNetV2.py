import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .conv import conv_1x1_bn, conv_nxn_bn


class MV2Block(pl.LightningModule):
    def __init__(self, c_in, c_out, stride, expansion=4):
        super().__init__()

        assert stride in (1, 2)
        self.stride = stride

        h_dim = int(c_in * expansion)

        self.conv1 = conv_1x1_bn(c_in, h_dim)
        self.conv2 = conv_nxn_bn(h_dim, h_dim, 3, stride, groups=h_dim)
        self.conv3 = conv_1x1_bn(h_dim, c_out)

    def forward(self, x):
        if self.stride == 1:
            return self.conv3(self.conv2(self.conv1(x)))
        else:
            return x + self.conv3(self.conv2(self.conv1(x)))
