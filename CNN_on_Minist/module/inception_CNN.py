# @Time    : 2021/01/19 30:19
# @Author  : SY.M
# @FileName: inception_CNN.py

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class Inception_CNN(Module):
    """
    Inception网络
    """

    def __init__(self,
                 in_channels: int,
                 number_of_classes: int):
        super(Inception_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=12, kernel_size=3)
        self.inception1 = Inception_block(in_channels=12)

        self.conv2 = nn.Conv2d(in_channels=88, out_channels=24, kernel_size=3)
        self.inception2 = Inception_block(in_channels=24)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.layer_norm1 = nn.LayerNorm(2200)
        self.linear1 = nn.Linear(2200, 512)
        self.layer_norm2 = nn.LayerNorm(512)
        self.linear2 = nn.Linear(512, number_of_classes)

        self.model_name = self.__class__.__name__
        print('use model:', self.model_name)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = self.inception1(x)
        x = F.relu(self.max_pool(self.conv2(x)))
        x = self.inception2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer_norm1(x)
        x = F.relu(self.linear1(x))
        x = self.layer_norm2(x)
        x = self.linear2(x)

        return x


class Inception_block(Module):
    """
    Inception模块  输出各分支在channel维度进行拼接  输出维度为88
    """

    def __init__(self,
                 in_channels: int):
        super(Inception_block, self).__init__()

        self.inc1_avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.inc1_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

        self.inc2_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)

        self.inc3_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.inc3_conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.inc4_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.inc4_conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.inc4_conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc1_avg_pool(x)
        x1 = self.inc1_conv1(x1)

        x2 = self.inc2_conv1(x)

        x3 = self.inc3_conv1(x)
        x3 = self.inc3_conv2(x3)

        x4 = self.inc4_conv1(x)
        x4 = self.inc4_conv2(x4)
        x4 = self.inc4_conv3(x4)

        inception_out = torch.cat([x1, x2, x3, x4], dim=1)

        return inception_out
