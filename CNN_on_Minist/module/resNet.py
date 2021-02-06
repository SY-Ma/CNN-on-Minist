# @Time    : 2021/02/14 02:46
# @Author  : SY.M
# @FileName: resNet.py


import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

class ResNet(Module):
    """
    ResNet网络，利用残差网络与全剧平均池化进行特征提取，使用卷积核为1的卷积层进行channel的变换,最后使用一层全连接将维度映射为输出类别
    """
    def __init__(self,
                 in_channels: int,
                 number_of_classes: int):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.res_block1 = ResNet_Block(d_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.res_block2 = ResNet_Block(d_channels=128)
        self.res_block3 = ResNet_Block(d_channels=128)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.layer_norm = nn.LayerNorm(128)
        self.linear = nn.Linear(128, number_of_classes)

        self.model_name = self.__class__.__name__
        print('use model:', self.model_name)

    def forward(self, x):

        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.res_block1(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.GAP(x).reshape(x.shape[0], -1)
        x = self.layer_norm(x)
        x = self.linear(x)

        return x


class ResNet_Block(Module):
    """
    残差模块，残差链接需要进行相加时维度相同，所以残差模块不会改变向量的维度
    """
    def __init__(self,
                 d_channels: int):
        super(ResNet_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(d_channels)
        self.conv2 = nn.Conv2d(in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(d_channels)
        self.conv3 = nn.Conv2d(in_channels=d_channels, out_channels=d_channels, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(d_channels)

    def forward(self, x):

        residual = x
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm1(self.conv1(x)) + residual)

        return x
