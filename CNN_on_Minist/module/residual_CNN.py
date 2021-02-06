# @Time    : 2021/01/19 30:51
# @Author  : SY.M
# @FileName: residual_CNN.py

from torch.nn import Module
import torch.nn.functional as F
import torch.nn as nn


class Residual_CNN(Module):
    """
    残差网络，最后使用全连接进行到分类类别的映射
    """
    def __init__(self,
                 in_channels: int,
                 number_of_classes: int):
        super(Residual_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)

        self.res_block1 = Residual_block(12)
        self.res_block2 = Residual_block(24)

        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.layer_norm = nn.LayerNorm(384)
        self.linear = nn.Linear(384, number_of_classes)

        self.model_name = self.__class__.__name__
        print('use model:', self.model_name)

    def forward(self, x):

        x = F.relu(self.max_pool(self.conv1(x)))
        x = self.res_block1(x)
        x = F.relu(self.max_pool(self.conv2(x)))
        x = self.res_block2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.layer_norm(x)
        x = self.linear(x)

        return x


class Residual_block(Module):
    """
    残差网络，残差block不改张量的维度，因为残差相加需要保证维度一致
    """
    def __init__(self,
                 in_channels: int):
        super(Residual_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self, x):

        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x) + residual)

        return x