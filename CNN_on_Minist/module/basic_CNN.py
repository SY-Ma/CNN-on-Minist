# @Time    : 2021/01/18 28:20
# @Author  : SY.M
# @FileName: basic_CNN.py


from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class Basic_CNN(Module):
    """
    基础CNN架构 使用卷积层，最大池化，全连接，LayerNorm，ReLu
    """
    def __init__(self, in_channels: int,
                 number_of_classes: int):
        super(Basic_CNN, self).__init__()

        # batch*1*28*28
        # conv1 -> relu -> pooling -> conv2 -> relu -> pooling -> FCN1 -> FCN2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.layer_normal = nn.LayerNorm(128)
        self.linear = nn.Linear(128, number_of_classes)

        self.model_name = self.__class__.__name__
        print('use model:', self.model_name)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = F.relu(self.max_pool(self.conv3(x)))

        x = x.reshape(x.shape[0], -1)
        x = self.layer_normal(x)
        x = self.linear(x)

        return x
