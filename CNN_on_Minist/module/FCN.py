# @Time    : 2021/02/09 02:15
# @Author  : SY.M
# @FileName: FCN.py


from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class FCN(Module):
    """
    全卷积模型 使用卷积层与BatchNorm进行特征提取，卷积之后使用全局平均池化进行降维操作，最后使用一个线性层将维度映射到分类类别的维度
    """
    def __init__(self, in_channels: int,
                 number_of_classes: int):
        super(FCN, self).__init__()

        # conv1 -> relu -> pooling -> conv2 -> relu -> pooling -> FCN1 -> FCN2

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(48)
        self.GVP = nn.AdaptiveAvgPool2d(1)
        # self.conv4 = nn.Conv2d(in_channels=48, out_channels=number_of_classes, kernel_size=1)
        # self.layer_norm = nn.LayerNorm(48)
        # self.linear = nn.Linear(48, number_of_classes)

        self.model_name = self.__class__.__name__
        print('use model:', self.model_name)

    def forward(self, x):
        """
        前向传播
        :param x: 输入
        :return: 输出经过从最后一次卷积之后的特征向量，用于CAM可视化
        """
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))

        CAM_data = x

        x = self.GVP(x)
        # x = self.conv4(x).reshape(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(self.layer_norm(x))

        return x, CAM_data
