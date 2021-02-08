# @Time    : 2021/01/18 28:28
# @Author  : SY.M
# @FileName: loss.py

import torch
from torch.nn import Module

class Myloss(Module):
    """
    使用交叉熵损失
    """
    def __init__(self):
        super(Myloss, self).__init__()

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, y_pre, y_true):

        loss = self.loss_function(y_pre, y_true.long())

        return loss
