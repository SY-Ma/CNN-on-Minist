# @Time    : 2021/02/16 05:58
# @Author  : SY.M
# @FileName: CAM.py

import torch
import numpy as np
from cv2 import cv2
from datetime import datetime


def cam_figure(draw_data, labels, net):
    """
    绘制热力图
    :param draw_data: 最后一层卷积之后的 四维向量  batch channel H W
    :param labels: 对应标签向量
    :param net: 使用的网络，获取其linear层的权重
    :return: 无
    """
    m = 0  # 表示获取一个batch中第几个样本
    draw_data = draw_data[m]
    # weight的下标所对应的权重 是 使得特征预测出来的 数字 为下标的权重 所以图片中数字是几，这个下标应该就是几
    weight = net.linear.weight[int(labels[m])]
    # 乘以相应的权重
    for index, (i, j) in enumerate(zip(draw_data, weight)):
        draw_data[index] = torch.mul(i, j)
    # 所有channel相加求平均

    # draw_data = torch.mean(draw_data, dim=0)
    # mean = torch.mean(draw_data.unsqueeze(dim=0), dim=0)
    # std = torch.std(draw_data.unsqueeze(dim=0), dim=0)
    # draw_data = (draw_data - mean) / std * 255
    draw_data = torch.mean(draw_data, dim=0) * 200
    draw_data = draw_data.numpy().astype(np.uint8)

    # 获取图片的矩阵 用于 与图片的交叠显示
    # img = cv2.imread('x0 7.jpg')
    # height, width, _ = img.shape
    # heatmap = cv2.applyColorMap(cv2.resize(draw_data, (height, width)), cv2.COLORMAP_JET)
    # result = heatmap * 0.3 + img * 0.5

    heatmap = cv2.applyColorMap(draw_data, cv2.COLORMAP_JET)
    result = heatmap
    time = datetime.strftime(datetime.now(), '%Y-%m-%d %H-%M-%S')
    cv2.imwrite(f'CAM_result_figure/number={int(labels[m])} {time}.jpg', result)
