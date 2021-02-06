# @Time    : 2021/01/17 31:16
# @Author  : SY.M
# @FileName: CAM_test.py

import torch
from cv2 import cv2
from data_process.data_process import MyDataset
import idx2numpy
import matplotlib.pyplot as plt
import numpy as np


# net = torch.load('saved_model/FCN 98.98 batch=4 torch_version=1.6.0 .pkl')
#
# print(net)

train_X_path = 'E:/PyCharmWorkSpace/dataset/Minist/train-images.idx3-ubyte'
train_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\train-labels.idx1-ubyte'

train_X = idx2numpy.convert_from_file(train_X_path)
train_Y = idx2numpy.convert_from_file(train_Y_path)

print(type(train_X))
print(train_X.shape)
print(type(train_Y))
print(train_Y.shape)
print(set(train_Y.tolist()))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

plt.figure(figsize=(0.1, 0.1), dpi=1000)
plt.figimage(train_X[0])
plt.show()


CAMs = torch.Tensor(range(28*28)).reshape(28, 28).numpy().astype(np.uint8)
print(type(CAMs))
img = cv2.imread('../5.jpg')
height, width, _ = img.shape
print(img.shape)
heatmap = cv2.applyColorMap(cv2.resize(CAMs, (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('../CAM.jpg', result)