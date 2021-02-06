# @Time    : 2021/01/13 30:00
# @Author  : SY.M
# @FileName: run_with_saved_model.py

from data_process.data_process import MyDataset
from torch.utils.data import DataLoader
import torch
print('当前pytorch版本：', torch.__version__)
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
from datetime import datetime
from utils.CAM import cam_figure

test_X_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-images.idx3-ubyte'
test_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-labels.idx1-ubyte'

model_path = 'saved_model/FCN 99.01 batch=4 torch_version=1.6.0 .pkl'
model_name = model_path.split('/')[-1].split(' ')[0]
print(f'use model:{model_name}')
BATCH_SIZE = int(model_path.split(' ')[2][model_path.split(' ')[2].index('=')+1:])
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'use device: {DEVICE}')

net = torch.load(model_path)

test_dataset = MyDataset(test_X_path, test_Y_path)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
TEST_LEN = test_dataset.X_len

def test():
    net.eval()
    correct = 0
    CAM_or_not = True
    with torch.no_grad():
        for num, (x, y) in enumerate(test_dataLoader):
            y_pre, CAM_data = net(x.to(DEVICE))
            _, max_index = torch.max(y_pre, dim=-1)
            correct += torch.eq(max_index, y).sum().item()

            if CAM_or_not:
                cam_figure(draw_data=CAM_data, labels=y, net=net)
                CAM_or_not = False

        print(f'Accuracy:{round(100 * correct / TEST_LEN, 2)}%')


if __name__ == '__main__':
    test()
