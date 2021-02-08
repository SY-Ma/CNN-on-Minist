# @Time    : 2021/01/17 28:13
# @Author  : SY.M
# @FileName: data_process.py


import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import idx2numpy
import numpy as np


class MyDataset(Dataset):
    def __init__(self, X_path, Y_path):
        super(MyDataset, self).__init__()
        self.X = torch.Tensor(idx2numpy.convert_from_file(X_path).tolist()).unsqueeze(dim=1)
        self.Y = torch.Tensor(idx2numpy.convert_from_file(Y_path).tolist())
        self.X_len = self.X.shape[0]
        self.Y_len = self.Y.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.X_len


if __name__ == '__main__':
    train_X_path = 'E:/PyCharmWorkSpace/dataset/Minist/train-images.idx3-ubyte'
    train_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\train-labels.idx1-ubyte'

    # train_X = idx2numpy.convert_from_file(train_X_path)
    # train_Y = idx2numpy.convert_from_file(train_Y_path)

    # print(type(train_X))
    # print(train_X.shape)
    # print(type(train_Y))
    # print(train_Y.shape)
    # print(set(train_Y.tolist()))  # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    # plt.figimage(train_X[0])
    # plt.show()

    dataset = MyDataset(X_path=train_X_path, Y_path=train_Y_path)
    print(type(dataset.X))

    a = torch.Tensor([1, 2, 2])
    b = torch.Tensor([1, 2, 3])

    print(torch.sum(torch.equal(a, b)))
