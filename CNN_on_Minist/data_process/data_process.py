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
