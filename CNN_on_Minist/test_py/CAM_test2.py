# @Time    : 2021/02/16 04:00
# @Author  : SY.M
# @FileName: CAM_test2.py


import torch

t1 = torch.Tensor(range(12)).reshape(2, 2, 3)

t2 = torch.Tensor(range(2))

print(t1)
for index, (i, j) in enumerate(zip(t1, t2)):
    t1[index] = torch.mul(i, j)

print(t1)

print(t2.shape)