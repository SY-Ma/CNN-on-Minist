# @Time    : 2021/01/13 30:00
# @Author  : SY.M
# @FileName: run_with_saved_model.py

from data_process.data_process import MyDataset
from torch.utils.data import DataLoader
import torch
print('当前pytorch版本：', torch.__version__)
from utils.CAM import cam_figure

# 测试集路径
test_X_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-images.idx3-ubyte'
test_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-labels.idx1-ubyte'

# 使用的.pkl文件
model_path = 'saved_model/Residual_CNN 99.47 batch=4 torch_version=1.7.0+cu101 .pkl'
model_name = model_path.split('/')[-1].split(' ')[0]  # 获取使用的模型
print(f'use model:{model_name}')
BATCH_SIZE = int(model_path.split(' ')[2][model_path.split(' ')[2].index('=')+1:])  # 获取Batch Size
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'use device: {DEVICE}')

# 加载模型
net = torch.load(model_path, map_location=torch.device('cpu'))
print('模型结构:\r\n', net)

test_dataset = MyDataset(test_X_path, test_Y_path)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
TEST_LEN = test_dataset.X_len


def test():
    net.eval()
    correct = 0
    CAM_or_not = False  # 是否进行CAM绘图
    with torch.no_grad():
        for num, (x, y) in enumerate(test_dataLoader):

            if net.model_name == 'FCN' or net.model_name == 'ResNet':
                y_pre, CAM_data = net(x.to(DEVICE))
            else:
                y_pre = net(x.to(DEVICE))

            _, max_index = torch.max(y_pre, dim=-1)
            correct += torch.eq(max_index, y).sum().item()

            if CAM_or_not:
                cam_figure(draw_data=CAM_data, labels=y, net=net)
                CAM_or_not = False

        print(f'Accuracy:{round(100 * correct / TEST_LEN, 2)}%')


if __name__ == '__main__':
    test()
