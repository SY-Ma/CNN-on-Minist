# @Time    : 2021/01/18 28:07
# @Author  : SY.M
# @FileName: run.py


import torch
from data_process.data_process import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from time import time
from tqdm import tqdm
import os

from utils.random_seed import setup_seed
from utils.visualization import result_visualization
from module.basic_CNN import Basic_CNN
from module.inception_CNN import Inception_CNN
from module.residual_CNN import Residual_CNN
from module.FCN import FCN
from module.resNet import ResNet
from module.loss import Myloss

setup_seed(30)  # 设置随机种子

# 定义数据集文件路径
train_X_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\train-images.idx3-ubyte'
train_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\train-labels.idx1-ubyte'
test_X_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-images.idx3-ubyte'
test_Y_path = 'E:\\PyCharmWorkSpace\\dataset\\Minist\\t10k-labels.idx1-ubyte'

# 保存结果图像的根目录
result_figure_path = 'result_figure'

draw_key = 1  # 大于等于draw_key才会保存结果图
test_interval = 5  # 测试间隔 单位epoch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'use device: {DEVICE}')

EPOCH = 100
BATCH_SIZE = 4
LR = 1e-4
optimizer_name = 'Adam'  # 优化器选择
number_of_classes = 10  # 分类类别

train_dataset = MyDataset(train_X_path, train_Y_path)
test_dataset = MyDataset(test_X_path, test_Y_path)
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
DATA_LEN = train_dataset.X_len

# 模型选择
# net = Basic_CNN(in_channels=1, number_of_classes=number_of_classes).to(DEVICE)  # 基础CNN架构
# net = Inception_CNN(in_channels=1, number_of_classes=number_of_classes).to(DEVICE)  # Inception
# net = Residual_CNN(in_channels=1, number_of_classes=number_of_classes).to(DEVICE)  # 残差网络
net = FCN(in_channels=1, number_of_classes=number_of_classes).to(DEVICE)  # 全卷积
# net = ResNet(in_channels=1, number_of_classes=number_of_classes).to(DEVICE)  # 全卷积+残差链接

if optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=LR)
elif optimizer_name == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = Myloss()  # 交叉熵损失函数

accuracy_on_train = []  # 记录训练集上的准确率变化
accuracy_on_test = []  # 记录测试集上的准确率变化


def test(dataLoader, dataset='test'):
    """
    测试方法
    :param dataLoader: 测试集或训练集的dataloader
    :param dataset: 表明是测试集还是训练集
    :return: 准确率
    """
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataLoader:
            if net.model_name == 'FCN' or net.model_name == 'ResNet':
                y_pre, _ = net(x.to(DEVICE))
            else:
                y_pre = net(x.to(DEVICE))
            _, label_index = torch.max(y_pre, dim=-1)
            total += label_index.shape[0]
            correct += torch.sum(torch.eq(label_index, y.long().to(DEVICE))).item()
        if dataset == 'train':
            accuracy_on_train.append(round(100 * correct / total, 2))
        else:
            accuracy_on_test.append(round(100 * correct / total, 2))
        print(f'accuracy on {dataset}:{round(100 * correct / total, 2)}%')

    return round(100 * correct / total, 2)


loss_list = []  # 记录损失变化 这里记录的是每个样本上的平均损失


def train():
    """
    训练方法
    :return: 无
    """
    net.train()
    pbar = tqdm(total=EPOCH)
    begain = time()
    max_accuracy = 0
    for index in range(EPOCH):
        loss_sum = 0
        for x, y in train_dataLoader:
            optimizer.zero_grad()

            if net.model_name == 'FCN' or net.model_name == 'ResNet':
                y_pre, _ = net(x.to(DEVICE))
            else:
                y_pre = net(x.to(DEVICE))

            loss = loss_function(y_pre, y.to(DEVICE))

            loss_sum += loss.item()

            loss.backward()

            optimizer.step()

        loss_list.append(round(loss_sum / DATA_LEN, 2))
        print(f'Epoch:{index + 1}\t\tloss:{loss_sum}')

        if (index + 1) % test_interval == 0:
            current_accuracy = test(test_dataLoader, 'test')
            test(train_dataLoader, 'train')
            print(f'目前最大准确率 测试集:{max(accuracy_on_test)}%\t训练集:{max(accuracy_on_train)}%')
            # 保存最高准确率的模型为 .pkl 文件
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'saved_model/{net.model_name} batch={BATCH_SIZE}.pkl')

        pbar.update()

    # 为.pkl文件命名
    if not os.path.exists(
            f'saved_model/{net.model_name} {max_accuracy} batch={BATCH_SIZE} torch_version={torch.__version__}.pkl'):
        os.rename(f'saved_model/{net.model_name} batch={BATCH_SIZE}.pkl',
                  f'saved_model/{net.model_name} {max_accuracy} batch={BATCH_SIZE} torch_version={torch.__version__} .pkl')
    else:
        print(
            f"文件 saved_model/{net.model_name} {max_accuracy} batch={BATCH_SIZE} torch_version={torch.__version__} .pkl 已存在！")

    end = time()
    time_cost = round((end - begain) / 60, 2)

    # 结果可视化操作
    result_visualization(loss_list=loss_list, accuracy_on_test=accuracy_on_test, accuracy_on_train=accuracy_on_train,
                         test_interval=test_interval, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
                         time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=result_figure_path,
                         optimizer_name=optimizer_name, LR=LR, model_name=net.model_name)


if __name__ == '__main__':
    train()
