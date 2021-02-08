# CNN-on-Minist
基于Pytorch框架，使用CNN模型应用于Minist数据集上的分类任务

## 测试结果
网络|准确率|
----|-----|
Basic CNN|99.19|
Inception CNN|99.47|
Residual CNN|99.47|
FCN|99.15|
ResNet|99.62|

## 各模型简介
### Basic CNN
使用基础的卷积模型架构，使用卷积层、最大池化层、ReLu激活层等对模型进行训练。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Basic%20CNN.png)
### Inception CNN
不同尺寸的卷积核能够对分类的结果产生不同的影响，因此将输入数据使用不同的分支，每个分支使用**不同的卷积核**与输出channel，对数据进行处理，处理过程中通过添加padding保持图像的大小(H和W)不变，对于不同分支的结果，按照channel的维度进行拼接，然后进行输出。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Inception%20CNN.png)
### Residual CNN
随着网络模型的不断加深，在进行反向传播时，由于loss经过层层的求导几乎接近于0，导致靠近模型入口处的权重更新效果不明显，甚至发生梯度消失的情况导致他们没有得到训练。因此构建残差网络模型，即使在模型很深的情况下仍然能够对权重进行很好的训练。残差模块中，输入数据的尺寸不会发生变化(C,H,W都不会变化)。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Residual%20CNN.png)
### FCN
全卷积网络模型，首先舍弃基础CNN中使用最大池化层减少训练参数的方法，使用**BatchNorm**在所有样本上进行归一化操作，加快训练、提高精确度。其次在卷积层后使用**全局平均池化层**(GAP, Global Average Pooling)代替全连接层，由于GAP层没有需要训练的参数，使得模型中需要训练的参数量减少，有效防止过拟合的发生。最后使用线性层对GAP之后的结果映射到分类类别的维度。但是模型的训练会花费更多时间。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/FCN.png)
### ResNet
结合FCN和Residual Net的优点，使用BatchNorm与GAP分别代替Max Pooling和全连接，然后使用残差模块保证反向传播的有效传递。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Resnet.png)

## 数据集介绍
Minist数据集是一个非常经典的数据集，它是由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。常常用于CNN的基础入门数据集，因此第一次接触CNN模型，本文使用此数据集进行学习实验。<br>
下载地址：http://yann.lecun.com/exdb/mnist/ <br>
共包含四个文件，分别为训练集数据、训练集标签、测试集数据、测试集标签，解压缩后为.idx3-ubyte类型文件。

## 数据集处理
使用idx2numpy模块中的convert_from_file函数，直接将文件中的数据读取为numpy数组，进而转化为Tensor向量。<br>
具体代码：`idx2numpy.convert_from_file(path)` <br>
idx2numpy模块的下载：环境命令窗口中输入`pip install idx2numpy` <br>
详细请看data_process.py中的数据处理过程。
