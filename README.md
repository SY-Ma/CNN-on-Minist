# CNN-on-Minist
基于Pytorch框架，使用CNN模型应用于Minist数据集上的分类任务，**开箱即用**

## 测试结果
网络|准确率|错误率|
----|-----|------|
Basic CNN|99.19|0.81|
Inception CNN|99.47|0.53|
Residual CNN|99.47|0.53|
FCN|99.16|0.84|
ResNet|99.62|0.38|

其他state-of-the-art的模型结果可以从以下网址查看。<br>
地址：http://yann.lecun.com/exdb/mnist/ <br>
本实验未对数据集进行任何的预处理操作，相比之下可以看出，在与Convolutional nets中的结果相比，本文的ResNet的错误率非常之低，相比于大部分未进行预处理的模型甚至进行了预处理的模型结果更好。

## 各模型简介
### Basic CNN
使用基础的卷积模型架构，使用卷积层、**最大池化层**、ReLu激活层等对模型进行训练。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Basic%20CNN.png)

---

### Inception CNN
不同尺寸的卷积核能够对分类的结果产生不同的影响，因此将输入数据使用不同的分支，每个分支使用**不同的卷积核**与输出channel，对数据进行处理，处理过程中通过添加padding保持图像的大小(H和W)不变，对于不同分支的结果，按照channel的维度进行拼接，然后进行输出。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Inception%20CNN.png)

---

### Residual CNN
随着网络模型的不断加深，在进行反向传播时，由于loss经过层层的求导几乎接近于0，导致靠近模型入口处的权重更新效果不明显，甚至发生**梯度消失**的情况导致他们没有得到训练。因此构建残差网络模型，即使在模型很深的情况下仍然能够对权重进行很好的训练。残差模块中，输入数据的尺寸**不会发生变化**(C,H,W都不会变化)。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Residual%20CNN.png)

---

### FCN
全卷积网络模型，首先舍弃基础CNN中使用最大池化层减少训练参数的方法，使用**BatchNorm**在所有样本上进行归一化操作，加快训练、提高精确度。其次在卷积层后使用**全局平均池化层**(GAP, Global Average Pooling)代替全连接层，由于GAP层没有需要训练的参数，使得模型中需要训练的参数量减少，有效防止过拟合的发生。最后使用线性层对GAP之后的结果映射到分类类别的维度。但是模型的训练会花费更多时间。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/FCN%20Network.png)

---

### ResNet
**结合FCN和Residual Net的优点**，使用BatchNorm与GAP分别代替Max Pooling和全连接，然后使用残差模块保证反向传播的有效传递。<br>
![Image text](https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN%20Structure%20Chart/Resnet.png)

## 数据集简介
Minist数据集是一个非常经典的数据集，它是由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。常常用于CNN的基础入门数据集，因此第一次接触CNN模型，本文使用此数据集进行学习实验。<br>
下载地址：http://yann.lecun.com/exdb/mnist/ <br>
共包含四个文件，分别为训练集数据、训练集标签、测试集数据、测试集标签，解压缩后为.idx3-ubyte类型文件。

## 实验环境
环境|描述|
----|----|
语言|python3.7|
框架|pytorch1.6.0 pytorch0.4.1 pytorch1.7.0|
IDE|Pycharm and Colab|
操作系统|Windows10|
packages|torch time tqdm os idx2numpy numpy matplotlib cv2 datetime|

## 数据集处理
使用idx2numpy模块中的convert_from_file函数，直接将文件中的数据读取为numpy数组，进而转化为Tensor向量。<br>
具体代码：`idx2numpy.convert_from_file(path)` <br>
idx2numpy模块的下载：环境命令窗口中输入`pip install idx2numpy` <br>
详细请看data_process.py中的数据处理过程。

## 文件描述
文件名称|描述|
-------|----|
data_process|数据集处理|
font|存储字体，用于结果图中的文字|
CAM_result_figure|CAM结果图|
module|模型的各个模块|
mytest|各种测试代码|
reslut_figure|准确率结果图|
saved_model|保存的pkl文件|
utils|工具类文件夹|
run.py|训练模型|
run_with_saved_model.py|使用训练好的模型（保存为pkl文件）测试结果|

## utils工具描述
- random_seed:设置随机种子，使每一次的实验结果可复现。
- visualization：结果图的绘制，包括loss的变化趋势，测试集与训练集准确率变化趋势等。
- CAM：绘制CAM热力图，旨在观察模型更加关注于图片的什么位置来进行类别的归类。

## CAM(Class Activation Mapping)结果分析
CAM旨在观察CNN模型在进行分类时更加关注图片的哪部分特征。得到的结果并非一开始认为的，会更加关注图片的表述数字的部分。如下图中的黄颜色的部分。<br>
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CAM%20result%20figure/x2%201.jpg" width="100" height="100" align="bottom" /> 
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/1.png" width="100" height="100" align="bottom" /> 
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/3.png" width="100" height="100" align="bottom" /> 
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/7.png" width="100" height="100" align="bottom" /> 
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CAM%20result%20figure/4.png" width="100" height="100" align="bottom" /> <br>
然而，模型似乎更加关注图片中数字的轮廓，即颜色快速变化的陡坡，相当于灰度值迅速变化的边界。使用FCN模型在数据集上的CAM绘制结果如下图所示,其对应数字分别为1，1，3，7，4。对于CAM图像与结果分析，期望更多实验与数据的探索。<br>
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CAM%20result%20figure/number%3D1%202021-02-04%2020-30-21.jpg" width="100" height="100" align="bottom" />
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/number%3D1%202021-02-08%2019-45-11.jpg" width="100" height="100" align="bottom" />
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/number%3D3%202021-02-08%2019-47-07.jpg" width="100" height="100" align="bottom" />
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CNN_on_Minist/CAM_result_figure/number%3D7%202021-02-08%2019-41-02.jpg" width="100" height="100" align="bottom" />
<img src="https://github.com/SY-Ma/CNN-on-Minist/blob/main/CAM%20result%20figure/number%3D4%202021-02-08%2019-37-17.jpg" width="100" height="100" align="bottom" />

## CNN学习心得
- pytorch中，CNN模型使用四维向量作为输入矩阵，维度为：`[BatchSize, Channels, Height, Width]`
- BatchNorm2d是对Batch，H，W三者上的归一化，因此传入参数应该为输入的四维向量的channel数。LayerNorm是对C，H，W三者上的归一化，输入维度为三者之积。
- 随着卷积层数的叠加，模型的准确率也趋于增高。
- Max Pooling效果好于Average Pooling。
- BatchNorm有利于加快学习的速度，增加准确率，使用BatchNorm代替MaxPooling，不会改变图形的尺寸。可以对比ResNet和Residual Net。
- 经验总结，使用BatchNorm代替MaxPooling不改变图形尺寸，使得卷积核得到更多次的训练，有助于提升模型的准确率。
- 建议不断增加channel的深度，不改变图片尺寸。虽然这样会花费更多的时间进行训练。
- 残差链接在模型较深的情况下效果显著，对比ResNet和FCN。
- ResNet使用尺寸为1的卷积核，在各个Residual Block之间改变维度。使得模型自始自终未改变图形的尺寸。
- 卷积层的输出为四维向量，若想要映射到输出类别，需要进行降维操作，在这里有两种方案：一是进行reshape操作；二是使用全剧平均池化操作(GAP)，在每个channel上求其平均值，再使用squeeze降维。两者均可以达到降维的目的，不过，GAP相比于全连接层数度运算速度更快，没有参数需要训练，但是模型的收敛速度会变慢，可能是因为训练的压力都堆积在卷积层。
- 若使用GAP进行降维操作，若想将维度映射到分类类别有两种方法：一是在GAP之前最近的卷积层，使其输出的channel即为分类类别数，经过GAP与squeeze之后即为得到的最终输出；二是不要求GAP层的channel数，在GAP之后，squeeze之前再接一层kernel size为1的卷积层进行channel的改变；三是不要求GAP层的channel数，而是在squeeze操作之后再接一个全连接进行映射映射。实验表明，方法三的效果更好。
- reshape之后，全连接之前，使用LayerNorm能够大大提升模型的准确率。
- CAM旨在观察训练得到的模型关注了原图片中的什么部分。只有使用了GAP且在之后进行全连接操作才可以进行CAM的分析。在最后一层卷积层中，维度为`[B, C, H, W]`,经过GAP之后为`[B, C, 1, 1]`,经过squeeze与线性层之后为`[B, number of classes]`,因此全连接层中的权重矩阵为`[number of classes, C]`,那么对于任意一个下标的向量维度即为`[C,]`，相当于最后一层卷积输出向量的每一个channel上的权重。而且，特定下标的`[C,]`向量是驱使卷积输出得到的预测为对应下标的权重向量。具体请参考参考中的相关内容。

## 参考
- 论文：
  ```
  @inproceedings{2017Time,
  title={Time series classification from scratch with deep neural networks: A strong baseline},
  author={ Wang, Zhiguang  and  Yan, Weizhong  and  Oates, Tim },
  booktitle={2017 International Joint Conference on Neural Networks (IJCNN)},
  year={2017},
  }
  ```
- pytorch中的归一化方法：https://mp.weixin.qq.com/s/jlAHWNTjZkaS5ps2rSvsrQ
- 全局平均池化：https://blog.csdn.net/kinggang2017/article/details/88869673
- CAM：https://www.cnblogs.com/luofeel/p/10400954.html <br>
  https://blog.csdn.net/u012426298/article/details/82689969 
  https://blog.csdn.net/u014264373/article/details/85415921 
- 基础知识视频：https://www.bilibili.com/video/BV1Y7411d7Ys?p=11&t=2956


## 本人学识浅薄，代码和文字若有不当之处欢迎批评与指正！
## 联系方式：masiyuan007@qq.com

