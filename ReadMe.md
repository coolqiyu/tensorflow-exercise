###DL_LEARNING
深度学习
硬件条件有限，都是用mnist数据集测试的(⊙︿⊙)

####AlextNet_like
http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html
与AlexNet相似的网络结构来做cifar数据集的识别
- 相关核心数学对象，如卷积、修正线性激活、最大池化以及局部响应归一化；
- 训练过程中一些网络行为的可视化，这些行为包括输入图像、损失情况、网络行为的分布情况以及梯度；
- 算法学习参数的移动平均值的计算函数，以及在评估阶段使用这些平均值提高预测性能；
- 实现了一种机制，使得学习率随着时间的推移而递减；
- 为输入数据设计预存取队列，将磁盘延迟和高开销的图像预处理操作与模型分离开来处理；

####common
用来放公共的内容，如input_data(获取mnist数据集)

####experiments
放各种不好归类的
- basic_function.py是tensorflow中一些函数的练习
- numpy_learn.py numpy的练习
- tf_example.py tensorflow入门的几个小实例 http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html
- tf_small_nn.py 用tensorflow实现的小网络 http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html

####SimpleNN
自己实现的简单网络，不用tensorflow来实现相关的操作，如反向求导
- Cnn.py：用numpy实现的4层cnn网
- Dnn.py: 用纯python实现的一层简单网络
- DnnTest.py和CnnTest.py：分别是对应上述文件的单测，用unittest框架

###vgg
用tensorflow实现的vgg-16结构，全部使用3x3卷积核


###ResNet
tensorflow实现的ResNet
- subsample: 降采样，用最大池
- block中包含单元unit，一个单元是一个短连接模块
- bottleneck：对一个神经单元前后分别加上1x1卷积，可以较少/增加维度
- depth/bottleneck_depth：最后输出/1x1卷积层 核个数


###overfeat
一个分类、定位、检测的统一框架。对tensorflow给出的代码的改写
LFW数据集：http://vis-www.cs.umass.edu/lfw/
13233 images, 5749 people, 1680 people with two or more images

###YOLO
https://github.com/gliese581gg/YOLO_tensorflow/blob/master/YOLO_face_tf.py


