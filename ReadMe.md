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
- Cnn.py：用numpy实现的4层cnn网络
- Dnn.py: 用纯python实现的一层简单网络
- DnnTest.py和CnnTest.py：分别是对应上述文件的单测，用unittest框架