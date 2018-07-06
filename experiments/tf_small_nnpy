#
# tensorflow实现的对mnist数据集的几个简单网络
#==================================================

import tensorflow as tf
from common import input_data

DEBUG = True

# 一个隐藏层网络
def one_layer_mnist_train():
    # None表示此张量的第一个维度可以是任何长度的
    x = tf.placeholder("float", [None, 784])
    # 初始化两个参数
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # softmax函数
    y = tf.nn.softmax(tf.matmul(x, W) + b) # 执行结果
    y_ = tf.placeholder("float", [None, 10]) # 标签，正确结果

    # 交叉熵，成本函数
    # tf.reduce_sum 计算张量的所有元素的总和
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # 梯度下降法来优化成本函数
    # 下行代码往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    with tf.Session() as sess:
        sess.run(init)

        # 循环遍历1000次训练模型
        for i in range(1000):
            # 每一步迭代加载100个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y张量占位符用训练训练数据替代
            batch_xs, batch_ys = mnist.train.next_batch(100)
            print(sess.run([train_step, W, b, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys}))


        # 下面判断测试集，应该相当于在网络后增加节点
        # 利用test数据集判断模型的结果
        # tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 多层卷积网络
def four_layers_mnist_train():
    # 数据集
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])

    # 第一层卷积
    W_conv1 = weight_variable([5, 5, 1, 32]) # 5*5*1
    b_conv1 = bias_variable([32]) # 32
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 28*28*1
    # x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28*28*32
    h_pool1 = max_pool_2x2(h_conv1) # 14*14*32

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64]) # 5*5*32
    b_conv2 = bias_variable([64]) # 64
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #14*14*64
    h_pool2 = max_pool_2x2(h_conv2)#7*7*64

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    #shape中使用-1，张量该维度被压缩，少一维。then tensor is flattened and the operation outputs a 1-D tensor with all elements of tensor
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])# 把h_pool2变成一维7*7*64行
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)# 1024

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层 softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # y_conv是结果
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 计算精确度，其实也是在图上加了节点：将最后的输出再往下走
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # 遍历mini-batch
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            if i % 50 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print(
        "test accuracy %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 0.5}))

# 两个用于初始化的函数
def weight_variable(shape):
    # 以shape产生正太分布，值随机，stddev是标准差，mean：均值
    # tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布，用来初始化权重变量
    # 其中第一个维度代表该层中权重变量所连接（connect from）的单元数量，第二个维度代表该层中权重变量所连接到的（connect to）单元数量。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"""
tf.nn.conv2d(input, filter, strides, padding)
 input:输入4维，[batch_size, in_height, in_width, in_channel]
 filter：[in_height, in_width, in_channel, filter_size]  filter_size：包含几个过滤器=输出的channel
 strides：步长，4维，顺序为NHWC，也就是[batch, height, width, channel]，且要求strides[0]=strides[3]=1
 padding: A `string` from: `"SAME", "VALID"`.决定padding的算法，不是自己指定padding的大小
 use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
 data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
 Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
 输出的形状：shape(output) = [batch, (in_height - filter_height + 1) / strides[1], (in_width - filter_width + 1) / strides[2],...]
            padding = 'SAME': 向下取整 (only full size windows are considered).
            padding = 'VALID': 向上取整 (partial windows are included).
            padding填充的个数为0
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
def max_pool_2x2(x):
    """
    ksize: 要池化的窗口大小
    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 2, 1], padding="SAME")


# tensorboard例子
def tensorboard():
    # None表示此张量的第一个维度可以是任何长度的
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])  # 标签，正确结果

    # 初始化两个参数
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    m = [1,2,3,4,5,6]
    tf.summary.histogram("xx", b)
    # softmax函数
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # 执行结果

    # 交叉熵，成本函数
    # tf.reduce_sum 计算张量的所有元素的总和
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 梯度下降法来优化成本函数
    # 下行代码往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    prediction_train = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy_train = tf.reduce_mean(tf.cast(prediction_train, "float"))
    tf.summary.scalar("accuarcy_train", accuracy_train)

    # 显示图像
    batch_xs, batch_ys = mnist.train.next_batch(100)
    tf.summary.image('images', tf.reshape(batch_xs, [100, 28, 28, 1]))

    # 用于tensorboard
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter("./resource/mnist_logs", sess.graph)

        # 循环遍历1000次训练模型
        for i in range(1000):
            # 每一步迭代加载100个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y张量占位符用训练训练数据替代
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})

            if i % 10 == 0:
                train_writer.add_summary(summary, i)
                for index, d in enumerate(m):
                    m[index] -= 0.1
            batch_xs, batch_ys = mnist.train.next_batch(100)
        train_writer.close()

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))