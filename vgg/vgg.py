# vgg-16网络
import tensorflow as tf
import Mnist.input_data as input_data
from tensorflow.python import debug as tfdbg

BATCH_SIZE = 100
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1


def weight_variable(shape = []):
    """
    根据shape，随机生成张量w
    :param data:
    :param shape:
    :return:
    """
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)


def bias_variable(shape):
    """
    根据shape，随机生成张量b
    :param shape:
    :return:
    """
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)


def read_data():
    """
    从磁盘读取数据
    :return:
    """


def loss(y_, y):
    """
    定义loss函数
    :param y_: 执行结果
    :param y: 标签label
    :return:
    """
    return -tf.reduce_sum(y * tf.log(y_))

def train_net_large(x, y):
    """
    网络结构
    :param input_data 输入的数据
    :return:
    """
    # x, y = input_data.x, input_data.y

    # conv1
    w = weight_variable([3, 3, IMAGE_CHANNEL, 64])
    b = bias_variable([64])
    conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv1 = tf.nn.relu(conv1)

    # conv2
    w = weight_variable([3, 3, 64, 64])
    b = bias_variable([64])
    conv2 = tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv2 = tf.nn.relu(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv3
    w = weight_variable([3, 3, 64, 128])
    b = bias_variable([128])
    conv3 = tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv3 = tf.nn.relu(conv3)

    # conv4
    w = weight_variable([3, 3, 128, 128])
    b = bias_variable([128])
    conv4 = tf.nn.conv2d(conv3, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv4 = tf.nn.relu(conv4)

    # pool4
    pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv5
    w = weight_variable([3, 3, 128, 256])
    b = bias_variable([256])
    conv5 = tf.nn.conv2d(pool4, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv5 = tf.nn.relu(conv5)

    # conv6
    w = weight_variable([3, 3, 256, 256])
    b = bias_variable([256])
    conv6 = tf.nn.conv2d(conv5, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv6 = tf.nn.relu(conv6)

    # conv7
    w = weight_variable([3, 3, 256, 256])
    b = bias_variable([256])
    conv7 = tf.nn.conv2d(conv6, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv7 = tf.nn.relu(conv7)

    # pool7
    pool7 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv8
    w = weight_variable([3, 3, 256, 512])
    b = bias_variable([512])
    conv8 = tf.nn.conv2d(pool7, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv8 = tf.nn.relu(conv8)

    # conv9
    w = weight_variable([3, 3, 512, 512])
    b = bias_variable([512])
    conv9 = tf.nn.conv2d(conv8, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv9 = tf.nn.relu(conv9)

    # conv10
    w = weight_variable([3, 3, 512, 512])
    b = bias_variable([512])
    conv10 = tf.nn.conv2d(conv9, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv10 = tf.nn.relu(conv10)

    # pool10
    pool10 = tf.nn.max_pool(conv10, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv11
    w = weight_variable([3, 3, 512, 512])
    b = bias_variable([512])
    conv11 = tf.nn.conv2d(pool10, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv11 = tf.nn.relu(conv11)

    # conv12
    w = weight_variable([3, 3, 512, 512])
    b = bias_variable([512])
    conv12 = tf.nn.conv2d(conv11, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv12 = tf.nn.relu(conv12)

    # conv13
    w = weight_variable([3, 3, 512, 512])
    b = bias_variable([512])
    conv13 = tf.nn.conv2d(conv12, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv13 = tf.nn.relu(conv13)

    # pool13
    pool13 = tf.nn.max_pool(conv13, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
    pool13 = tf.reshape(pool13, [BATCH_SIZE, 7 * 7 * 512])

    # fc14
    w = weight_variable([7*7*512, 4096])
    b = bias_variable([4096])
    fc14 = tf.nn.relu(tf.matmul(pool13, w) + b)

    # fc15
    w = weight_variable([4096, 4096])
    b = bias_variable([4096])
    fc15 = tf.nn.relu(tf.matmul(fc14, w) + b)

    # fc16
    w = weight_variable([4096, 10])
    b = bias_variable([10])
    fc16 = tf.nn.relu(tf.matmul(fc15, w) + b)

    result = tf.nn.softmax(fc16)

    return result


def train_net_small(x0, y):
    """
    网络结构
    :param input_data 输入的数据
    :return:
    """
    # x, y = input_data.x, input_data.y
    x = tf.reshape(x0, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    # conv1
    global w_conv1
    w_conv1 = weight_variable([3, 3, IMAGE_CHANNEL, 64])
    b_conv1 = bias_variable([64])
    conv1 = tf.nn.conv2d(x, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1
    conv1 = tf.nn.relu(conv1)

    # conv2
    w_conv2 = weight_variable([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
    conv2 = tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2
    conv2 = tf.nn.relu(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv3
    w_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    conv3 = tf.nn.conv2d(pool2, w_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3
    conv3 = tf.nn.relu(conv3)

    # conv4
    w_conv4 = weight_variable([3, 3, 128, 128])
    b_conv4 = bias_variable([128])
    conv4 = tf.nn.conv2d(conv3, w_conv4, strides=[1, 1, 1, 1], padding="SAME") + b_conv4
    conv4 = tf.nn.relu(conv4)

    # pool4
    pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
    pool4 = tf.reshape(pool4, [BATCH_SIZE, 7 * 7 * 128])

    # fc14
    w_fc14 = weight_variable([7*7*128, 1024])
    b_fc14 = bias_variable([1024])
    fc14 = tf.nn.relu(tf.matmul(pool4, w_fc14) + b_fc14)

    # fc15
    w_fc15 = weight_variable([1024, 1024])
    b_fc15 = bias_variable([1024])
    fc15 = tf.nn.relu(tf.matmul(fc14, w_fc15) + b_fc15)

    # fc16
    w_fc16 = weight_variable([1024, 10])
    b_fc16 = bias_variable([10])
    fc16 = tf.nn.relu(tf.matmul(fc15, w_fc16) + b_fc16)

    result = tf.nn.softmax(fc16)

    return result


def train():
    """
    训练过程
    :return:
    """
    x = tf.placeholder("float", shape=[None, IMAGE_CHANNEL * IMAGE_WIDTH * IMAGE_HEIGHT])
    y = tf.placeholder("float", shape=[None, 10])
    y_ = train_net_small(x, y)
    cross_entropy = -tf.reduce_sum(y * tf.log(y_))# loss(y_, y)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(10):
            batch = mnist.train.next_batch(BATCH_SIZE)
            # mx = tf.reshape(mx, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
            # mx = tf.transpose(mx, [0, 2, 3, 1])
            # my = tf.reshape(my, [BATCH_SIZE, 10])
            # mx, my = sess.run([mx, my])
            print(sess.run([train_step, cross_entropy, y_[0]], feed_dict={x: batch[0], y: batch[1]}))

        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # mx = mnist.test.images
        # mx = tf.reshape(mx, [len(mx), IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])
        # mx = tf.transpose(mx, [0, 2, 1, 3])
        # mx = sess.run(mx)
        # my = mnist.test.labels
        # x = tf.placeholder("float", mx.shape)
        # y = tf.placeholder("float", my.shape)
        # print(sess.run(accuracy, feed_dict={x: mx, y: my}))


def test():
    pass


if __name__ == "__main__":
    train()