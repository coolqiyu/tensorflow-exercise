# vgg-16网络
import tensorflow as tf

BATCH_SIZE = 100
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3


def weight_variable(shape = []):
    """
    根据shape，随机生成张量w
    :param data:
    :param shape:
    :return:
    """
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)


def bias_variable(shape=[]):
    """
    根据shape，随机生成张量b
    :param shape:
    :return:
    """
    b = tf.constant(0.1, shape)
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


def train_net(input_data):
    """
    网络结构
    :param input_data 输入的数据
    :return:
    """
    x, y = input_data.x, input_data.y

    # conv1
    with tf.name_scope("cov1") as scope:
        w = weight_variable([3, 3, 64])
        b = bias_variable([64])
        conv1 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME") + b
        conv1 = tf.nn.relu(conv1)

    # conv2
    w = weight_variable([3, 3, 64])
    b = bias_variable([64])
    conv2 = tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv2 = tf.nn.relu(conv2)

    # pool3
    pool3 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv4
    w = weight_variable([3, 3, 128])
    b = bias_variable([128])
    conv4 = tf.nn.conv2d(pool3, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv4 = tf.nn.relu(conv4)

    # conv5
    w = weight_variable([3, 3, 128])
    b = bias_variable([128])
    conv5 = tf.nn.conv2d(conv4, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv5 = tf.nn.relu(conv5)

    # pool6
    pool6 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv7
    w = weight_variable([3, 3, 256])
    b = bias_variable([256])
    conv7 = tf.nn.conv2d(pool6, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv7 = tf.nn.relu(conv7)

    # conv8
    w = weight_variable([3, 3, 256])
    b = bias_variable([256])
    conv8 = tf.nn.conv2d(conv7, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv8 = tf.nn.relu(conv8)

    # conv9
    w = weight_variable([3, 3, 256])
    b = bias_variable([256])
    conv9 = tf.nn.conv2d(conv8, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv9 = tf.nn.relu(conv9)

    # pool10
    pool10 = tf.nn.max_pool(conv9, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv11
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv11 = tf.nn.conv2d(pool10, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv11 = tf.nn.relu(conv11)

    # conv12
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv12 = tf.nn.conv2d(conv11, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv12 = tf.nn.relu(conv12)

    # conv13
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv13 = tf.nn.conv2d(conv12, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv13 = tf.nn.relu(conv13)

    # pool14
    pool14 = tf.nn.max_pool(conv13, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    # conv15
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv15 = tf.nn.conv2d(pool14, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv15 = tf.nn.relu(conv15)

    # conv16
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv16 = tf.nn.conv2d(conv15, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv16 = tf.nn.relu(conv16)

    # conv17
    w = weight_variable([3, 3, 512])
    b = bias_variable([512])
    conv17 = tf.nn.conv2d(conv16, w, strides=[1, 1, 1, 1], padding="SAME") + b
    conv17 = tf.nn.relu(conv17)

    # fc18
    w = weight_variable([7*7*512, 4096])
    b = bias_variable([4096])
    fc18 = tf.matmul(conv17, w) + b

    # fc19
    w = weight_variable([4096, 4096])
    b = bias_variable([4096])
    fc19 = tf.matmul(fc18, w) + b

    # fc20
    w = weight_variable([4096, 1000])
    b = bias_variable([1000])
    fc18 = tf.matmul(fc19, w) + b

    result = tf.nn.softmax(fc18)


def train():
    """
    训练过程
    :return:
    """

def test():
    pass