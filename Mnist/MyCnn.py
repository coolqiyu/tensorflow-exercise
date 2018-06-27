#
# 用numpy实现minist的卷积网络
# =================================
import numpy as np
import input_data


BATCH_SIZE = 10
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNEL = 1


def pad_algorithm(x, f, stride, padding):
    """
    执行padding算法
    :param x: 4维数据[N, H, W, C]
    :param f: 4维数据[H, W, InputC, OutputC]
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return: 返回pad算法后的数据张量的shape[N, H, W, C]，以及pad之后的x
    """
    x_shape = np.shape(x)
    f_shape = np.shape(f)

    if padding.lower() == "same":
        # h/stride的上界
        o_shape = [x_shape[0], np.ceil(x_shape[1]/stride[1]),
                   np.ceil(x_shape[2]/stride[2]), f_shape[3]]
        # 对输入进行填充，取计算结果的上界
        h_pad = np.ceil((stride[1] * x_shape[1] - stride[1] + f_shape[0] - x_shape[1]) / 2)
        w_pad = np.ceil((stride[2] * x_shape[2] - stride[2] + f_shape[1] - x_shape[1]) / 2)
        x = np.pad(x, [[0, 0], [np.ceil(h_pad/2), np.floor(h_pad/2)], [np.ceil(w_pad/2), np.floor(w_pad/2)], [0, 0]], 'constant', constant_values=0)
    elif padding.lower() == "valid":
        # (h-f/stride)+1的下界
        o_shape = [x_shape[0], np.floor((x_shape[1] - f_shape[1]) / stride[1]) + 1,
                        np.floor((x_shape[2] - f_shape[2]) / stride[2]) + 1, f_shape[3]]
    else:
        raise Exception("参数padding值{0}不合法，应该为same或valid".format(padding))
    return o_shape, x


def conv2d(x, f, stride=[1, 1, 1, 1], padding="SAME"):
    """
    x为输入数据，f为卷积过滤器
    :param x: 4维数据[N, H, W, C]
    :param f: 4维数据[H, W, InputC, OutputC]
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return: 卷积结果 [N, H, W, C]
    """
    x_shape = np.shape(x)
    f_shape = np.shape(f)

    # 要求两个channel一样
    assert x_shape[3] == f_shape[2]

    o_shape, x = pad_algorithm(x, f, stride, padding)
    out = np.zeros(o_shape)

    # batch
    for b_i in range(o_shape[0]):
        # 纵向
        for h_i in range(o_shape[1]):
            # 横向
            for w_i in range(o_shape[2]):
                #过滤器
                for f_i in range(o_shape[3]):
                    out[b_i][h_i][w_i][f_i] = np.sum(np.multiply(x[b_i][h_i * stride[1]:h_i * stride[1] + f_shape[0]][w_i * stride[2]: w_i * stride[2] + f_shape[1]][:], f[:][:][:][f_i]))
    return out


def max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 3, 1], padding="VALID"):
    """
    最大池化
    :param x: 4维数据[N, H, W, C]
    :param ksize: 4维数据[N, H, W, C] 每一维要池化的范围大小
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return:池化的结果 [N, H, W, C]
    """
    x_shape = np.shape(x)
    k_shape = np.shape(ksize)

    # 为了直接使用pad_algorithm，需要对ksize进行维度调整
    o_shape, x = pad_algorithm(x, np.transpose(ksize, [1, 2, 0, 3]), stride, padding)
    o_shape[3] = x_shape[3]
    out = np.zeros(o_shape)

    # batch
    for b_i in range(o_shape[0]):
        # 纵向
        for h_i in range(o_shape[1]):
            # 横向
            for w_i in range(o_shape[2]):
                # channel
                for f_i in range(o_shape[3]):
                    out[b_i][h_i][w_i][f_i] = np.max(x[b_i][h_i * stride[1]:h_i * stride[1] + k_shape[1]]
                                                     [w_i * stride[2]: w_i * stride[2] + k_shape[2]][f_i])
    return out


def relu(x):
    # 按照元素计算绝对值
    return np.absolute(x)


def softmax(x):
    """
    对x执行softmax函数，并输出
    :param x:
    :return:
    """
    x_exp = np.exp(x)
    x_shape = np.shape(x)
    x_exp_sum = np.sum(x, len(x) - 1)
    result = np.zeros(np.shape(x))
    for b_i in range(len(x)):
        for i in range(len(x[0])):
            result[b_i][i] = x[b_i][i] / x_exp_sum[b_i]


def loss(y, y_):
    pass


def accuracy(y, y_):
    pass


def derive_conv2d(x, f, stride=[1, 1, 1, 1], padding="SAME"):
    """
    x为输入数据，f为卷积过滤器
    :param x: 4维数据[N, H, W, C]
    :param f: 4维数据[H, W, InputC, OutputC]
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return: dx, df
    """
    x_shape = np.shape(x)
    # x的求导
    dx = np.zeros(x_shape)
    f_shape = np.shape(f)
    # f的求导
    df = np.zeros(f_shape)
    # 要求两个channel一样
    assert x_shape[3] == f_shape[2]

    o_shape, x = pad_algorithm(x, f, stride, padding)

    # 纵向
    for h_i in range(x_shape[0]):
        # 横向
        for w_i in range(x_shape[1]):
            # channel
            for c_i in range(x_shape[2]):
                df[h_i][w_i][c_i] = np.divide(np.sum(x[:][h_i::stride[1]][w_i::stride[2]][:]), len(x))

    # batch
    for b_i in range(o_shape[0]):
        # 纵向
        for h_i in range(o_shape[1]):
            # 横向
            for w_i in range(o_shape[2]):
                sum = np.zeros(np.shape(f))
                # 过滤器
                for f_i in range(o_shape[3]):
                    sum = np.add(sum, f[:][:][:][f_i])
                dx[b_i][h_i:h_i + f_shape[0]][w_i: w_i + f_shape[1]][:] = np.add(dx[b_i][h_i:h_i + f_shape[0]][w_i: w_i + f_shape[1]][:], sum)
    return dx, df


def derive_max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 3, 1], padding="VALID"):
    """
    
    :param x:
    :return:
    """
    x_shape = np.shape(x)
    k_shape = np.shape(ksize)

    # 为了直接使用pad_algorithm，需要对ksize进行维度调整
    o_shape, x = pad_algorithm(x, np.transpose(ksize, [1, 2, 0, 3]), stride, padding)
    o_shape[3] = x_shape[3]
    dx = np.zeros(x_shape)

    # batch
    for b_i in range(o_shape[0]):
        # 纵向
        for h_i in range(o_shape[1]):
            # 横向
            for w_i in range(o_shape[2]):
                # channel
                for f_i in range(o_shape[3]):
                    max_index = np.argmax(x[b_i][h_i * stride[1]:h_i * stride[1] + k_shape[1]]
                                                     [w_i * stride[2]: w_i * stride[2] + k_shape[2]][f_i])
                    dx[b_i][h_i + max_index / k_shape[1]][w_i + max_index % k_shape[2]][f_i] = 1
    return dx


def derive_matmul(x, y):
    """
    np.matmul(x, y)
    :param x:
    :param y:
    :return:
    """
    dx = np.sum(y, 1)
    dy = np.sum(x, 0)
    return dx, dy

def derive_cross(y, y_):
    """
    loss函数的反向求导
    :param y:
    :param y_:
    :return:
    """
    pass


def derive_softmax(a, y):
    """
    softmax的反向求导
    :param a: a = softmax(z)
    :param y: 真实的标签
    :return: 返回dz
    """
    return np.subtract(a, y)


def derive_relu(x):
    """
    y = relu(x)
    :return: dy/dx
    """
    return [1 if i > 0 else 0 for i in x.flat]


def nn(x, y):
    """
    网络结构
    :param x: 输入
    :return:
    """
    BATCH_SIZE = len(x)
    ALPHA = 0.1

    # 第一层卷积
    W_conv1 = np.zeros([5, 5, 1, 32])  # 5*5*1
    b_conv1 = np.zeros([32])  # 32
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = np.reshape(x, [-1, 28, 28, 1])  # 28*28*1
    # x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
    h_conv1_z = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = relu(h_conv1_z)  # 28*28*32
    h_pool1 = max_pool(h_conv1)  # 14*14*32

    # 第二层卷积
    W_conv2 = np.zeros([5, 5, 32, 64])  # 5*5*32
    b_conv2 = np.zeros([64])  # 64
    h_conv2_z = conv2d(h_pool1, W_conv2) + b_conv2  # 14*14*64
    h_conv2 = relu(h_conv2_z)
    h_pool2 = max_pool(h_conv2)  # 7*7*64

    # 全连接层
    W_fc1 = np.zeros([7 * 7 * 64, 1024])
    b_fc1 = np.zeros([1024])
    h_pool2_flat = np.reshape(h_pool2, [-1, 7 * 7 * 64])  # 把h_pool2变成一维7*7*64行
    h_fc1_z = np.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = relu(h_fc1_z)  # 1024

    # 输出层 softmax
    W_fc2 = np.zeros([1024, 10])
    b_fc2 = np.zeros([10])
    # y_conv是结果
    y_conv = softmax(np.matmul(h_fc1, W_fc2) + b_fc2)

    # loss函数
    cross_entropy = loss(y, y_conv)

    # 反向求导更新
    # y_conv=softmax(z)
    dz = derive_softmax(y_conv, y)
    # z = np.matmul(h_fc1, W_fc2) + b_fc2
    db_fc2 = dz
    b_fc2 -= np.multiply(ALPHA, db_fc2)
    dh_fc1, dW_fc2 = np.multiply(derive_matmul(h_fc1, W_fc2), dz)
    dW_fc2 = np.divide(dW_fc2, BATCH_SIZE)
    W_fc2 -= np.multiply(ALPHA, dW_fc2)
    # h_fc1 = relu(h_fc1_z)
    dh_fc1_z = derive_relu(h_fc1_z) * dh_fc1
    # h_fc1_z = np.matmul(h_pool2_flat, W_fc1) + b_fc1
    db_fc1 = dh_fc1_z * dh_fc1_z
    b_fc1 -= db_fc1
    dh_pool2_flat, dW_fc1 = np.multiply(derive_matmul(h_pool2_flat, W_fc1), dh_fc1_z)
    dW_fc1 = np.divide(dW_fc1, BATCH_SIZE)
    W_fc1 -= np.multiply(ALPHA, dW_fc1)
    # h_pool2_flat = np.reshape(h_pool2, [-1, 7 * 7 * 64])
    dh_pool2 = np.reshape(dh_pool2_flat, [-1, 7, 7, 64])
    # h_pool2 = max_pool(h_conv2)
    dh_conv2 = np.multiply(dh_pool2, derive_max_pool(h_conv2))
    # h_conv2 = relu(h_conv2_z)
    dh_conv2_z = np.multiply(dh_conv2, derive_relu(h_conv2_z))
    # h_conv2_z = conv2d(h_pool1, W_conv2) + b_conv2
    dh_pool1, dW_conv2 = derive_conv2d(h_pool1, W_conv2)
    dh_pool1 = np.multiply(dh_pool1, dh_conv2_z)
    dW_conv2 = np.multiply(dW_conv2, dh_conv2_z)
    W_conv2 -= np.multiply(ALPHA, dW_conv2)
    db_conv2 = dh_conv2_z
    b_conv2 -= np.multiply(ALPHA, db_conv2)
    # h_pool1 = max_pool(h_conv1)
    dh_conv1 = np.multiply(dh_pool1, derive_max_pool(h_conv1))
    # h_conv1 = relu(h_conv1_z)
    dh_conv1_z = np.multiply(dh_conv1, derive_relu(h_conv1_z))
    # h_conv1_z = conv2d(x_image, W_conv1) + b_conv1
    _, dW_conv1 = derive_conv2d(x_image, W_conv1)
    dW_conv1 = np.multiply(dW_conv1, dh_conv1_z)
    W_conv2 -= np.multiply(ALPHA, dW_conv1)
    db_conv1 = dh_conv1_z
    b_conv1 -= np.multiply(ALPHA, db_conv1)
    
    
def train():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    for i in range(1000):
        x, y = mnist.train.next_batch(BATCH_SIZE)
        x = np.reshape(x, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        y_conv = nn(x, y)
        cross_entropy = loss(y, y_conv)
        # 怎么把每次的反向就放s


if __name__ == "__main__":
    train()