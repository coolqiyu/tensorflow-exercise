#
# 用numpy实现minist的卷积网络
# =================================
import numpy as np
from common import input_data
import threading

BATCH_SIZE = 50
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNEL = 1
ALPHA = 0.1
W_conv1 = b_conv1 = h_conv1_z = h_conv1 = h_pool1 = 0
W_conv2 = b_conv2 = h_conv2_z = h_conv2 = h_pool2 = 0
W_fc1 = b_fc1 = h_pool2_flat = h_fc1_z = h_fc1 = 0
W_fc2 = b_fc2 = 0


"""
same:28/2 = 14
valid: 28-3/2+1=13.x=13
same: 28/3=9.x=10
valid: 28-3/3+1=9.x=9
same: 15/2=7.x=8
valid: 15-3/2+1=7
same: 15/3=5
valid: 15-3/3+1=5

same: 向上求
valid：向下求
"""
def pad_algorithm(x, f_shape, stride, padding):
    """
    执行padding算法
    :param x: 4维数据[N, H, W, C]
    :param f: 1维数据，表示过滤器的大小[H, W, InputC, OutputC]
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return: 返回pad算法后的数据张量的shape[N, H, W, C]，以及pad之后的x, pad时对x进行padding的大小
    """
    x_shape = np.shape(x)
    #f_shape = np.shape(f)
    pad = np.zeros((4, 2))
    if padding.lower() == "same":
        # h/stride的上界
        o_shape = [x_shape[0], int(np.ceil(x_shape[1]/stride[1])),
                   int(np.ceil(x_shape[2]/stride[2])), f_shape[3]]
        # 对输入进行填充，取计算结果的上界
        # 原来写的不对，same是在stride=1时输入和输出才一样大，其他时候还是要除以stride的。也就是2p=f
        h_pad = f_shape[0] // 2 # int(np.ceil(stride[1] * x_shape[1] - stride[1] + f_shape[0] - x_shape[1]))
        w_pad = f_shape[1] // 2 # int(np.ceil(stride[2] * x_shape[2] - stride[2] + f_shape[1] - x_shape[1]))
        pad = [[0, 0], [h_pad, f_shape[0] - h_pad], [w_pad, f_shape[1] - w_pad], [0, 0]]
        x = np.pad(x, pad, 'constant', constant_values=(0, 0))
    elif padding.lower() == "valid":
        # (h-f/stride)+1的下界
        o_shape = [x_shape[0], int(np.floor((x_shape[1] - f_shape[0]) / stride[1])) + 1,
                   int(np.floor((x_shape[2] - f_shape[1]) / stride[2])) + 1, f_shape[3]]
    else:
        raise Exception("参数padding值{0}不合法，应该为same或valid".format(padding))
    return o_shape, x, pad


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

    o_shape, x, _ = pad_algorithm(x, f_shape, stride, padding)
    out = np.zeros(o_shape)

    # # batch
    # for b_i in range(o_shape[0]):
    #     # 纵向
    #     for h_i in range(o_shape[1]):
    #         # 横向
    #         for w_i in range(o_shape[2]):
    #             #过滤器
    #             for f_i in range(o_shape[3]):
    #                 out[b_i][h_i][w_i][f_i] = np.sum(np.multiply(x[b_i,h_i * stride[1]:h_i * stride[1] + f_shape[0],w_i * stride[2]: w_i * stride[2] + f_shape[1],:], f[:,:,:,f_i]))

    # 一个线程中的计算
    def conv2d_thread(start_batch):
        # batch
        end_batch = o_shape[0] if (start_batch + 1) * thread_batch_size > o_shape[0] else (start_batch + 1) * thread_batch_size
        for b_i in range(start_batch * thread_batch_size, end_batch):
            # 纵向
            for h_i in range(o_shape[1]):
                # 横向
                for w_i in range(o_shape[2]):
                    # 过滤器
                    for f_i in range(o_shape[3]):
                        out[b_i][h_i][w_i][f_i] = np.sum(
                            np.multiply(x[b_i, h_i * stride[1]:h_i * stride[1] + f_shape[0],
                                        w_i * stride[2]: w_i * stride[2] + f_shape[1], :],
                                        f[:, :, :, f_i]))
    # 多线程的方式，每50个在一个线程中
    threads = []
    thread_batch_size = 10
    thread_cnt = int(np.ceil(o_shape[0] / thread_batch_size))
    for i in range(thread_cnt):
        thread = threading.Thread(target=conv2d_thread, args=(i,))
        threads.append(thread)
    for i in range(thread_cnt):
        threads[i].start()
    for i in range(thread_cnt):
        threads[i].join()
    return out


def max_pool(x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding="VALID"):
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
    nksize = np.zeros(4)
    nksize[:] = ksize[1], ksize[2], ksize[0], ksize[3]
    o_shape, x, _ = pad_algorithm(x, nksize, stride, padding)
    o_shape[3] = x_shape[3]
    out = np.zeros(o_shape)

    # batch
    # for b_i in range(o_shape[0]):
    #     # 纵向
    #     for h_i in range(o_shape[1]):
    #         # 横向
    #         for w_i in range(o_shape[2]):
    #             # channel
    #             for f_i in range(o_shape[3]):
    #                 out[b_i][h_i][w_i][f_i] = np.max(x[b_i,h_i * stride[1]:h_i * stride[1] + ksize[1],w_i * stride[2]: w_i * stride[2] + ksize[2],f_i])


    #
    def max_pool_thread(start_batch):
        end_batch = (start_batch + 1) * thread_batch_cnt if (start_batch + 1) * thread_batch_cnt < o_shape[0] else o_shape[0]
        for b_i in range(start_batch * thread_batch_cnt, end_batch):
            # 纵向
            for h_i in range(o_shape[1]):
                # 横向
                for w_i in range(o_shape[2]):
                    # channel
                    for f_i in range(o_shape[3]):
                        out[b_i][h_i][w_i][f_i] = np.max(x[b_i, h_i * stride[1]:h_i * stride[1] + ksize[1],
                                                         w_i * stride[2]: w_i * stride[2] + ksize[2], f_i])
    # 多线程计算方式
    threads = []
    thread_batch_cnt = 10
    thread_cnt = int(np.ceil(o_shape[0] / thread_batch_cnt))
    for i in range(thread_cnt):
        thread = threading.Thread(target=max_pool_thread, args=(i,))
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return out


def relu(x):
    """
    y = relu(x)
    小于0时为0，大于0时y=x
    :param x:
    :return:
    """
    return np.reshape([i if i > 0 else 0 for i in x.flatten()], np.shape(x))


def softmax(x):
    """
    对x执行softmax函数，并输出
    :param x: [batch_size, class_cnt]
    :return:
    """
    x_exp = np.exp(x)
    x_shape = np.shape(x)
    x_exp_sum = np.sum(x_exp, 1)
    result = np.zeros(x_shape)
    for b_i in range(x_shape[0]):
        for i in range(x_shape[1]):
            result[b_i][i] = np.exp(x[b_i,i]) / x_exp_sum[b_i]
    return result


def loss(y, y_):
    # y_shape = np.shape(y)
    # d = 1
    # for i in y_shape:
    #     d *= i
    #cost = np.divide(np.sum(np.multiply(np.multiply(-1, y), np.log2(y_))), d)
    cost = np.sum(np.multiply(np.multiply(-1, y), np.log(y_)))
    return cost

def accuracy(y, y_):
    """
    对y和y_比较，判断是否正确性
    :param y: [size, class_cnt]
    :param y_: [size, class_cnt]
    :return:
    """
    batch_size = np.shape(y)[0]
    # 真实和预测结果相同的个数
    same_cnt = 0
    for i in range(batch_size):
        right_index = np.argmax(y[i])
        predict_index = np.argmax(y_[i])
        if right_index == predict_index:
            same_cnt += 1
    return same_cnt / batch_size


def derive_conv2d(dy, x, f, stride=[1, 1, 1, 1], padding="SAME"):
    """
    y = conv2d(x)
    x为输入数据，f为卷积过滤器
    :param x: 4维数据[N, H, W, C]
    :param f: 4维数据[H, W, InputC, OutputC]
    :param stride: [N, H, W, C]  N和C设置为1
    :param padding: same或valid
    :return: dx, df
    """
    f_shape = np.shape(f)
    # f的求导
    df = np.zeros(f_shape)
    # 要求两个channel一样

    o_shape, x, pad = pad_algorithm(x, f_shape, stride, padding)
    x_shape = np.shape(x)
    dx = np.zeros(x_shape)
    # 对f进行求导
    # 纵向
    for h_i in range(f_shape[0]):
        # 横向
        for w_i in range(f_shape[1]):
            # filter
            for f_i in range(f_shape[3]):
                # 每次访问下一个x的位置是正确的(+stride)，但是没有考虑到边界，最大的不总是最后一个，filter的占位
                df[h_i][w_i][:, f_i] = np.divide(np.sum(np.multiply(x[:, h_i:x_shape[1] - f_shape[0] + h_i: stride[1],
                                                                    w_i:x_shape[2] - f_shape[1] + w_i:stride[2], :], dy[:, :, :, f_i : f_i+1]), (0, 1, 2)), x_shape[0])
    # 对x进行求导
    # 如果用x原始大小来计算：则第一次计算中f不一定从0开始，所以这里用pad过的x大小计算，最后返回时进行切片
    # batch
    for b_i in range(x_shape[0]):
        # 纵向
        for h_i in range(0, x_shape[1] - f_shape[0], stride[1]):
            # 横向
            for w_i in range(0, x_shape[2] - f_shape[1], stride[2]):
                dx[b_i][h_i:h_i + f_shape[0],w_i: w_i + f_shape[1],:] = np.add(dx[b_i][h_i:h_i + f_shape[0], w_i: w_i + f_shape[1], :], np.sum(np.multiply(f[:,:,:,:], dy[b_i][h_i][w_i]),3))
    return dx[:, pad[1][0]: x_shape[1] - pad[1][1], pad[2][0]: x_shape[2] - pad[2][1], :], df


def derive_max_pool(dy, x, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding="VALID"):
    """
    y = max_pool(x)
    max pool中没有要学习的参数，所以只要吧误差传递到上一层即可，没有梯度计算。
    下一层的误差项的值会原封不动的传递到上一层对应区块中的最大值所对应的神经元，而其他神经元的误差项的值都是0
    :param x:
    :return:
    """
    x_shape = np.shape(x)

    # 为了直接使用pad_algorithm，需要对ksize进行维度调整
    nksize = np.zeros(4)
    nksize[:] = ksize[1], ksize[2], ksize[0], ksize[3]
    o_shape, x, _ = pad_algorithm(x, nksize, stride, padding)
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
                    # 在一定区域内的计算，起始位置不总是h_i, w_i
                    start_h = h_i * stride[1]
                    start_w = w_i * stride[2]
                    max_index = np.argmax(x[b_i, start_h: start_h + ksize[1],
                                          start_w: start_w + ksize[2],f_i])
                    dx[b_i][start_h + max_index // ksize[1]][start_w + max_index % ksize[2]][f_i] = dy[b_i][h_i][w_i][f_i]
    return dx


def derive_matmul(x, y, dz):
    """
    z = np.matmul(x, y)
    反向求导时，dx = dz/dx * dL/dz 要注意不是直接元素乘也不是矩阵乘，应该看不同的y对应不同的dz
    :param x:
    :param y:
    :return:
    """
    dx = []
    x_len = len(x)
    for i in range(x_len):
        dx.append(np.sum(np.multiply(y, dz[i]), 1))
    dy = np.zeros(np.shape(y))

    for i in range(len(y[0])):
        dy[:, i] = np.sum(np.multiply(x, np.reshape(dz[:, i], [-1, 1])), 0)
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
    return np.reshape([1 if i > 0 else 0 for i in x.flat], np.shape(x))

def init_variable():
    """
    对参数进行初始化
    :return:
    """
    global W_conv1, b_conv1
    W_conv1 = np.zeros((5, 5, 1, 32))  # 5*5*1
    b_conv1 = np.zeros((32))  # 32
    global W_conv2, b_conv2
    W_conv2 = np.zeros((5, 5, 32, 64))  # 5*5*32
    b_conv2 = np.zeros((64))  # 64
    global W_fc1, b_fc1
    W_fc1 = np.zeros((7 * 7 * 64, 1024))
    b_fc1 = np.zeros((1024))
    global W_fc2, b_fc2
    W_fc2 = np.zeros((1024, 10))
    b_fc2 = np.zeros((10))

def back_prop(x_image, y, y_conv):
    """
     反向求导更新
     :param: x_image 输入
     :param: y 真实的输出label
     :param: y_conv 预测的结果
    :return:
    """
    global W_conv1, b_conv1, h_conv1_z, h_conv1, h_pool1
    global W_conv2, b_conv2, h_conv2_z, h_conv2, h_pool2
    global W_fc1, b_fc1, h_pool2_flat, h_fc1_z, h_fc1
    global W_fc2, b_fc2
    # y_conv=softmax(z)
    dz = derive_softmax(y_conv, y)
    assert np.shape(dz) == np.shape(y)

    # z = np.matmul(h_fc1, W_fc2) + b_fc2
    db_fc2 = np.divide(np.sum(dz, 0), np.shape(dz)[0])
    assert np.shape(db_fc2) == np.shape(b_fc2)
    b_fc2 = np.subtract(b_fc2, np.multiply(ALPHA, db_fc2))
    dh_fc1, dW_fc2 = derive_matmul(h_fc1, W_fc2, dz)
    assert np.shape(dW_fc2) == np.shape(W_fc2)
    assert np.shape(dh_fc1) == np.shape(h_fc1)
    W_fc2 = np.subtract(W_fc2, np.multiply(ALPHA, dW_fc2))

    # h_fc1 = relu(h_fc1_z)
    dh_fc1_z = derive_relu(h_fc1_z) * dh_fc1
    assert np.shape(h_fc1_z) == np.shape(dh_fc1_z)

    # h_fc1_z = np.matmul(h_pool2_flat, W_fc1) + b_fc1
    db_fc1 = np.divide(np.sum(dh_fc1_z, 0), np.shape(dh_fc1_z)[0])
    assert np.shape(db_fc1) == np.shape(b_fc1)
    b_fc1 = np.subtract(b_fc1, np.multiply(ALPHA, db_fc1))
    dh_pool2_flat, dW_fc1 = derive_matmul(h_pool2_flat, W_fc1, dh_fc1_z)
    W_fc1 = np.subtract(W_fc1, np.multiply(ALPHA, dW_fc1))
    assert np.shape(dW_fc1) == np.shape(W_fc1)
    assert np.shape(dh_pool2_flat) == np.shape(h_pool2_flat)
    # h_pool2_flat = np.reshape(h_pool2, [-1, 7 * 7 * 64])
    dh_pool2 = np.reshape(dh_pool2_flat, [-1, 7, 7, 64])
    assert np.shape(dh_pool2) == np.shape(h_pool2)
    # h_pool2 = max_pool(h_conv2)
    dh_conv2 = derive_max_pool(dh_pool2, h_conv2)
    assert np.shape(dh_conv2) == np.shape(h_conv2)
    # h_conv2 = relu(h_conv2_z)
    dh_conv2_z = np.multiply(dh_conv2, derive_relu(h_conv2_z))
    assert np.shape(dh_conv2_z) == np.shape(h_conv2_z)
    # h_conv2_z = conv2d(h_pool1, W_conv2) + b_conv2
    dh_pool1, dW_conv2 = derive_conv2d(dh_conv2_z, h_pool1, W_conv2)
    assert np.shape(dh_pool1) == np.shape(h_pool1)
    assert np.shape(dW_conv2) == np.shape(W_conv2)
    W_conv2 = np.subtract(W_conv2, np.multiply(ALPHA, dW_conv2))
    dh_conv2_z_shape = np.shape(dh_conv2_z)
    db_conv2 = np.divide(np.sum(dh_conv2_z, (0, 1, 2)), dh_conv2_z_shape[0] * dh_conv2_z_shape[1] * dh_conv2_z_shape[2])
    assert np.shape(db_conv2) == np.shape(b_conv2)
    b_conv2 = np.subtract(b_conv2, np.multiply(ALPHA, db_conv2))

    # h_pool1 = max_pool(h_conv1)
    dh_conv1 = derive_max_pool(dh_pool1, h_conv1)
    assert np.shape(dh_conv1) == np.shape(h_conv1)
    # h_conv1 = relu(h_conv1_z)
    dh_conv1_z = np.multiply(dh_conv1, derive_relu(h_conv1_z))
    assert np.shape(dh_conv1_z) == np.shape(h_conv1_z)
    # h_conv1_z = conv2d(x_image, W_conv1) + b_conv1
    _, dW_conv1 = derive_conv2d(dh_conv1_z, x_image, W_conv1)
    W_conv1 = np.subtract(W_conv1, np.multiply(ALPHA, dW_conv1))
    dh_conv1_z_shape = np.shape(dh_conv1_z)
    db_conv1 = np.divide(np.sum(dh_conv1_z, (0, 1, 2)), dh_conv1_z_shape[0] * dh_conv1_z_shape[1] * dh_conv1_z_shape[3])
    b_conv1 = np.subtract(b_conv1, np.multiply(ALPHA, db_conv1))
    assert np.shape(W_conv1) == np.shape(dW_conv1)
    assert np.shape(db_conv1) == np.shape(b_conv1)

def forward_prop(x, y):
    """
    前向计算
    :param x: 输入
    :param y: 输出的label
    :return:
    """
    # 第一层卷积
    global W_conv1, b_conv1, h_conv1_z, h_conv1, h_pool1
    # x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = np.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])  # 28*28*1
    # x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling
    h_conv1_z = conv2d(x_image, W_conv1) + b_conv1
    h_conv1 = relu(h_conv1_z)  # 28*28*32
    h_pool1 = max_pool(h_conv1)  # 14*14*32

    # 第二层卷积
    global W_conv2, b_conv2, h_conv2_z, h_conv2, h_pool2
    h_conv2_z = conv2d(h_pool1, W_conv2) + b_conv2  # 14*14*64
    h_conv2 = relu(h_conv2_z)
    h_pool2 = max_pool(h_conv2)  # 7*7*64

    # 全连接层，转换成全连接层时，只是把一个batch中的展开
    global W_fc1, b_fc1, h_pool2_flat, h_fc1_z, h_fc1
    h_pool2_flat = np.reshape(h_pool2, [-1, 7 * 7 * 64])  # 把h_pool2变成一维7*7*64行
    h_fc1_z = np.matmul(h_pool2_flat, W_fc1) + b_fc1
    h_fc1 = relu(h_fc1_z)  # 1024

    # 输出层 softmax
    global W_fc2, b_fc2
    # y_conv是结果
    y_conv = softmax(np.matmul(h_fc1, W_fc2) + b_fc2)
    return x_image, y_conv
    
def train():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    for i in range(1000):
        x, y = mnist.train.next_batch(BATCH_SIZE)
        x = np.reshape(x, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
        x_image, y_conv = forward_prop(x, y)
        back_prop(x_image, y, y_conv)
        cross_entropy = loss(y, y_conv)
        print("交叉熵：{0}, {1}".format(i, cross_entropy))

def test():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    x, y = mnist.test.images, mnist.test.labels
    _, y_conv = forward_prop(x, y)
    print("正确率: {0}".format(accuracy(y, y_conv)))

if __name__ == "__main__":
    init_variable()
    train()
    test()