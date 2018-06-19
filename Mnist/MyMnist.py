##################################
# 这个文件使用纯Python写一个网络，不借助tf或np
##################################
import math
from . import input_data

MAX_NUM = 1.79e+307
MIN_NUM = -1.79e+307
def my_mnist_train():
    # 读取数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 图像的大小
    width = 28
    height = 28
    channel = 1
    pic_size = width * height * channel
    batch_size = 100

    # 初始化变量
    # x = Shape.reshape(x, (1, pic_size))
    w = Mat.zeros((pic_size, 10))
    b = Mat.zeros((1, 10))

    it_cnt = 1000 # 执行100次迭代
    alpha = 0.01 # 学习率

    for it in range(it_cnt):
        # 读出来的数据 x[batchsize, 784]    y[batchsize, 10]
        x, y = mnist.train.next_batch(batch_size)
        # 前向  z:[batchsize, 10]
        z = Mat.matadd(Mat.matmul(x, w), b)
        y_ = NN.softmax(z)
        # 反向求导，更新参数
        NN.derive(y_, y, x, alpha, w, b)
        # loss函数
        loss = NN.loss(y_, y)
        print(str(it) + "   "  + str(loss))
    print("W: {0}".format(w))
    print("b: {0}".format(b))
    my_mnist_test(w, b)


def my_mnist_test(w, b):
    # 读取测试数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = mnist.test.images
    y = mnist.test.labels

    z = Mat.matadd(Mat.matmul(x, w), b)
    y_ = NN.softmax(z)
    accuracy = NN.prediction(y_, y)
    print("accuracy:  {0}".format(accuracy))


class NN:
    """
    和网络相关的操作
    """
    @staticmethod
    def softmax(z):
        """
        对z执行softmax函数
        :param z:
        :return: softmax(z)
        """
        results = []

        for z_ in z:
            sum = 0
            result = []
            # 对一个样本的处理
            for z_i in z_:
                try:
                    e_z = math.exp(z_i)
                    sum += e_z
                except OverflowError:
                    sum = MAX_NUM
                    e_z = MAX_NUM
                result.append(e_z)

            for i, z_i in enumerate(z_):
                result[i] = (float)(result[i])/float(sum)
            results.append(result)

        return results


    @staticmethod
    def loss(y_, y):
        """
        y和y_的loss函数 cost = sum y log y_
        :param y_: 训练的结果
        :param y: 真实的label
        :return:
        """
        batch_size = len(y_)
        sum = 0

        for batch_index in range(batch_size):
            for i, y_i in enumerate(y[batch_index]):
                if y_i == 1.0:
                    if y_[batch_index][i] == 0:
                        sum = MAX_NUM
                    else:
                        sum += -1 * math.log(y_[batch_index][i])
        return sum/batch_size


    @staticmethod
    def prediction(y_, y):
        """
        以y_和y中每个的最大值作为评估结果，相同的个数/总数就是正确率
        :param y_: 模型计算的结果
        :param y: 标签结果
        :return:
        """
        size = len(y_)
        same = 0
        for i, y_i in enumerate(y_):
            ymax = Mat.vecmax(y[i])
            y_max = Mat.vecmax(y_i)
            if ymax == y_max:
                same += 1
        return same / size

    @staticmethod
    def derive(a, y, x, alpha, w, b):
        """
        反向求导，更新w和b
        :param:a 最后输出的向量 m样本数，n个分类 [n, m]
        :param:y 输入的Label向量 m样本数，n个分类 [n, m] n个数中只有一个是1， 其他为0
        :param:x 输入的样本值 k为一个样本的特征值数量 [k, m]
        :param:alpha 学习率
        :return:
        """
        batch_size = len(a)
        class_cnt = len(a[0])
        feature_cnt = len(x[0])
        dz = Mat.zeros((batch_size, class_cnt))
        db = Mat.zeros((1, class_cnt))
        dw = Mat.zeros((feature_cnt, class_cnt))

        for i in range(class_cnt):
            for batch_index in range(batch_size):
                #dz = a - y
                dz[batch_index][i] = a[batch_index][i] - y[batch_index][i]
                # db = (a- y)
                # db.append(dz[i] *1)
                db[0][i] += dz[batch_index][i]
            db[0][i] = db[0][i]/batch_size
            b[0][i] = b[0][i] - alpha * db[0][i]


        for i in range(feature_cnt):
            for j in range(class_cnt):
                for batch_index in range(batch_size):
                # dw = (a - y) * x
                # dw[i].append(x[i] * dz[j])
                    dw[i][j] += x[batch_index][i] * dz[batch_index][j]
                dw[i][j] = dw[i][j] / batch_size
                w[i][j] = w[i][j] - alpha * dw[i][j]


    @staticmethod
    def optimizer():
        """
        优化函数
        :return:
        """
        pass


class Mat:
    """
    矩阵类的操作
    """
    @staticmethod
    def matmul(x, y):
        """
        矩阵乘法
        :param x: n x m_x
        :param y: m_y x p
        :return: z = x * y
        """
        x_shape = Shape.get_shape(x)
        y_shape = Shape.get_shape(y)

        if len(x_shape) == 1:
            n = 1
            x = [x]
        else:
            n = x_shape[0]
        m_x = x_shape[0] if len(x_shape) == 1 else x_shape[1]
        if len(y_shape) == 1:
            m_y = 1
            y = [y]
        else:
            m_y = y_shape[0]
        p = y_shape[0] if len(y_shape) == 1 else y_shape[1]

        if m_x != m_y:
            return

        result = []
        for row_index in range(n):
            for column_index in range(p):
                sum = 0
                for item_index in range(m_x):
                    sum += x[row_index][item_index] * y[item_index][column_index]
                result.append(sum)
        return Shape.reshape(result, (n, p))

    @staticmethod
    def matadd(x, y):
        """
        矩阵加法，增加广播的功能
        :param x:
        :param y:
        :return: z = x + y
        """
        x_shape = Shape.get_shape(x)
        y_shape = Shape.get_shape(y)
        new_y = y[:]
        for i in range(x_shape[0] - y_shape[0]):
            new_y.append(y[0])

        shape_len = len(x_shape)
        # # ==会比较里面的数值  is可以比较是否指向同一个对象
        # if x_shape != y_shape:
        #     raise Exception("输入的形状不相同: x {x} y {y}".format(x=x_shape, y=y_shape))

        def add_(x, y, index_dim):
            """

            :param index_dim: shape中的第index_dim维
            :return:
            """
            if index_dim == shape_len - 1:
                for i in range(x_shape[index_dim]):
                    x[i] += y[i]
            else:
                for i in range(x_shape[index_dim]):
                    add_(x[i], y[i], index_dim + 1)

        add_(x, new_y, 0)
        return x

    @staticmethod
    def zeros(shape):
        """
        生成shape形的值都为0
        :param shape:
        :return:
        """
        dim = 1
        for d in shape:
            dim *= d

        result = []
        for i in range(dim):
            result.append(0)
        return Shape.reshape(result, shape)

    @staticmethod
    def vecmax(vec):
        """
        判断vec中哪个位置的数字最大，并返回
        :param vec:
        :return:
        """
        max_index = 0
        for i, data in enumerate(vec):
            if data > vec[max_index]:
                max_index = i
        return max_index

class Shape:
    """
    形状相关
    """
    @staticmethod
    def get_shape(x):
        """
        获取向量x的形状
        :param x:
        :return: 返回形状的序列(1,2,3)
        """
        try:
            size = len(x)
        except TypeError as ex:
            #print('x: {x} 没有len()'.format(x=x))
            return ()
        else:
            return (size,) + Shape.get_shape(x[0])


    @staticmethod
    def flattern(x):
        """
        把矩阵转换成nx1的向量
        :param x:
        :return:
        """
        if not isinstance(x, list):
            return list(x)
        if isinstance(x[0], list):
            data = []
            for i in x:
                data += i
            return Shape.flattern(data)
        else:
            return x

    @staticmethod
    def reshape(x, new_shape=-1):
        """
        把x重新填充成new_shape形状  -1表示转成nx1的
        :param x:
        :param new_shape:
        :return:
        """
        flattern_x = Shape.flattern(x)

        if new_shape == -1 or len(new_shape) == 1:
            return flattern_x

        # 判断两个形状的数值个数是否相同。如果不同，则返回输入值
        dim = 1
        for d in new_shape:
            dim *= d
        if dim != len(flattern_x):
            return x

        # 做形状变换
        def reshape_(data, dim, size):
            """
            把data整理一个list，包含dim个size大小的元素
            :param data: 数据
            :param dim: 维数
            :param size: 一维的大小
            :return:
            """
            result = []
            for i in range(dim):
                result.append(data[i * size:(i + 1) * size])
            return result

        new_shape = list(new_shape)
        new_shape.reverse()
        data = flattern_x
        size = new_shape[0]
        for i in new_shape[1:]:
            data = reshape_(data, i, size)
            size = i

        return data