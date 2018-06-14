##################################
# 这个文件使用纯Python写一个网络，不借助tf或np
##################################
import math
from . import input_data

def my_mnist():
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

    it_cnt = 100 # 执行100次迭代
    alpha = 0.1 # 学习率
    for it in range(it_cnt):
        # 读出来的数据 x[batchsize, 784]    y[batchsize, 10]
        x, y = mnist.train.next_batch(1)
        # 前向  z:[batchsize, 10]
        z = Mat.matadd(Mat.matmul(x, w), b)
        y_ = NN.softmax(z)
        # 反向求导，更新参数
        NN.derive(y_, y, x, alpha, w, b)
        # loss函数
        loss = NN.loss(y_, y)
        print(loss)


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
            for z_i in z_:
                e_z = math.exp(z_i)
                sum += e_z
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
                sum = sum - y_i * math.log(y_[batch_index][i])
        return sum/batch_size

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
        dz = []
        class_cnt = len(a)
        feature_cnt = len(x)
        for i in range(class_cnt):
            #dz = a - y
            dz.append(a[i] - y[i])
            # db = (a- y)
            # db.append(dz[i] *1)
            b[i] = b[i] - alpha * dz[i]
        dw = []
        db = []
        batch_size = 1
        for batch_index in batch_size:
            for i in range(feature_cnt):
                dw.append([])
                for j in range(class_cnt):
                    # dw = (a - y) * x
                    # dw[i].append(x[i] * dz[j])
                    w[i][j] = w[i][j] - alpha * x[batch_index][i] * dz[j]


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
        矩阵加法
        :param x:
        :param y:
        :return: z = x + y
        """
        x_shape = Shape.get_shape(x)
        y_shape = Shape.get_shape(y)
        shape_len = len(x_shape)
        # ==会比较里面的数值  is可以比较是否指向同一个对象
        if x_shape != y_shape:
            raise Exception("输入的形状不相同: x {x} y {y}".format(x=x_shape, y=y_shape))

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

        add_(x, y, 0)
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
        except TypeError:
            print('x: {x} 没有len()'.format(x=x))
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