##################################
# 这个文件使用纯Python写一个网络，不借助tf或np
##################################

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
        size = len(x) if isinstance(x, list) else 0
        if size == 0:
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
        pass

    @staticmethod
    def loss(y_, y):
        """
        y和y_的loss函数
        :param y_: 训练的结果
        :param y: 真实的label
        :return:
        """
        pass

    @staticmethod
    def optimizer():
        """
        优化函数
        :return:
        """
        pass