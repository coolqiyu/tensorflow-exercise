import unittest
import MyCnn
import numpy as np

class MyCnnTest(unittest.TestCase):
    def test_conv2d_01(self):
        """
        batch_size = 1， 1个过滤器的情况
        :return:
        """
        x = np.full((1, 3, 3, 1), 2)
        f = np.full((2, 2, 1, 1), 2)
        # np.array(object).flatten() 把object变成数组并且flatten
        result = np.array(MyCnn.conv2d(x, f)).flatten()
        expect = [4, 8, 8, 8, 16, 16, 8, 16, 16]
        # expect==result返回的是每个位置对应的比较结果 .all() 的话如果都是true，则返回true
        self.assertTrue((expect==result).all())

    def test_conv2d_02(self):
        """
        batch_size = 2， 4个过滤器的情况
        :return:
        """
        x = np.full((2, 3, 3, 3), 2)
        f = np.full((2, 2, 3, 4), 2)
        result = np.array(MyCnn.conv2d(x, f)).flatten()
        expect = [12, 24, 24, 24, 48, 48, 24, 48, 48]
        expect = np.repeat(np.reshape(expect, (-1, 1)), 4, 1).flatten()
        expect = np.repeat(np.reshape(expect, (-1, len(expect))),2,0).flatten()
        # 因为expect的大小一开始没有设置对，导致expect==result直接就是false，而不是各个位置比较结果的数组
        # 并不是类型的问题
        self.assertTrue((expect == result).all())


    def test_derive_conv2d_01(self):
        """
        batch_size = 1， 1个过滤器的情况
        :return:
        """
        x = np.full((1, 3, 3, 1), 2)
        f = np.full((2, 2, 1, 1), 2)
        dy = np.reshape([4, 8, 8, 8, 16, 16, 8, 16, 16], (1, 3, 3, 1))
        dx, df = MyCnn.derive_conv2d(dy, x, f)
        dx = np.array(dx).flatten()
        df = np.array(df).flatten()
        dx_expect = [72, 96, 48, 96, 128, 64, 48, 64, 32]
        df_expect = [128, 160, 160, 200]
        self.assertTrue((dx==dx_expect).all())
        print(df)
        self.assertTrue((df == df_expect).all())

    def test_derive_conv2d_02(self):
        """
        batch_size = 2， 4个过滤器的情况
        :return:
        """
        x = np.full((2, 3, 3, 3), 2)
        f = np.full((2, 2, 3, 4), 2)
        dy = np.full((2, 3, 3, 4), 1)

        dx, df = MyCnn.derive_conv2d(dy, x, f)
        dx = np.array(dx).flatten()
        df = np.array(df).flatten()
        dx_expect = np.repeat(np.repeat(np.reshape([32, 32, 16, 32, 32, 16, 16, 16, 8], (1, 3, 3, 1)), 3, 3), 2, 0).flatten()
        df_expect = np.repeat(np.repeat(np.reshape([8, 12, 12, 18], (2, 2, 1, 1)), 3, 2), 4, 3).flatten()
        self.assertTrue((dx == dx_expect).all())
        self.assertTrue((df == df_expect).all())

    def test_maxpool_01(self):
        """
        batch_size = 1
        :return:
        """
        x = np.full((1, 3, 3, 1), 2)
        result = np.array(MyCnn.max_pool(x)).flatten()
        expect = [2]
        self.assertTrue((expect==result).all())

    def test_maxpool_02(self):
        """
        batch_size = 2
        :return:
        """
        x = np.full((1, 4, 4, 3), 1)
        x = np.concatenate((x, np.full((1, 4, 4, 3), 2)), 0)
        result = np.array(MyCnn.max_pool(x)).flatten()
        expect = np.concatenate((np.repeat(np.full((1, 2, 2, 1), 1), 3, 3), np.repeat(np.full((1, 2, 2, 1), 2), 3, 3)), 0).flatten()
        self.assertTrue((expect == result).all())

    def test_derive_maxpool_01(self):
        """
        batch_size = 1
        :return:
        """
        x = np.full((1, 3, 3, 1), 2)
        dy = np.reshape([2], (1, 1, 1, 1))
        dx = np.array(MyCnn.derive_max_pool(dy, x)).flatten()
        dx_expect = [2, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertTrue((dx == dx_expect).all())

    def test_derive_maxpool_02(self):
        """
        batch_size = 2
        :return:
        """
        x = np.full((1, 4, 4, 3), 1)
        x = np.concatenate((x, np.full((1, 4, 4, 3), 2)), 0)
        dy = np.full((2, 2, 2, 3), 1)
        dx = np.array(MyCnn.derive_max_pool(dy, x)).flatten()
        dx_expect = np.repeat(np.repeat(np.reshape([1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], (1, 4, 4, 1)), 3, 3), 2, 0).flatten()
        self.assertTrue((dx == dx_expect).all())

    def test_softmax(self):
        x = np.full((2, 5), 0)
        result = MyCnn.softmax(x)
        expect = np.full((2, 5), 0.2)
        self.assertTrue((result==expect).all())

    def test_derive_softmax(self):
        pass

    def test_relu(self):
        x = np.reshape([-1, 2, 3, 4, -5, -6, 7, 8, 9, -10], (2, 5))
        expect = [0, 2, 3, 4, 0, 0, 7, 8, 9, 0]
        result = np.array(MyCnn.relu(x)).flatten()
        self.assertTrue((expect == result).all())

    def test_derive_relu(self):
        x = np.reshape([-1, 2, 3, 4, -5, -6, 7, 8, 9, -10], (2, 5))
        result = np.array(MyCnn.derive_relu(x)).flatten()
        expect = [0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
        self.assertTrue((expect == result).all())

    def test_derive_matmul(self):
        x = np.concatenate((np.full((1, 5), 1), np.full((1, 5), 2)), 0)
        y = np.concatenate((np.full((5, 1), 1), np.full((5, 1), 2), np.full((5, 1), 1)), 1)
        dz = np.concatenate((np.full((1, 3), 0), np.full((1, 3), 1)), 0)
        dx_expect = [0, 0, 0, 0, 0, 4, 4, 4, 4, 4]
        dy_expect = np.full((5, 3), 2).flatten()
        dx, dy = MyCnn.derive_matmul(x, y, dz)
        dx = np.array(dx).flatten()
        dy = np.array(dy).flatten()
        self.assertTrue((dx == dx_expect).all())
        self.assertTrue((dy == dy_expect).all())

    def test_loss_01(self):
        cost = MyCnn.loss([[2]], [[2]])
        expect = [-2]
        self.assertTrue((cost == expect).all())

    def test_loss_02(self):
        y = np.full((2, 2), 2)
        y_ = np.full((2, 2), 2)
        cost = MyCnn.loss(y, y_)
        expect = [-2, -2]
        self.assertTrue((cost == expect).all())

if __name__ == "__main__":
    unittest.main()