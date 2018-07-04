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
        dy = [4, 8, 8, 8, 16, 16, 8, 16, 16]
        dx, df = MyCnn.derive_conv2d(dy, x, f)
        dx = np.array(dx).flatten()
        df = np.array(df).flatten()
        dx_expect = [72, 96, 48, 96, 128, 64, 48, 64, 32]
        df_expect = [128, 160, 128, 168]
        self.assertTrue((dx==dx_expect).all())
        self.assertTrue((df == df_expect).all())

    def test_maxpool_01(self):
        """
        batch_size = 1， 1个过滤器的情况
        :return:
        """
        x = np.full((1, 3, 3, 1), 2)
        result = np.array(MyCnn.max_pool(x)).flatten()
        print(type(result))
        expect = [2]
        print(type(expect))
        self.assertTrue((expect==result).all())

    def test_maxpool_02(self):
        """
        batch_size = 2
        :return:
        """
        x = np.full((1, 3, 3, 3), 1)
       # print(x)
        x = np.concatenate((x, np.full((1, 3, 3, 3), 2)), 0)
        print(x)
        result = np.array(MyCnn.max_pool(x)).flatten()
        print(result)
        expect = [1, 1, 1, 2, 2, 2]
        self.assertTrue((expect == result).all())

if __name__ == "__main__":
    unittest.main()