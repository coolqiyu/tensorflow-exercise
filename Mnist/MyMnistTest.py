from . import Dnn

import unittest
import copy

class TestMat(unittest.TestCase):
    """
    MyMnist中几个类方法的测试
    """
    def listAlmostEqual(self, result, expect):
        for d1 in range(len(expect)):
            for d2 in range(len(expect[0])):
                self.assertAlmostEqual(result[d1][d2], expect[d1][d2], delta=0.09)

    # Mat类的测试
    def test_matmul(self):
        x = [[1, 2, 3], [3, 2, 1]]
        y = [[1, 1], [1, 1], [1, 1]]
        result = Dnn.Mat.matmul(x, y)
        self.assertEqual(result, [[6, 6], [6, 6]])

    def test_matadd(self):
        x = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        y = [[1, 1, 1, 1, 1]]
        result = Dnn.Mat.matadd(x, y)
        self.assertEqual(result, [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])

    def test_vecmax(self):
        x = [0, 1, 2, 3]
        result = Dnn.Mat.vecmax(x)
        self.assertEqual(result, 3)

    def test_vecmax2(self):
        x = [0, 1, 2, 3]
        x.reverse()
        result = Dnn.Mat.vecmax(x)
        self.assertEqual(result, 0)

    def test_vecmax3(self):
        x = [0, 3, 1, 2]
        result = Dnn.Mat.vecmax(x)
        self.assertEqual(result, 1)

    def test_zeros(self):
        result = Dnn.Mat.zeros((1, 2))
        self.assertEqual(result, [[0, 0]])

    # NN类的测试
    def test_softmax(self):
        z = [[5, 2, -1, 3], [5, 2, -1, 3]]
        result = Dnn.NN.softmax(z)
        expect = [[0.842, 0.042, 0.002, 0.114], [0.842, 0.042, 0.002, 0.114]]
        self.listAlmostEqual(result, expect)

    def test_prediction(self):
        y_ = [[0, 1], [1, 0], [0, 1]]
        y = [[0, 1], [0, 1], [1, 0]]
        result = Dnn.NN.prediction(y_, y)
        self.assertAlmostEqual(result, 1/3, delta=0.001)


    def test_derive(self):
        x = [[1, 1], [2, 2], [3, 3]]
        w = [[2, 2, 2, 2], [3, 3, 3, 3]]
        b = [[1, 1, 1, 1]]
        y = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
        alpha = 0.01
        dw = [[0.5 / 3, 0.5, -3.5 / 3, 0.5], [0.5 / 3, 0.5, -3.5 / 3, 0.5]]
        db = [[-0.25 / 3, 0.25, -1.25 / 3, 0.25]]
        new_w = copy.deepcopy(w)
        new_b = copy.deepcopy(b)
        for d1 in range(len(w)):
            for d2 in range(len(w[0])):
                new_w[d1][d2] = new_w[d1][d2] - alpha * dw[d1][d2]
                new_b[0][d2] = new_b[0][d2] - alpha * db[0][d2]

        z = Dnn.Mat.matadd(Dnn.Mat.matmul(x, w), b)
        self.listAlmostEqual(z, [[6, 6, 6, 6], [11, 11, 11, 11], [16, 16, 16, 16]])
        a = Dnn.NN.softmax(z)
        print(a)
        Dnn.NN.derive(a, y, x, alpha, w, b)

        self.listAlmostEqual(w, new_w)
        self.listAlmostEqual(b, new_b)


    # def test_derive(self):
    #     x = [[1, 1], [2, 2], [3, 3]]
    #     w = [[2, 2, 2, 2], [3, 3, 3, 3]]
    #     b = [[1, 1, 1, 1]]
    #     z = MyMnist.Mat.matadd(MyMnist.Mat.matmul(x, w), b)
    #     self.assertEqual(z, [[6, 6, 6, 6], [11, 11, 11, 11], [16, 16, 16, 16]])
    #     a = MyMnist.NN.softmax(z)
    #     y = []
    #     alpha = 0.01
    #     MyMnist.NN.derive(a, y, x, alpha, w, b)

if __name__ == "__main__":
    unittest.main()
