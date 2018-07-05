import Example
import Mnist.TFMnist as TFMnist
import numpy as np
import Mnist.MyMnist as MyMnist
from sympy import *

if __name__ == '__main__':
   #MyMnist.my_mnist_train()
   #TFMnist.four_layers_mnist_train()
   # a = "aa"
   # if type(a) == str:
   #    print("dslfdj")
   a = np.multiply([1,2], [2, 4])
   m = 0

   print(m)
   print(np.sum([[0, 0], [1, 1]], 1))
   print(np.divide([2, 2], 2))
   print(np.multiply([1, 2],2))
   print(2 *[1, 2])
   a = [[[1, 4, 3]], [[1, 2, 3]]]
   print(np.argmax(a[0][0:2][0:2]))
   a, b = 1, 2
   print((a,b) * 2)
   print(int(np.ceil(1)))
   a,b = np.multiply((1, 2), 2)
   print(a)
   print(b)
   a = [[1, 2, 3], [2, 3, 4]]

   print(np.transpose(a, (1, 0)))
   print([1, 2,3] *2)
   expect = [12, 24, 24, 24, 48, 48, 24, 48, 48]
   print(np.repeat(np.reshape(expect, (-1, 1)), 4, 1).flatten())
