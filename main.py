import Example
import Mnist.TFMnist as TFMnist
import numpy as np
import Mnist.MyMnist as MyMnist

if __name__ == '__main__':
   # Example.Example.Example4()
   #  TFMnist.tensorboard()
    # a = [1,2,3,4,5,6,7,8,9,10,11,12]
    # np.reshape(a, [2,3,2])
    # print(a)
    a = [1, 2]
    b = [[[[[1],[2]],[[1],[2]]]]]
    a=[[1, 2, 3]]
    b=[[1], [2], [1]]
    print(np.matmul(a, b))
    print(MyMnist.Mat.matmul(a, b))
    # print(np.shape(a))
