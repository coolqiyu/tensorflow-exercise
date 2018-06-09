import Example
import Mnist.TFMnist as TFMnist
import numpy as np
import Mnist.MyMnist as MyMnist
from sympy import *
if __name__ == '__main__':
   x = Symbol("x")
   print(diff(x, x))
   print(diff(x**2, x))
