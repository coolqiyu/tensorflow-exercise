# https://www.machinelearningplus.com/python/101-numpy-exercises-python/上的学习

import numpy as np

# 2.创建在一定范围内的一维数据
# arange(start=None, stop=None, step=None, dtype=None)
# [10 12 14 16 18]
print(np.arange(10, 20, 2))

# 3.创建bool矩阵
# full(shape, fill_value, dtype=None, order='C'):
print(np.full((3, 3), True))
# full_like: return a full array with the same shape and type as a given array.
print(np.full_like([[1, 2, 3], [1, 2, 3]], 1))

# 4.从一维数组中找出满足条件的数据
# 找出是奇数的, arry[]中不仅可以写下标也可以写条件
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# [5]
print(arr[arr - 1 == 4])

# 5.把数组中满足条件的数据替换
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
arr[arr % 2 == 1] = -1
# [-1  2 -1  4 -1  6 -1  8]
print(arr)
arr = [1, 2, 3, 4]
arr[0] = 4
# [4, 2, 3, 4]
print(arr)

# 6.对数组中满足条件的数据进行修改，同时保持原数组
# where(condition, x=None, y=None)
# 条件判断：When True, yield `x`, otherwise yield `y`.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
out = np.where(arr % 2 == 1, -1, arr)
# [-1  2 -1  4 -1  6 -1  8]
print(arr)
# [1, 2, 3, 4, 5, 6, 7, 8]
print(out)

# 7.把一维数组变成两行两列
arr = np.arange(4)
arr = np.reshape(arr, (2, 2))
print(arr)

# 8.把两个数组在垂直方向堆积
# concatenate(a_tuple, axis=0, out=None)
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
print(np.concatenate((a, b), 0))
print(np.vstack((a, b)))
print(np.r_[a, b])

# 9.把两个数组在水平方向堆积
# concatenate(a_tuple, axis=0, out=None)
a = np.arange(10).reshape(2,-1)
b = np.repeat(1, 10).reshape(2,-1)
print(np.concatenate((a, b), 1))
print(np.hstack((a, b)))
print(np.c_[a, b])

# 10.根据输入生成特定pattern的数组
# desire: array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
# tile(A, reps)：针对数组
#       根据reps来对整个数组A进行repeat 结果取reps和A中维度最大的，两者中维度较小的会自动进行扩展。如(3)变成(1,3)
# repeat(a, repeats, axis=None)：针对元素
#       对一个数组的元素进行重复。默认是会把a展开成一维，返回也是一维。默认对所有元素
#       repeats 每个元素表示axis维度上各维的repeat次数
arr = np.array([1, 2, 3])
np.r_[np.repeat(arr, 3), np.tile(arr, 3)]

# 11.获取两个数组中公共的数据
# intersect1d找到两个一维数组的相同值，且返回排序后的unique结果
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
np.intersect1d(a, b)
print(a)

# 12.从a中移除b中也存在的数据
# setdiff1d
a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 6, 7, 8, 9])
np.setdiff1d(a, b)
print(a)

# 13.找出两个数组中值相匹配的位置
# where: 如果只有condition，返回condition.nonzero()元组, 表明在哪一次比较时条件结果为true（按元素比较）
a = np.array([1, 2, 3, 2, 3, 4, 3, 4, 5, 6])
b = np.array([7, 2, 10, 2, 7, 4, 9, 4, 9, 8])
print(np.where(a == b))

# 14.从一个数组中找出在一定范围内的结果
# &优先级比较高，需要把>的部分()
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a >= 5) & (a <= 10))
# [ 6  9 10]
print(a[index])
# [ 6  9 10]
print(a[(a >= 5) & (a <= 10)])

# 15.对函数进行改写，让原来处理标量的改成可以处理数组
# np.vectorize(func)
def maxx(x, y):
   """Get the maximum of two items"""
   if x >= y:
       return x
   else:
       return y
maxx(1, 5)
vec_maxx = np.vectorize(maxx)
a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])
print(vec_maxx(a, b))

# 16.交换二维数组的两列
arr = np.arange(9).reshape(3,3)
# arr[:,[1]], arr[:,[0]], arr[:,[2]]
print(arr[:, [1, 0, 2]])

# 17.交换二维数组的两行
arr = np.arange(9).reshape(3,3)
print(arr[[1, 0, 2], :])

# 18.反转数组各行
arr = np.arange(9).reshape(3,3)
print(arr[::-1])

# 19.反转数组各列
arr = np.arange(9).reshape(3,3)
print(arr[:, ::-1])