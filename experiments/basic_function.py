# 写一些基本函数的使用例子
# ===================================================

import tensorflow as tf
import numpy as np

def myslice():
    input = tf.constant([[[1, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]]])
    # slice(data, begin, size)
    #  begin和size都是张量，每一维一一对应
    #  begin中第一个元素表示为0(zero_based)， size中第一个元素表示为1(one_based)
    slice_result = tf.slice(input, [0,0,0], [1,2,1])
    with tf.Session() as sess:
        print(sess.run(slice_result))

def my_decode_raw():
    bytes = "3"
    # decode_raw(bytes, out_type, little_endian=True, name=None)
    # bytes: 应该是ASCII编码符号
    result = tf.decode_raw(bytes, tf.uint8)
    with tf.Session() as sess:
        print(sess.run(result))

def my_transpose():
    input= tf.constant([[1, 2, 3], [4, 5, 6]])
    # y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])
    # input[i][j][k] => output的第0维就是perm[0]的结果，第1维就是perm[1]的结果
    # 比如perm=[2][0][1] 则input[i][j][k]=>input的第0维变成output的第perm[0]=2维
    # output[k][i][j]
    result = tf.transpose(input, [1,0])
    input1 = tf.constant([[[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [10, 11, 12]]])
    result1 = tf.transpose(input1, [1,0,2])
    with tf.Session() as sess:
        print(sess.run(result))
        print(sess.run(result1))
"""
用numpy实现
"""
def my_transpose1(input, output, perm):
    for i in range(2):
        for j in range(2):
            for k in range(3):
                output[j][i][k] = input[i][j][k]

if __name__ == "__main__":
    # myslice()
    #my_decode_raw()
    my_transpose()
    output = np.zeros((2,2,3), np.uint32)

    my_transpose1([[[1, 2, 3], [4, 5, 6]],
                   [[7, 8, 9], [10, 11, 12]]], output, 0)
    print(output)

# tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
#  tf.concat(1, [indices, sparse_labels])