# 写一些基本函数的使用例子
# ===================================================

import tensorflow as tf


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
    result = tf.transpose(input, [1,0])
    input1 = tf.constant([[[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [10, 11, 12]]])
    result1 = tf.transpose(input1, [1,0,2])
    with tf.Session() as sess:
        print(sess.run(result))
        print(sess.run(result1))

if __name__ == "__main__":
    # myslice()
    #my_decode_raw()
    my_transpose()