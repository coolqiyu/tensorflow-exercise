# 写一些基本函数的使用例子
# ===================================================

import tensorflow as tf
import numpy as np


def myslice():
    """
    slice(data, begin, size)
    begin和size都是张量，每一维一一对应
    begin中第一个元素表示为0(zero_based)， size中第一个元素表示为1(one_based)
    :return:
    """
    input = tf.constant([[[1, 1, 1], [2, 2, 2]],
                         [[3, 3, 3], [4, 4, 4]],
                         [[5, 5, 5], [6, 6, 6]]])
    slice_result = tf.slice(input, [0,0,0], [1,2,1])
    with tf.Session() as sess:
        print(sess.run(slice_result))


def my_decode_raw():
    """
    decode_raw(bytes, out_type, little_endian=True, name=None)
    bytes: 应该是ASCII编码符号
    :return:
    """
    bytes = "3"
    result = tf.decode_raw(bytes, tf.uint8)
    with tf.Session() as sess:
        print(sess.run(result))


def my_transpose():
    """
    tf.transpose(input, perm)
    y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])
    input[i][j][k] => output的第0维就是perm[0]的结果，第1维就是perm[1]的结果
    比如perm=[2][0][1] 则input[i][j][k]=>input的第0维变成output的第perm[0]=2维
    output[k][i][j]
    :return:
    """
    input= tf.constant([[1, 2, 3], [4, 5, 6]])
    result = tf.transpose(input, [1,0])
    input1 = tf.constant([[[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [10, 11, 12]]])
    result1 = tf.transpose(input1, [1,0,2])
    with tf.Session() as sess:
        print(sess.run(result))
        print(sess.run(result1))
"""
用python直接实现
"""
def my_transpose1(input, output, perm):
    for i in range(2):
        for j in range(2):
            for k in range(3):
                output[j][i][k] = input[i][j][k]


"""
tf.app.flags：相当于程序的执行参数
tf.app.flags.DEFINE_string(参数名称，参数默认值，描述) DEFINE_参数类型
要求main函数必须有参数，否则会报错
https://blog.csdn.net/leiting_imecas/article/details/72367937
"""
tf.app.flags.DEFINE_string('str_name', 'def_v_1', "descript1")
FLAGS = tf.app.flags.FLAGS
def app_flag():
    pass

def main(_):
    print(FLAGS.str_name)


def my_reshape():
    data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    with tf.Session() as sess:
        out = tf.reshape(data, [2, 2, 3])
        print(sess.run(out))


def all_variables():
    """
    tf.all_variables: 返回所有trainable=True的参数
    :return:
    """
    # 默认为True
    var1 = tf.Variable(0, trainable=True)
    var2 = tf.Variable(1, trainable=False)
    print(tf.all_variables())


def my_shuffle_batch():
    tensor = [1]
    images = tf.train.shuffle_batch(tensor, batch_size=2,
    num_threads=1,
    capacity=2,
    min_after_dequeue=1)
    with tf.Session() as sess:
        print(sess.run(images))


def my_concat():
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    a = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    b = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(b))


if __name__ == "__main__":
    # myslice()
    #my_decode_raw()
    # my_transpose()
    # output = np.zeros((2,2,3), np.uint32)
    #
    # my_transpose1([[[1, 2, 3], [4, 5, 6]],
    #                [[7, 8, 9], [10, 11, 12]]], output, 0)
    # print(output)
    """
    通过一个可选的main函数和参数列表执行程序
    argv = flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)
    main = main or _sys.modules['__main__'].main
    Call the main function, passing through any arguments to the final program.
    _sys.exit(main(argv))
    """
    # tf.app.run()
    # all_variables()
    #b my_reshape()
    # my_shuffle_batch()
    my_concat()

# tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
#  tf.concat(1, [indices, sparse_labels])
# tf.train.ExponentialMovingAverage
# exponential_decay 学习率衰减