# 写一些tf基本函数的使用例子
# ===================================================

import tensorflow as tf
import numpy as np


def my_pad():
    """
    tf.pad(t, p)：根据p，对t进行填充
    t: n 维度
    p: [n, 2]
    输出：第D维 paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]
    :return:
    """
    t1 = tf.constant([[1, 2, 3]])
    p1 = tf.constant([[1, 2], [1, 2]])
    with tf.Session() as sess:
        r = tf.pad(t1, p1)
        print(sess.run(r))


def my_variable_scope():
    with tf.Session() as sess:
        with tf.variable_scope("foo") as scope:
            v = tf.get_variable("v", [1])
            # 设置这个区域是可重用的，这样该语句下方的代码才可以访问其前面创建的变量
            # 实际上把reuse标签设置为True
            scope.reuse_variables()
            # assert v.name == "foo/v:0"
            # with tf.variable_scope("foo"):
            v2 = tf.get_variable("v", [1])
            assert v == v2

        # reuse=True说明这个作用域是为了重用，也就是要访问已经存在的变量，如果变量不存在，就会报错
        with tf.variable_scope("foo", reuse=True) as scope:
            v = tf.get_variable("v", [1])
            # assert v.name == "foo/v:0"

            # with tf.variable_scope("foo"):
            v2 = tf.get_variable("v", [1])
            assert v == v2

        # reuse=False说明这个作用域是为了创建变量，如果变量存在，就会报错
        with tf.variable_scope("foo", reuse=False) as scope:
            v = tf.get_variable("v", [1])
            v2 = tf.get_variable("v", [1])
            assert v == v2

def my_name_scope():
    with tf.Session() as sess:
        with tf.name_scope("name_scope"):
            v = tf.get_variable("v", [1])
            # v:0
            print(v.name)
        with tf.variable_scope("name_scope", reuse=False):
            v = tf.get_variable("v", [1])
            # name_scope/v:0
            print(v.name)

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
    """
    concate把某一维拼接
    :return:
    """
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    t3 = [[1], [2], [3]]
    t4 = [[4], [5], [6]]
    a = tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    b = tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    c = tf.concat([t3, t4], 0)
    d = tf.concat([t3, t4], 1)
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(c))
        print(sess.run(d))


def my_sparse_to_dense():
    """
    def sparse_to_dense(sparse_indices,
                       output_shape,
                       sparse_values,
                       default_value=0,
                       validate_indices=True,
                       name=None):
    将sparse_indices扩展成一个更大的tensor(output_shape)，dense中与sparse_indices标识相同的位置置为sparse_value，其他的为默认值
    :return:
    """
    data = [i for i in range(10)]
    indices = tf.reshape(data, [10, 1])
    labels = tf.concat([indices, indices], 1)

    dense = tf.sparse_to_dense(labels, [10, 10], 1.0, 0.0)
    with tf.Session() as sess:
        print(sess.run(dense))


def my_zero_fraction():
    z = [0, 1, 0, 0, 0]
    with tf.Session() as sess:
        print(sess.run(tf.nn.zero_fraction(z)))


def fetch():
    """
    sess.run中第一个参数是执行并取出的数据
    :return:
    """
    with tf.Session() as sess:
        a = tf.constant([2, 2])
        b = tf.constant([2, 2])
        a, b = sess.run([a, b])
        print(a)
        print(b)

if __name__ == "__main__":
    # fetch()
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
    # my_concat()
    # my_sparse_to_dense()
    # indices = tf.reshape([i for i in range(128)], [128, 1])
    # indices2 = tf.reshape([i for i in range(128)], [128, 1])
    # print(tf.concat(1, [indices, indices2]))
    # my_zero_fraction()
    # my_variable_scope()
    # my_pad()

# tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, name=None)
#  tf.concat(1, [indices, sparse_labels])
# tf.train.ExponentialMovingAverage
# exponential_decay 学习率衰减
# softmax_cross_entropy_with_logits这个！
    my_name_scope()