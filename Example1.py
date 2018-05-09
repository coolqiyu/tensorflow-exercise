# 第一个例子
#################
# TF的基本使用
# 图：计算的过程
# 节点：操作，如add
# 边：数据=tensor
# 会话：执行计算
#################
import tensorflow as tf

def Example1():
    # tf.constant：创建常量的op
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])

    # tf.matmul：矩阵乘法的op，以matrix1和matrix2作为输入
    # product是乘法Op的输出
    product = tf.matmul(matrix1, matrix2)

    # 创建一个会话，启动一个默认图
    sess = tf.Session()
    # sess.run：希望取回product的结果，触发图中的三个op
    result = sess.run(product)
    print(result)

    # 关闭会话
    sess.close()