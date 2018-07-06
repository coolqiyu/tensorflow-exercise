"""
 TF的基本使用
 图：计算的过程
 节点：操作，如add
 边：数据=tensor
 会话：执行计算
"""
import tensorflow as tf


class Example(object):
    # 第一个是最基本的例子
    @staticmethod
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

    """ 第二个例子：使用变量Variable
    为什么用Variable: 
    # 一个Variable代表一个可修改的张量，
    # 存在在TensorFlow的用于描述交互性操作的图中。
    # 它们可以用于计算输入值，也可以在计算中被修改。
    # 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示
    """
    @staticmethod
    def Example2():
        # 定义一个变量
        state = tf.Variable(0, name="counter")
        one = tf.constant(1)
        new_value = tf.add(state, one)
        # 用new_value来更新state，同时返回一个tensor保存了state的值
        update = tf.assign(state, new_value)

        # initialize_all_variables()：增加一个op，用来初始化所有变量
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(state)
            print(state)
            for _ in range(3):
                print(sess.run([update, state]))
                # 直接打印的结果和上面是不同的！还是不变的
                print(update, state)

    # 第三个例子：使用fetch：用run执行时，传入多个参数，就可以取回多个tensor
    @staticmethod
    def Example3():
        input1 = tf.constant(3.0)
        input2 = tf.constant(2.0)
        input3 = tf.constant(5.0)

        intermed = tf.add(input2, input3)
        mul = tf.multiply(input1, intermed)

        with tf.Session() as sess:
            # 传入多个参数，这样可以返回中间结果
            result = sess.run([mul, intermed])
            print(result)

    # 第四个列子：feed：可以不一开始给值，而是中间可以替换
    @staticmethod
    def Example4():
        input1 = tf.constant([3.0]) #tf.placeholder(tf.float32)
        input2 = tf.placeholder(tf.float32)
        output = tf.multiply(input1, input2)

        # input用placeholder先占位(也可以用变量，但是也要保证类型一致)，然后run的时候传入feed_dict一个字典，来替换未赋值的变量
        with tf.Session() as sess:
            result = sess.run(output, feed_dict={input1:[7.], input2:[2.]})
            print(result)