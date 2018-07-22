from . import overfeat
import tensorflow as tf
from tensorflow.python.platform import test
from common import read_data
import numpy as np


# 测试overfeat
class OverFeatTest(test.TestCase):
    def testTrainEvalWithReuse(self):
        train_batch_size = 2
        eval_batch_size = 1
        train_height, train_width = 231, 231
        eval_height, eval_width = 281, 281
        num_classes = 1000
        with self.test_session():
            # 训练阶段
            train_inputs = tf.Variable(tf.random_uniform(
                [train_batch_size, train_height, train_width, 3]))
            logits, _ = overfeat.overfeat(train_inputs)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [train_batch_size, num_classes])
            tf.get_variable_scope().reuse_variables()
            # 评估阶段
            eval_inputs = tf.random_uniform(
                (eval_batch_size, eval_height, eval_width, 3))
            logits, _ = overfeat.overfeat(
                eval_inputs, is_training=False, spatial_squeeze=False)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [eval_batch_size, 2, 2, num_classes])
            # 评估获得正确率
            logits = tf.reduce_mean(logits, [1, 2])
            predictions = tf.argmax(logits, 1)
            self.assertEquals(predictions.get_shape().as_list(), [eval_batch_size])


def train_OLR():
    num_classes = 40
    train_batch_size = 20
    train_height, train_width = 92, 112

    with tf.Session() as sess:
        x, y = read_data.read_ORL_face_data()
        x = np.reshape(x, [len(x), train_height, train_width, 1])

        sess.run(tf.initialize_all_variables())

        for i in range(1000):
            # 前向传播
            logits, _ = overfeat.overfeat(x, num_classes)
            # 计算loss函数
            cross_entropy = overfeat.loss(logits, y)
            # 反向求导，梯度下降
            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

            _, cross_entropy = sess.run([train_step, cross_entropy])
            print(cross_entropy)







if __name__ == '__main__':
  test.main()