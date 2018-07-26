from PIL import Image

import overfeat
import tensorflow as tf
from tensorflow.python.platform import test
from common import read_data
import numpy as np
from tensorflow.python import debug as tfdbg

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


num_classes = 5749
train_batch_size = 10
train_height, train_width = 231, 231

def get_inputs():
    """
    对LFW做数据增强，裁剪成231x231的数据
    :return:
    """
    filenames = read_data.get_lfw_paths() #[index, path]
    # 文件名的queue，已经包含了enqueue_many操作，已经定义了一个queue runner处理enqueue操作
    filenames_queue = tf.train.string_input_producer(filenames)
    # 从queue中dequeue一个文件名，并读取数据
    filename = filenames_queue.dequeue()
    with Image.open(filename) as f:
        image = np.array(f).astype(np.float32)
    # 创建一个shuffle queue，并添加enqueue以及dequeue操作
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=train_batch_size,
        num_threads=2,
        capacity=2 + 3 * train_batch_size,
        min_after_dequeue=2
    )
    return images, label_batch

def train_LFW():
    # 读取数据
    x, labels = get_inputs()
    with tf.Session() as sess:
        #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # 设置输入数据
        # 前向传播
        logits = overfeat.overfeat(x, num_classes)
        # 计算loss函数
        cross_entropy = overfeat.loss(logits, labels)
        # 反向求导，梯度下降
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        for i in range(1000):
            logits_,_ = sess.run([logits, train_step])
            print(logits_)


if __name__ == '__main__':
    train_LFW()