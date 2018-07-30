#coding:utf-8
from PIL import Image

import overfeat
import tensorflow as tf
from tensorflow.python.platform import test
from common import read_data
import numpy as np
from tensorflow.python import debug as tfdbg
import os

num_classes = 5749
train_batch_size = 50
train_height, train_width = 231, 231
train_channel = 3

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

def get_inputs(n = None):
    """
    获取输入
    :return:
    """
    labels, filenames = read_data.get_lfw_paths(n) #[index, path]
    # 文件名的queue，已经包含了enqueue_many操作，已经定义了一个queue runner处理enqueue操作
    filenames_queue = tf.train.string_input_producer(filenames, shuffle=False)
    labels_queue = tf.train.input_producer(labels, shuffle=False)
    # filename = filenames_queue.dequeue()
    # 从queue中dequeue一个文件名，并读取数据
    reader = tf.WholeFileReader()
    filename, image = reader.read(filenames_queue)
    label = labels_queue.dequeue()
    # 解码数据
    #  tf.image.decode_jpeg对图片进行解码，如果用decode_raw结果是不一样的
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [250, 250, train_channel])
    image = tf.random_crop(image, [train_height, train_width, train_channel])
    label = tf.cast(label, tf.int32)
    # 创建一个shuffle queue，并添加enqueue以及dequeue操作
    images, filenames, labels = tf.train.shuffle_batch(
        [image, filename, label],
        batch_size=train_batch_size,
        num_threads=4,
        capacity=2 + 3 * train_batch_size,
        min_after_dequeue=0
    )
    return images, filenames, labels
    #return filename, image, label


def train_LFW():
    # 读取数据
    images, filenames, labels0 = get_inputs()

    images = tf.nn.batch_normalization(images, 0, 1, 0, 1, 10e-8)
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # 前向传播
        logits, fc8 = overfeat.overfeat(images, num_classes)
        # 计算loss函数
        cross_entropy = overfeat.loss(logits, labels0, train_batch_size, num_classes)
        # 反向求导，梯度下降
        train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

        # 保存数据
        saver = tf.train.Saver(tf.all_variables())
        merged = tf.summary.merge_all()
        tb_writer = tf.summary.FileWriter("../result/overfeat")
        # 执行
        sess.run(tf.global_variables_initializer())
        qrs = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            cross_entropy_, summary, _ = sess.run([cross_entropy, merged, train_step])
            print(cross_entropy_)
            # 保存数据
            if i % 100 == 0:
                tb_writer.add_summary(summary, i)
            if i % 100 == 0 or (i + 1) == 1000:
                checkpoint_path = os.path.join("./result/overfeat/model", 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)

        tb_writer.close()
        coord.request_stop()
        coord.join(qrs)


if __name__ == '__main__':
    #test.main()
    train_LFW()