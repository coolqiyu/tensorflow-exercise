#coding:utf-8
from PIL import Image

import overfeat
import tensorflow as tf
from tensorflow.python.platform import test
from common import read_data
import numpy as np
from tensorflow.python import debug as tfdbg


num_classes = 5749
train_batch_size = 50
train_height, train_width = 231, 231
train_channel = 3

def get_inputs(n):
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
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=train_batch_size,
        num_threads=1,
        capacity=2 + 3 * train_batch_size,
        min_after_dequeue=0
    )
    return images, labels
    #return filename, image, label

def train_LFW():
    # 读取数据
    images, labels = get_inputs()
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
        # 设置输入数据
        # 前向传播
        logits, fc8 = overfeat.overfeat(images, num_classes)
        # 计算loss函数
        labels, cross_entropy  = overfeat.loss(logits, labels)
        # 反向求导，梯度下降
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        sess.run(tf.global_variables_initializer())
        qrs = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            cross_entropy_ , _ = sess.run([cross_entropy, train_step])
            print(cross_entropy_)
        coord.request_stop()
        coord.join(qrs)


if __name__ == '__main__':
    train_LFW()