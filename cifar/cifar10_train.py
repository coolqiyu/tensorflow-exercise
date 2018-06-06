# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.
这个文件用来执行训练，是主入口
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
from cifar import cifar10

FLAGS = tf.app.flags.FLAGS
# 用于支持接受命令行传递参数，相当于接受argv
# 第一个是参数名称，第二个参数是默认值，第三个是参数描述
# tf.app.flags
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  # 为什么要写这行？多线程时每个线程有一个图？
  # the Graph.as_default() context manager, which overrides the current default graph for the lifetime of the context:
  # This class is not thread-safe for graph construction. All operations should be created from a single thread, or external synchronization must be provided. Unless otherwise specified, all methods are not thread-safe.
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # 获取训练的数据集和label
    images, labels = cifar10.distorted_inputs()

    # inference构建训练网络，logits为结果
    logits = cifar10.inference(images)

    # loss函数
    loss = cifar10.loss(logits, labels)

    train_op = cifar10.train(loss, global_step)

    # 创建一个saver，第一个参数是要保存的参数列表
    # 这个有问题：all_variables是一个函数
    saver = tf.train.Saver(tf.all_variables())

    # merge_all_summaries变了
    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    # 开始训练，参数设置是什么意思？
    sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # 开始图中所有的queue runners，sess表示用来执行queue操作的session
    tf.train.start_queue_runners(sess=sess)

    # 可视化的写
    # tf.train.SummaryWriter改成tf.summary.FileWriter
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)

    # xrange返回一个生成器
    for step in xrange(FLAGS.max_steps):
      # 执行并计时
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      # 输出辅助信息
      # 每10步打印执行的信息，时间、loss
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value,
                            examples_per_sec, sec_per_batch))

      # 每100步输出可视化数据
      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # 每1000步保存数据
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  # 对训练的目标目录进行判断，有则删，五则新建
  if gfile.Exists(FLAGS.train_dir):
    gfile.DeleteRecursively(FLAGS.train_dir)
  gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  # 执行当前模块的main函数？
  tf.app.run()