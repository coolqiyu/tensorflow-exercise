"""
http://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-10数据集：60000 32*32 彩色图，分成10类，每类6000个图。分成50000 train图，10000 test图
数据集文件结构(python)：
    data_batch1~data_batch_5为train图
    test_batch为test图
    每个batch文件：10000x3072 每行一个图，32*32*3 红-绿-蓝顺序
    batches.meta.txt：ASCII文件把数字标签(0-9)映射成有意义的类型名称
"""
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

"""Builds the CIFAR-10 network.
构建网络
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10_input
from . import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data/',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  调用cifar10_input中的distorted_inputs的功能，并构成mini-batch
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  构建评估数据集的输入
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
      raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def _activation_summary(x):
  """Helper to create summaries for activations.
  创建激活的summary
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = re.sub('%_[0-9]*/' %TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
      # tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None, trainable=True, collections=None)
      # 根据给定的参数获取或创建变量
      var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  为什么要做这个？add_to_collection是什么意义？
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  # 因为要使用L2正则化权重，所以使用collection来保存多个权重向量的L2值
  if wd:
      # tf.multiply是元素相乘
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      # tf.add_to_collection(name, value)
      # collection提供一个全局的存储机制，不会受到变量名生存空间的影响。一处保存，到处可取
      tf.add_to_collection('losses', weight_decay)
  return var


def inference(images):
  """Build the CIFAR-10 model.
  构建网络
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  # 用tf.get_variable来创建变量，这样就可以再多GPU上共享。如果只是单GPU，可以使用tf.Variable

  # conv1
  with tf.variable_scope('conv1') as scope:
      # = image.height x image.width x 3
      # w
      kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv1)
      # = image.height x image.width x 64

      # pool1
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
      # = image.height/2 x image.width/2 x 64
      # norm1
      # DROPOUT和数据增强作为relu激励之后防止数据过拟合而提出的一种处理方法,全称是 local response normalization--局部响应标准化
      norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm1')

  # = image.height/2 x image.width/2 x 64
  # conv2
  with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                           stddev=1e-4, wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      bias = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(bias, name=scope.name)
      _activation_summary(conv2)
      # = image.height/2 x image.width/2 x 64

      # norm2
      norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                        name='norm2')
      # pool2
      pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                             strides=[1, 2, 2, 1], padding='SAME', name='pool2')
      # = image.height/4 x image.width/4 x 64


  # local3
  with tf.Variable_scope('local3') as scope:
      dim = 1
      # shape第一个值是batch_size，从[1:]获取一个图像现在有多少个参数表示
      for d in pool2.get_shape()[1:].as_list():
          dim *= d
      # 把pool2
      reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
      _activation_summary(local4)

  # softmax层，下面只是计算了wx+b，没有执行softmax
  with tf.variable_scope('softmax_linear') as scope:
      weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                            stddev=1 / 192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
      _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  定义loss函数，优化目标
  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # 对label进行变形，以便后面和logits做交叉熵计算
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  # 生成[0 1 ... batch_size-1]，后面作为sparse_labels的标签
  indices = tf.reshape(range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  # tf.concat(concat_dim, values, name='concat')
  # 把values中的tensor在第dim维进行合并，结果为 [[1 label1],[2 label2]]
  concated = tf.concat(1, [indices, sparse_labels])
  # 这个操作有什么用？转换成one-shot？
  dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)
  # 交叉熵
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      logits, dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # add_n，对所有l2 loss进行求和
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # 指数加权移动平均值ExponentialMovingAverage类
  # def __init__(self, decay, num_updates=None, zero_debias=False, name="ExponentialMovingAverage")
  # decay：下降率 num_updates：要更新的次数
  loss_average = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  # apply: 增加训练变量的浅拷贝，并增加保存移动平均值的操作。这个操作一般在每一次训练步后执行
  loss_average = loss_average.apply(losses + [total_loss])

  # 为独立的loss和loss平均值可视化
  for l in losses + [total_loss]:
      tf.summary.scalar(l.op.name + ' (raw)', l)
      # average返回一个变量对应的shadow变量
      tf.summary.scalar(l.op.name, loss_average.average(l))

def train(total_loss, global_step):
  """Train CIFAR-10 model.
  训练模型
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.当前是第几步？
  Returns:
    train_op: op for training.
  """
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # 学习率衰减：decayed_learning_rate = learning_rate *
  #                                     decay_rate ^ (global_step / decay_steps)
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)

  tf.summary.scalar('learning_rate', lr)

  loss_averages_op = _add_loss_summaries(total_loss)

  # 控制依赖，当参数执行完成，才执行下面的
  # 为什么这里要做这个控制
  with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.GradientDescentOptimizer(lr)
      # tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1)
      # 为total_loss计算gradient
      grads = opt.compute_gradients(total_loss)

  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # 可视化
  for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
  for grad, var in grads:
    if grad:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')#no_op什么也不做？

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website.
     从Alex网站下载cifar数据集并解压到dest_directory下
  """
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
      os.mkdir(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                           float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                               reporthook=_progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)






















