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

"""Routine for decoding the CIFAR-10 binary file format.
read_cifar10: 读取本地CIFAR数据集文件
_generate_image_and_label_batch: 对输入的顺序随机化并生成batch
distorted_inputs：对输入数据集进行各种变形后输出
inputs：读取评估/测试数据集
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  """Reads and parses examples from CIFAR10 data files.
  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CIFAR10Record(object):
    pass
  # 存储读取到的图
  result = CIFAR10Record()
  label_bytes = 1
  # 设置一个图的大小32*32*3
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height *result.width * result.depth

  # 每个记录包含一个label和一个图
  record_bytes = label_bytes + image_bytes

  # tf.FixedLengthRecordReader类：一个reader，可以从文件输出固定长度的记录
  # 构造函数：__init__(record_bytes, header_bytes=None, footer_bytes=None, name=None)
  #         header_bytes头部数据大小，footer_bytes底部数据大小
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  # read每次读一条记录；这个会从queue中移出一个文件
  result.key, value = reader.read(filename_queue)

  # 把字节串value转换成uint8类型的向量
  # record_bytes =[label 第一个value ... 第height*width*depth-1个value]
  record_bytes = tf.decode_raw(value, tf.uint8)

  # tf.cast(x, dtype, name=None): 把x中的元素都转换成dtype类型
  # tf.slice(input_, begin, size, name=None)
  # 设置label
  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # 把记录转换成其它shape，NHW格式，变成3通道的结果
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])

  # transpose 移动矩阵，返回输入图像
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.
  对输入的image和标签label进行随机化组成batch
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  num_preprocess_threads = 16
  # tf.train.shuffle_batch(tensor_list, batch_size, capacity, min_after_dequeue, num_threads=1, seed=None, enqueue_many=False, shapes=None, name=None)
  # 通过对tensor_list随机洗牌来生成batch，最后输出[batch_size, x, y, z]
  # [image, label]原始数据，batch_size：batch大小；num_threds：enqueue tensor_list的线程数；capacity：queue中最多有多少元素
  # min_after_dequeue: 在dequeue后queue中最少有多少元素，以保证混合的程度
  images, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    num_threads=num_preprocess_threads,
    capacity=min_queue_examples + 3 *batch_size,
    min_after_dequeue=min_queue_examples
  )
  # 可视化
  tf.summary.image('images', images)
  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops
  打乱/随机变形输入图像
  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [os.path.join(data_dir, 'data_batch_{i}.bin'.format(i=i))
               for i in xrange(1, 6)]
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  # tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True, seed=None, capacity=32, name=None)
  # 以string_tensor为基础生成一个string_queue
  # shuffle为true表示在一个epoch中是否洗牌
  filenames_queue = tf.train.string_input_producer(filenames)

  # 调用上面的读取操作读取输入数据
  read_input = read_cifar10(filenames_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # tf.image.random_crop(image, size, seed=None, name=None)
  # 把image中每个随机裁剪[height, width]大小
  distorted_image = tf.image.random_crop(reshaped_image, [height, width])

  """
  下面这些是对图像的随机变形
  """
  # tf.image.random_flip_left_right(image, seed=None)
  # 对image进行从左到右随机翻转
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  #tf.image.random_brightness(image, max_delta, seed=None)
  #随机调整image的明亮程度
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  #tf.image.random_contrast(image, lower, upper, seed=None)
  # 调整图像的对比强度
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
  #tf.image.per_image_whitening(image)
  #对image线性变换，使得平均值为0，且归一化
  float_image = tf.image.per_image_whitening(distorted_image)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print('Filling queue with %d CIFAR images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)

  # 产生一个batch，通过构建一个example队列
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  构建输入（评估/测试）数据集：读文件，图像resize，归一化，构成batch
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # eval_data：表示是否用评估数据集还是测试数据集
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_{i}.bin'.format({"i": i}))
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  # 判断文件是否都存在
  for f in filenames:
    if not gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  #tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
  #对图像进行裁剪或填充，使得达到目标大小
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
  float_image = tf.image.per_image_whitening(resized_image)
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)