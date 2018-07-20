# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope



class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """一个描述ResNet块的元组
  collections.namedtuple: 返回一个元组的子类，包括了命名的字段
  调用Block()创建一个块时，给出的参数和[]中的一一对应
  成员包括：
    scope: 块的域
    unit_fn: ResNet单元的函数(激活函数)
    args: 每个单元有一个元组 (depth, depth_bottleneck, stride) 作为unit_fn的参数
  """

def subsample(inputs, factor, scope=None):
  """利用最大池来对输入进行采样
  Args:
    inputs: 输入[batch, height_in, width_in, channels].
    factor: stride
    scope: Optional variable_scope.
  Returns:
    output: 输出[batch, height_out, width_out, channels] with the
      input, either intact (if factor == 1) or subsampled (if factor > 1).
  """
  if factor == 1:
    return inputs
  else:
    return layers.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """
    卷积计算，padding=same
    分成两类，stride=1时不需要做pad；stride不是1时需pad
    :param inputs:
    :param num_outputs: 输出的filter个数
    :param kernel_size: 过滤器的核大小
    :param stride: 步长
    :param rate:
    :param scope:
    :return:
    """
    if stride == 1:
        return layers_lib.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            stride=1,
            rate=rate,
            padding='SAME',
            scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = array_ops.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return layers_lib.conv2d(
            inputs,
            num_outputs,
            kernel_size,
            stride=stride,
            rate=rate,
            padding='VALID',
            scope=scope)


@add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, outputs_collections=None):
    """
    把blocks中的block按照参数设定添加到net中
    1. 创建scopes，名字为block_name/unit_1, block_name/unit_2
    2. 允许用户显式控制ResNet的output_stride，就是输入和输出的空间分辨率比。
    :param net: BHWC
    :param blocks: 一个长度与ResNet块个数相同的list。每个元素是一个ResNet块，它描述了块中的单元
    :param output_stride: stride为步长。如果为None，则输出按照支配的网络步长计算。否则，
    :param outputs_collections:
    :return:
    """
    current_stride = 1
    rate = 1
    for block in blocks:#遍历每一块
        with variable_scope.variable_scope(block.scope, 'block', [net]) as sc:
            # 遍历block中的每一个单元，包含对应的参数(depth, depth_bottleneck, stride)，用来构建一个短连接的块
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:# 输出的维度只能比原来输入的x大或相同
                    raise ValueError('The target output_stride cannot be reached.')
                with variable_scope.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:# **dict(val) 作为实参时，会自动与命名参数对应起来
                        net = block.unit_fn(net, rate=1, **unit) # 为什么要用current_stride
                        current_stride *= unit.get('stride', 1)# unit为dict，dict.get('key',default_value) 如果key不存在，则返回default_value
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return net


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
  """定义默认的arg scope
  TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
  Args:
    weight_decay: 权重衰减
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': ops.GraphKeys.UPDATE_OPS,
  }

  with arg_scope(
      [layers_lib.conv2d],
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      weights_initializer=initializers.variance_scaling_initializer(),
      activation_fn=nn_ops.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([layers.batch_norm], **batch_norm_params):
      # The following implies padding='SAME' for pool1, which makes feature
      # alignment easier for dense prediction tasks. This is also used in
      # https://github.com/facebook/fb.resnet.torch. However the accompanying
      # code of 'Deep Residual Learning for Image Recognition' uses
      # padding='VALID' for pool1. You can switch to that choice by setting
      # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
      with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
        return arg_sc
