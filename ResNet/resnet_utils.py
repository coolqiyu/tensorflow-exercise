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


from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
import collections


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """一个描述ResNet块的元组
  collections.namedtuple: 返回一个元组的子类，包括了命名的字段
  Its parts are:
    scope: 块的域
    unit_fn: ResNet单元的函数(激活函数)
    args: 每个单元有一个元组 (depth, depth_bottleneck, stride) 作为unit_fn的参数
  """

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
        inputs = array_ops.pad(input, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
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
    stack ResNet 块以及控制输出feature的密度
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
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            net = utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return net