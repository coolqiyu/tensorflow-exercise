# tensorflow源码中的resnet
# https://github.com/tensorflow/tensorflow/blob/14f564990e7548b4a5c41da0fc533ec5a7a25abe/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py
# ============================================

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from . import resnet_utils
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import math_ops

@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck的构建
  shortcut = input*conv2d(1x1)     residual = input * conv2d(1x1) * conv2d(3x3) * conv2d(1x1)   output = relu(shortcut + residual)
  当把两个ResNet块放在一起时，第一个块的最后一个单元应该使用stride2
  Args:
    inputs: 输入 [batch, height, width, channels].
    depth: 单元输出的深度(channel)
    depth_bottleneck: bottelneck层的深度
    stride: ResNet单元的步长。决定单元的输出相对于输入的采样数量
    rate: An integer, rate for atrous convolution.
    outputs_collections: 保存ResNet单元的输出
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv2d(# (2,56,56,4)
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')
    # layers.conv2d(input, num_output, kernel_size, stride, padding="SAME")
    residual = layers.conv2d(# (2,56,56,1)
        inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = resnet_utils.conv2d_same(# (2,56,56,1)
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = layers.conv2d(# (2,56,56,64)
        residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
    # 这里不是直接用inputs与bottleneck的输出相加？
    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
    """
    生成一个resnet model
    :param inputs:
    :param blocks:
    :param num_classes:
    :param is_training:
    :param output_stride:
    :param include_root_block:
    :param reuse:
    :param scope:
    :return:(2,224,224,3)——conv2d(7,7,3,64) s=2 p=same(2,112,112,64)——maxpool(1,3,3,1) s=2(2,56,56,64)——
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with arg_scope(
                [layers.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
                outputs_collections=end_points_collection):
            with arg_scope([layers_lib.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The ouput_stride needs to be a multiple of 4.')
                            output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)#(2,7,7,32)
                # 全局平均池化
                if global_pool:
                    net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                if num_classes is not None:
                    net = layers.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn = None,
                        normalizer_fn = None,
                        scope='logits')
                end_points = utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = layers_lib.softmax(net, scope='predictions')
                    return net, end_points

def resnet_v1_block(scope, base_depth, num_units, stride):
  """创建resnet_vi bottleneck 块的帮助函数
  包含num_units个单元
  Args:
    scope: 块的域
    base_depth: bottleneck 决定过滤器的个数
    num_units: 块中的单元数
    stride: 块的步长，最后一个单元的。其他单元的stride=1
  Returns:
    返回一个resnet bottleneck块
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])