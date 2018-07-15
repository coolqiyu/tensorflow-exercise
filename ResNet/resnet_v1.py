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


@add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
  """Bottleneck residual unit variant with BN after convolutions.
  Xl——weight——BN——ReLU——weight——BN——addition——ReLU——Xl+1
  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: 单元输出的深度(channel)
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = layers.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=None,
          scope='shortcut')

    residual = layers.conv2d(
        inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
    residual = resnet_utils.conv2d_same(
        residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
    residual = layers.conv2d(
        residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')

    output = nn_ops.relu(shortcut + residual)

    return utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1(inputs, blocks, num_classes = None, is_training = True, output_stride = None, include_root_block = True, reuse = None, scope = None):
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
    :return:
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as scope:
        end_points_collection = scope.original_name_scope + '_end_points'
        with arg_scope([layers_lib.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The ouput_stride needs to be a multiple of 4.')
                        output_stride /= 4
                net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            if global_pool:
                net = math_ops.reduce_mean(net, [1, 2], name='pool5', keep_dims = True)
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
  Args:
    scope: 块的域
    base_depth: bottleneck的深度
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