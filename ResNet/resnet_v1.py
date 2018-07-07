# tensorflow源码中的resnet
# https://github.com/tensorflow/tensorflow/blob/14f564990e7548b4a5c41da0fc533ec5a7a25abe/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py
# ============================================

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import resnet_utils
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils

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
        with arg_scope([layers.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The ouput_stride needs to be a multiple of 4.')
                        output_stride /= 4
                net =resnet_utils.conv2d_name(net, 64, 7, stride = 2, scope = 'conv1')
                net = layers_lib.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            if global_pool:
                net = math_ops.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)
            if num_classes is not None:
                net = layers.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn = None,
                    normalizer_fn = None,
                    scope = 'logits')
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = layers_lib.softmax(net, scope = 'predictions')
                return net, end_points