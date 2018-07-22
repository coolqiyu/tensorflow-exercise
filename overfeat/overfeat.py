# https://github.com/tensorflow/tensorflow/blob/14f564990e7548b4a5c41da0fc533ec5a7a25abe/tensorflow/contrib/slim/python/slim/nets/overfeat.py
# 数据集：http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
# =====================================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope, arg_scope
from tensorflow.contrib.layers.python.layers import utils
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# 为什么要一层一层嵌套？有些参数对于有些操作并不需要。最后一个用arg_sc，这个是会包含前面所有的吗？
# 为什么用l2_regulaizer来初始化weights_regularizer
def overfeat_arg_scope(weight_decay=0.0005):
    """
    定义arg_scope
    :param weight_decay:
    :return:
    """
    with arg_scope(
            [tf.nn.conv2d],
            activation_fn=tf.nn.relu,
            weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            biases_initializer=tf.zeros_initializer()):
        with arg_scope([tf.nn.conv2d], padding='SAME'):
            with arg_scope([tf.nn.max_pool], padding='VALID') as arg_sc:
                return arg_sc


def variable_init(shape):
    """
    根据shape，初始化变量
    :param shape:
    :return:
    """
    return tf.Variable(tf.zeros(shape))


def loss(logits, labels):
    """
    交叉熵，以及权重衰减
    :param logits: 训练的结果
    :param labels: 真实的结果
    :return:
    """
    batch_size = len(logits)
    num_classes = 40
    labels = tf.sparse_to_dense(labels, [batch_size, num_classes], 1, 0)
    # Computes softmax cross entropy between `logits` and `labels`
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy
    # 加上weght的l2正则化


def overfeat(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='overfeat'):
    """overfeat 网络结构
    全连接层都用conv2d来代替。如果要用全卷积的模式，可以吧spatial_squeeze设置为false
    要注意：tf.nn.conv2d中第二个参数是filter，输入的不应该是[11,11,3,96]，这个是形状！！
    Args:
    inputs: [batch_size, height, width, channels].
    num_classes: 分类个数
    is_training: 是否是训练阶段
    dropout_keep_prob: 训练阶段的dropout
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    Returns:
    the last op containing the log predictions and end_points dict.
    """
    # end_points_collection = scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d

    net = tf.nn.conv2d( #[11, 11, in_, 64] stride = 4
      inputs, variable_init([11, 11, 3, 96]), [1, 4, 4, 1], padding='VALID', name='conv1')
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool1')

    net = tf.nn.conv2d(net, variable_init([5, 5, 96, 256]), [1, 1, 1, 1], padding='VALID', name='conv2')
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool2')
    net = tf.nn.conv2d(net, variable_init([3, 3, 256, 512]), [1, 1, 1, 1], padding='SAME', name='conv3')
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, variable_init([3, 3, 512, 1024]), [1, 1, 1, 1], padding='SAME', name='conv4')
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, variable_init([3, 3, 1024, 1024]), [1, 1, 1, 1], padding='SAME', name='conv5')
    net = tf.nn.relu(net)
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool5')
    # 接下来是3个full connect层，用conv2d来代替full connect

    # 这里为什么用6x6（spatial input size）？
    net = tf.nn.conv2d(net, variable_init([6, 6, 1024, 3072]), [1, 1, 1, 1], padding='VALID', name='fc6')
    net = tf.nn.relu(net)
    net = tf.nn.dropout(
        net, dropout_keep_prob, name='dropout6')
    net = tf.nn.conv2d(net, variable_init([1, 1, 3072, 4096]), [1, 1, 1, 1], padding='SAME', name='fc7')
    net = tf.nn.relu(net)
    net = tf.nn.dropout(
        net, dropout_keep_prob, name='dropout7')
    net = tf.nn.conv2d(
        net,
        variable_init([1, 1, 4096, num_classes]), [1, 1, 1, 1],
        padding='SAME',
        name='fc8')
    # 最后一层不用relu激活
    # net = tf.nn.relu(net)
    # 把collection转换成dict类型
    # end_points = utils.convert_collection_to_dict(end_points_collection)
    # if spatial_squeeze:
    #     # 删除特定维度，把第1和2维删除。默认是把所有大小为1的维度删除
    #     net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    #     end_points[scope + '/fc8'] = net

    return net
