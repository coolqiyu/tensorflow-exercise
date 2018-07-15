from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from . import resnet_v1
from . import resnet_utils
from tensorflow.contrib.framework.python.ops import arg_scope


def create_test_input(batch_size, height, width, channels):
    """创建测试输入的张量
    Args:
    batch_size: 每个batch的图盘数量
    height: 图片高
    width: 图片宽
    channels: 图片通道
    Returns:
    如果参数中有未知的，则返回placeholder。否则返回一个constant张量
    """
    if None in [batch_size, height, width, channels]:
        return array_ops.placeholder(dtypes.float32,
                                     (batch_size, height, width, channels))
    else:
        return math_ops.to_float(
            np.tile(
                np.reshape(
                    np.reshape(np.arange(height), [height, 1]) + np.reshape(
                        np.arange(width), [1, width]), [1, height, width, 1]),
                [batch_size, 1, 1, channels]))


class ResnetCompleteNetworkTest(test.TestCase):
    """测试类"""
    def _resnet_small(self,
                    inputs,
                    num_classes=None,
                    is_training=True,
                    global_pool=True,
                    output_stride=None,
                    include_root_block=True,
                    reuse=None,
                    scope='resnet_v1_small'):
        """一个简单的浅层ResNet"""
        block = resnet_v1.resnet_v1_block
        blocks = [
            block('block1', base_depth=1, num_units=3, stride=2),
            block('block2', base_depth=2, num_units=3, stride=2),
            block('block3', base_depth=4, num_units=3, stride=2),
            block('block4', base_depth=8, num_units=2, stride=1),
        ]
        return resnet_v1.resnet_v1(inputs, blocks, num_classes, is_training,
                                   global_pool, output_stride, include_root_block,
                                   reuse, scope)

    def testClassificationShapes(self):
        global_pool = True
        num_classes = 10
        inputs = create_test_input(2, 224, 224, 3)
        with arg_scope(resnet_utils.resnet_arg_scope()):
          _, end_points = self._resnet_small(
              inputs, num_classes, global_pool=global_pool, scope='resnet')
          endpoint_to_shape = {
              'resnet/block1': [2, 28, 28, 4],
              'resnet/block2': [2, 14, 14, 8],
              'resnet/block3': [2, 7, 7, 16],
              'resnet/block4': [2, 7, 7, 32]
          }
          for endpoint in endpoint_to_shape:
            shape = endpoint_to_shape[endpoint]
            self.assertListEqual(end_points[endpoint].get_shape().as_list(), shape)

if __name__=="__main__":
    test.main()