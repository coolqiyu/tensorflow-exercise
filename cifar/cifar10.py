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

"""
建立CIFAR-10网络
 # 输入：input, distorted_inputs
 # 推理模型得到预测：predictions = inference(inputs)
 # 损失函数：loss = loss(predictions, labels)
 # 创建图用来计算训练步骤：train_op = train(loss, global_step)
"""