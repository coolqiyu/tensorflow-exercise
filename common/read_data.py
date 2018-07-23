# 读取自己下载的数据集
# ============================
import re
import os
import numpy as np
from PIL import Image

def read_ORL_face_data():
    """
    读取ORL_face数据, pgm格式
    :return:
    """
    path = "../dataset/ORL_face/"
    # 列出path下的所有文件名
    s_names = os.listdir(path)
    images = []
    labels = []
    byteorder = '<'
    for index, s_name in enumerate(s_names):
        s_path = os.path.join(path, s_name)
        data_names = os.listdir(s_path)
        for data_name in data_names:
            data_path = os.path.join(s_path, data_name)
            with open(data_path, 'rb') as f:
                buffer = f.read()
                header, width, height, maxval = re.search(
                    b"(^P5\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n])*"
                    b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

                image_data = np.frombuffer(buffer,
                                 dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                                 count=int(width) * int(height),
                                 offset=len(header)
                                 ).reshape((int(height), int(width)))
                images.append(image_data)
                labels.append(int(s_name[1:]))
    # 最后输出的数据类型需要做一下转换
    # images = np.array(np.reshape(images, [len(images), train_height, train_width, 1])).astype(dtype=np.float32)
    return images, labels

def read_lfw_data(end = 9999999):
    """
    读取lfw数据，jpg格式
    :return:
    """
    path = "../dataset/lfw/lfw"
    # 列出path下的所有文件名
    s_names = os.listdir(path)
    images = []
    labels = []
    for index, s_name in enumerate(s_names):
        if index == end:
            break
        s_path = os.path.join(path, s_name)
        data_names = os.listdir(s_path)
        for data_name in data_names:
            data_path = os.path.join(s_path, data_name)
            with Image.open(data_path) as f:
                # 直接转换成(250,250,3)的数组
                image = np.array(f).astype(np.float32)
                images.append(image)
                labels.append(index)
    return images, labels
