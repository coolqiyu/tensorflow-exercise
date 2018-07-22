# 读取自己下载的数据集
# ============================
import re
import os
import numpy as np

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
    return images, labels

read_ORL_face_data()