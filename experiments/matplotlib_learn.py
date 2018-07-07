from matplotlib import pyplot
import numpy as np

# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
def plot1():
    """
    给定y，默认x=y
    :return:
    """
    pyplot.plot([1, 2, 3, 4])
    pyplot.ylabel('y label')
    pyplot.show()


def plot2():
    """
    同时给出x和y的取值
    :return:
    """
    pyplot.plot([1, 2, 3, 4], [1, 4, 9, 16])
    pyplot.show()


def plot3():
    """
    设置点的形状, r红色o圆
    axis([x_min, x_max, y_min, y_max])定义坐标轴的取值范围
    :return:
    """
    pyplot.plot([1, 2, 3, 4, 5], 'ro')
    pyplot.axis([0, 6, 0, 20])
    pyplot.show()


def plot4():
    """
    绘制多条线
    :return:
    """
    t = np.arange(0., 5., 0.2)
    pyplot.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    pyplot.show()


def plot5():
    """
    散点图
    scatter(x, y, s=None, c=None, data=None)
    x, y数据(y-x) s：大小  c: 颜色
    :return:
    """
    data = {'a': np.arange(50), 'c': np.random.randint(0, 50, 50), 'd': np.random.rand(50)}
    data['b'] = data['a'] + 10 * np.random.rand(50)
    data['d'] = np.abs(data['d']) * 100
    pyplot.scatter('a', 'b', s='d', c='c', data=data)
    pyplot.show()


def plot6():
    """
    在一个中画多图，柱状图、散点图、折线图
    可以不用figure来创建新图，pylot.bar()...会自动放在一个图中
    :return:
    """
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]
    # 创建一个图，大小为9rows, 3columns
    pyplot.figure(1, figsize=(9, 3))

    # 柱状图
    # 创建一个子图，大小为原图的(row, column, index): 1/row, 1/column, 第几个
    pyplot.subplot(131)
    pyplot.bar(names, values)
    # 散点图
    pyplot.subplot(132)
    pyplot.scatter(names, values)
    #
    pyplot.subplot(133)
    pyplot.plot(names, values)

    pyplot.suptitle('Categorical Plotting')
    pyplot.show()


def plot7():
    """
    直方图
    :return:
    """
    mu, sigma = 100, 15
    # randn: 标准正态分布
    x = mu + sigma * np.random.randn(10000)
    # hist(数据，50个柱子，是否频率直方图，颜色，柱形透明图)
    # density=True 频率直方图，Fals 频数直方图
    pyplot.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    pyplot.xlabel('Smarts')
    pyplot.ylabel('Probability')
    pyplot.title('Histogram of IQ')
    # text(x, y, data)
    pyplot.text(60, 0.025, r'$\mu=100,\ \sigma=15$')
    #pyplot.axis([40, 160, 0, 0.03])
    # 是否显示中间的格子
    pyplot.grid(True)
    pyplot.show()


if __name__ == "__main__":
    plot7()