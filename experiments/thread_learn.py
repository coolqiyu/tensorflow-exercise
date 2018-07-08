import threading
import time
# Thread是类，表示一个活动。有两种方式来创建线程。
# 1. 给构造函数传一个函数
# 2. 继承Thread，并重写run函数，只能重写__init__()和run。其它函数不可以重写
#
# 如何开始一个线程：start()，它会调用run函数
# 如何阻塞一个线程：在主线程中调用t.join()，则当t执行结束后，主线程才会继续执行
# 线程的名字：构造函数定义name属性
# daemon线程：设置成daemon线程，当daemon线程离开时，整个程序结束
# 把一个任务分成几个线程来执行，当这些线程结束时才继续接下来的任务：用for循环创建多个线程，接下来for循环start，再for循环join
def action(arg):
    time.sleep(1)
    print(arg)


for i in range(4):
    t = threading.Thread(target=action, args=(i, ))
    #threading.current_thread().join()
    t.start()
    # 当t执行结束是，才会继续join()后面的程序
    t.join()
