代码问题
- 要在结构以后初始化一下变量
- sparse_to_dense函数的使用
- feed_dict变量不能是Tensor
- queue的使用：作为输入
- 使用string_input_producer等queue时，遇到报错，图中没有该node：变量名写错
- tf.string_split([filename], '\\') 第一个参数要放在[]，不能直接用filename
- 输入的问题：使用的是jpg格式，图像和对应的分类不能直接读取：使用两个queue，一个读图，一个读分类
- ERROR:tensorflow:Exception in QueueRunner: Session has been closed; Skipping cancelled enqueue attempt with queue not closed：增加Coordinator来协调
参数问题
- logits中值太小，出现log(logits)存在nan
- weight初始化为0：中间激活值为0，而relu在0点不可导
- 为什么执行一次后，cross_entropy就变成0：学习率过大