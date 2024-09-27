torch/csrc/autograd/functions/accumulate_grad.h

一般情况会有2种情况，第一次反向，grad还没有初始化，猜测走这里
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240927005455.png)
第2次反向，猜测是在这里，因为这个文件中只有这么一处的 += 或者 add_
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240927005210.png)

还有几个TODO
1. 打一些日志，确认上述2个猜测
2. 什么是GradMode? 和AutoGradMode的关系？
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240927005638.png)
3. 为什么tensor.register_hook不行，而要post_accumulate_hook?我试了，改完是一样的

此外，`accumulateGrad`是在`AccumulateGrad::apply`中调用的，调用完成之后，再调用post_accumulate_hook：

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240927005834.png)