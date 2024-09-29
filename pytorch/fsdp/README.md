PyTorch FSDP 设计解读 - rainbow的文章 - 知乎
https://zhuanlan.zhihu.com/p/694288870

有几个和我想的不同的
1. 我之前想的是一个大的model 切分成多片（比如每一层为一片），切分的每部分在运行前从某个rank广播到其他rank；梯度也是从各个rank gather到这个rank；
但是实际上是模型中的某个部分（比如某个层）为一个FSDPModule，切分到不同的rank上；
前向时：
- 每个FSDPmodule（每个层）执行之前

