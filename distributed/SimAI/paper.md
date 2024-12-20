https://ennanzhai.github.io/pub/nsdi25spring-simai.pdf


一个端到端的LLM性能模拟器


## 方法

### 估计计算kernel的时间
称之为SimAI-CP。对于能够访问的GPU型号，直接预先跑各种形状的benchmark，然后存到db里面；用的时候查表。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241220142132.png)

这样做的准确率在96.9% to 99.5%

SimAI-CP-Model: 如果要估计不能访问的GPU，则使用A100的型号+roofline模型来估计。这里的model应该是建模的意思，起的烂名字。


如何hook每个module？是aten算子还是module级别？

### 估计通信的时间
这才是难点

## 实验
我比较关注的是广泛的实验下，SimAI的仿真结果和真实结果的diff有多大。论文中给出的数据是1.6%。

### 通信kernel
通信量越小，越不稳定，差距越多


### 计算kernel
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241220142605.png)

### 端到端时间
128 512 1024卡模拟时间和真实时间的diff：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241220142915.png)