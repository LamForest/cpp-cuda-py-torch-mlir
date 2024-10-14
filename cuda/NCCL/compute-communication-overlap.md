https://github.com/NVIDIA/nccl/issues/689


![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241014141316.png)


schedule gemm的时候，怎么预留资源给allgather:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241014141342.png)

本质上是同时运行，使用不同的SM，如何做到？

A100有128个SM，每个SM可以同时容纳多个warp（64 or 32），称为slot，所以A100可以容纳 128 * 64 = 8K warp

假设GEMM的block个数以及block的thread个数较少，占不满A100上所有的slot，那么SSE会尝试从另一个流launch kernel（比如ncclkernel），填满所有的slot。

2个kernel的warp在SM上是平等调度的吗？AMD有一个SETPRIO的指令，可以设置warp的优先级。

怀疑warp的优先级和stream优先级有关：https://forums.developer.nvidia.com/t/questions-of-cuda-stream-priority/250343/4