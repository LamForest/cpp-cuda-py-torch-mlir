TOOD
不同的fsdp配置，时间、显存、nsystem时序图。

fsdp zero1 2下的源码分析

## 为什么需要pytorch的fsdp

许多框架自己实现了fsdp，但是兼容性不好；比如torch升级了，可能DS的fsdp就失效了

许多框架的实现不是通用的，而是与某个结构绑定，比如Megatron





> The FSDP algorithm is motivated by theZeroRedundancyOptimizer [27, 28] technique from DeepSpeed butwith a revised design and implementation that is aligned with theother components of PyTorch.
>
> FSDP breaks down a model instanceinto smaller units and then flattens and shards all of the parameterswithin each unit. The sharded parameters are communicated andrecovered on-demand before computations, and then they are immediatelydiscarded afterwards. This approach ensures that FSDPonly needs to materialize parameters from one unit at a time, whichsignificantly reduces peak memory consumption.





FSDP introducesdeferred initialization that allows users to create a model instanceon a dummy device and record operations invoked during initialization.Then, the model can be initialized and sharded unit by unit byreplaying the recorded operations on a real GPU device.



FSDP can squeezeout bubbles using an abundant set of tools to aggressively overlapcommunication with computation through **operation reorderingand parameter prefetching**.





Lastly, FSDP optimizes memory usageby prudently restricting the amount of blocks allocated for inflightunsharded parameters and suspending CPU execution if necessary 没懂







To expedite training, DDP overlaps gradient communicationwith backward computation 是做完一层的bwd后，将这层参数allreducema 





## 系统设计





### 初始化

模型如果在一张卡上放不下，如何无侵入式的让模型的一部分初始化在不同的rank上





#### Sharding

Full replication(DDP)

Hyprid 在大规模集群中，减少了通讯量？具体通讯量需要计算下。那和zero2相比呢？

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929010333.png)

FSDP



#### autograd

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929023257.png)





## 通信



DistributedDataParallel leveragesthe async-collective-and-wait() approach to overlap the gradientAll-Reduces with backward computation. （DDP如何做通算并行的？一部分梯度算完后，开始通信？）





一个重点是，CUDA的通信和计算是完全并行的，所以任何计算都是和通信并行。但是我们的芯片要分情况，有可能和cluster的冲突了，比如laynorm在算的时候，通信就不能算了（流的优先级有帮助吗）。。。这块到时候注意下。所以fsdp应该要看做是torch位cuda特地设计的fsdp，其他的硬件不一定适用。





![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929012257.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929012319.png)




图中0102而不是012应该是0是最外层的module，先算自己的一部分逻辑，然后1，然后再有自己的逻辑，然后2，按照文中的解释，0作为outermost层，参数在前向完之后不会被释放掉。当然，反向结束之后，还是会被释放掉



这里感觉要做个实验验证下，找个小的模型，算下内存占用

#### 通算如何并行


**Backward Prefetching.（重排RS AG顺序）**

考虑不并行的情况，以反向为例
AG2 BWD2 RS2 AG1 BWD1 RS2 ...

假设使用1个通信流，这里AG1在RS2后发射，那么BWD1必须等待AG1完成，也被延后了，计算和通信是完全串行的。

所以需要重排：
AG2 BWD2 AG1 RS2 BWD1 AG0 RS1 ....

AG2进行完后，AG1在通信流上马上执行，为BWD1准备参数；当BWD2执行完后，计算流上BWD1可以立即进行，此时RS2同时在进行，接着AG0执行，接着RS1；这里假设AG + RS的时间小于BWD的时间，否则计算流无法一直在运行

这里提到有一个挑战，就是如果知道下一个需要预先AG的FSDP module是哪个？如果不通算并行，那么自然在走到下一个module的时候，就知道是哪个module了
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929013731.png)

**Forward Prefetching.**

**gradient accu**
2种方式：
1. 正常的fsdp，每个fsdp unit的梯度算完后（此时是unsharded），RS到每个rank上，每个rank上都是sharded gradient
2. 空间换时间的fsdp，每个fsdp unit的梯度算完后（此时是unsharded），不做RS，每个rank持续持有unsharded gradient。但是优化器step前，应该还是要RS上吧，比较优化器状态是sharded的。论文里没写，但我觉得是这样的，需要结合源码看

#### 内存管理：


**rate limiter(GPU别跑太快)** 
假设我们考虑fwd，fwd的cpu会一直issue以下操作
AG1 COM1 AG2 COM2 AG3 ...
此时只是下发，没有进行实际的计算，但是AG下发的时候，是需要预先分配AG的dst tensor的，这会占据CUDA的内存；
想象一种情况，CPU跑的超级快，COM1开始执行的时候，AG1 到 AG100都下发完成了，需要100个AG dst tensor，这种情况下会造成OOM报错或者性能变差。

**为什么会造成性能变差呢？**当AG dst占据了很多显存时，可能某个COM执行的时候，pytorch CCA(Cuda Caching Allocator)没有足够大小的显存了，但不会直接报错，而是进行以下操作：
```c
//source : https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html
void return_our_possibly_fragmented_memory_to_cuda() {
    <wait for all cross stream events to complete> //为啥需要等待？
    <for all free blocks in our reserved memory, cudaFree them> //cudaFree开销较大
}
```
因为GPU存在页表，所以pytorch将显存还给cuda后，cuda能够将小块显存拼起来给cuda使用。pytorch又能够分配出显存了，OOM被避免。但是等待 + cudaFree开销较大，会导致性能下降。可以理解是soft OOM。

可以用3种方式观测这种现象（参考https://medium.com/@alex.isenko/why-utilizing-the-maximum-amount-of-memory-almost-never-leads-to-a-better-training-throughput-63ddbdf1585f，这块的` Non-deterministic Memory Allocations Within FSDP`这一节没有细看）：

1. 查看num_alloc_retries，如果一直在增长，显然就不对，因为当模型稳定运行时，不会出现大量的num_alloc_retries：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930001329.png)
2. 观察reserved memory大小，应该稳定，如果一直在波动，说明在持续的通过cudaFree向gpu归还显存，然后申请：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930001433.png)
3. 观察allocated_mem / reservered_mem大小，如果持续在减小，说明内存碎片化变严重了。但是我认为内存碎片化严重不一定是AG dst太多导致的，可能有其他因素
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930001539.png)


As expounded in Section 3.4, launching
AllGather too aggressively can lead to unnecessarily high memory
footprint, as the CPU thread needs to allocate CUDA memory blocks
when the communication kernel is added into the CUDA stream.
This predicament may sometimes result in significant performance
problems when the CPU thread runs too fast in comparison to
CUDA streams.



## 实现

#### 初始化

#### FlatParameter

一个FSDP unit对应一个FlatParameter，The FlatParameter class inherits from nn.Parameter and behaves like
an nn.Parameter. FSDP implements an accompanying FlatParamHandle
class that is responsible for managing individual FlatParameter instances.
The frontend, either FullyShardedDataParallel or fully_shard,
interfaces with the FlatParameters only through FlatParamHandle.

#### runtime

重写了nn.Module的forward，注册了一些hook，其中反向时使用了以下hook
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240929233356.png)

总的来说，没有对autograd进行侵入式修改，而是使用hook

#### amp
略


## 实验

#### 与DDP对比
一个担忧是，如果模型比较小，可以fit 单卡，那么使用fsdp，相比于ddp，会增加overhead，性能下降吗？
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930012048.png)
这幅图证明了不会，但是难理解的是，DDP反而是最慢的。

#### Bwd prefetching, AG & RS reorder
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930012240.png)

证明了性能提升非常明显，一定要开

#### Rate Limiter
注意，图中使用了latency，是与Token/GPU相反的指标
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930012426.png)
RateLimiter与模型相关，要看情况设定


#### RAF NRAF
• RAF: reshard-after-forward frees AllGathered shards from
other GPUs after forward pass and unshards them again
before backward computation. This reduces peak memory
consumption at the cost of higher communication overhead. 前向param AG后释放，bwd时再次AG

• NRAF: no-reshard-after-forward is the opposite where the
unsharded model parameters stay in GPU memory after
forward pass until backward computations finish, which
trades higher memory。前向param AG后不释放，而是bwd完成后释放，bwd时不需要AG，减少了bwd中的AG。

疑惑：gradient accu时，是一个mbs的bwd之后释放，还是所有bwd之后释放？

从下图来看，结点数少的时候，建议使用NRAF：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930012856.png)
还是实测看下吧


#### GPT175B 的scale up
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930013315.png)
注意到mbs=2时，在128结点上不如mbs=1。论文中观察到如下现象
Notably, with 128 GPUs, setting the batch size to
2 resulted in a considerably lower per-GPU TFLOPs in comparison
to other scenarios. This was due to CUDA memory defragmentation
during the backward pass. The backward pass contributed 85.56%
of the iteration latency for the 128 GPU batch size equals 2 case,
while a normal backward pass only accounted for about 67% in
these experiments. Using 128 GPUs is more likely to trigger defragmentation,
as each GPU needs to accommodate a larger model
shard. Figure 8 confirms this explanation, where the PyTorch CUDA
caching allocator depletes all 80GB of the CUDA memory as shown
on the top left corner.
反向占据了不符合预期的时间，论文认为是触发了soft oom，可以从曲线图中128nodes占满了80G。感觉如果能拿出num_alloc_retries数据，能够更好说明这个问题

结论，尽量使用较大的MBS，但要注意soft oom问题。

#### 集群规模上去后，fsdp还是影响性能
Finally, for T5-11B models as shown in Figure 8 (c), all experiments
are executed comfortably below GPU memory capacity,
where defragmentations are unlikely to happen. Nevertheless, as
the number of GPUs increases from 8 to 512, a 7% regression in
per-GPU TFLOPS is still evident as illustrated in Figure 7 (c). This
suggests that communications begin to outweigh computations on
large clusters, and a near-perfect overlap between communication
and computation is no longer attainable.

T5 11B是一个小模型，这种情况下，fsdp完美执行，但是节点数量8->512，性能下降了7%，说明性能还是会被影响。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240930013826.png)

## Related work
略

## 和其他并行方式的结合
略

