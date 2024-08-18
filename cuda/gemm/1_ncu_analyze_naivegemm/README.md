
在学习 [CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://zhuanlan.zhihu.com/p/518857175)和对应[代码](https://github.com/nicolaswilde/cuda-sgemm/tree/main)的过程中，尝试手写naiveSgemm时，发现我的版本和原始版本的naiveSgemm性能相差十倍：

```
原始版本的naiveSgemm
===================== gemm function: naive_contiguous_gemm =====================
M N K =    128    128   1024, Max Error = 0.0000610, Time =   0.00010650   0.00011287   0.00013734 s, AVG Performance =   276.8662 Gflops
M N K =    192    192   1024, Max Error = 0.0000916, Time =   0.00010752   0.00010806   0.00011024 s, AVG Performance =   650.6561 Gflops
M N K =    256    256   1024, Max Error = 0.0000916, Time =   0.00019661   0.00019814   0.00020122 s, AVG Performance =   630.8543 Gflops
M N K =    384    384   1024, Max Error = 0.0000916, Time =   0.00028774   0.00028996   0.00029267 s, AVG Performance =   969.9667 Gflops
M N K =    512    512   1024, Max Error = 0.0000916, Time =   0.00047104   0.00047338   0.00047658 s, AVG Performance =  1056.2428 Gflops
M N K =    768    768   1024, Max Error = 0.0000916, Time =   0.00102093   0.00102235   0.00102496 s, AVG Performance =  1100.4072 Gflops
M N K =   1024   1024   1024, Max Error = Skip(太慢) Time =   0.00173261   0.00173384   0.00173466 s, AVG Performance =  1153.5111 Gflops
M N K =   1536   1536   1024, Max Error = Skip(太慢) Time =   0.00381952   0.00383037   0.00383488 s, AVG Performance =  1174.8199 Gflops
M N K =   2048   2048   1024, Max Error = Skip(太慢) Time =   0.00672973   0.00673362   0.00673894 s, AVG Performance =  1188.0684 Gflops
M N K =   3072   3072   1024, Max Error = Skip(太慢) Time =   0.01069158   0.01258783   0.01542963 s, AVG Performance =  1429.9529 Gflops
M N K =   4096   4096   1024, Max Error = Skip(太慢) Time =   0.01993933   0.01997619   0.02000282 s, AVG Performance =  1601.9069 Gflops
M N K =   6144   6144   1024, Max Error = Skip(太慢) Time =   0.04916327   0.04960195   0.05120000 s, AVG Performance =  1451.5559 Gflops
M N K =   8192   8192   1024, Max Error = Skip(太慢) Time =   0.08842854   0.08854589   0.08858726 s, AVG Performance =  1445.5781 Gflops
M N K =  12288  12288   1024, Max Error = Skip(太慢) Time =   0.19619124   0.19667436   0.19684659 s, AVG Performance =  1464.3495 Gflops
M N K =  16384  16384   1024, Max Error = Skip(太慢) Time =   0.35115010   0.35120148   0.35128832 s, AVG Performance =  1457.8526 Gflops

我写的naiveSgemm
===================== gemm function: naive_incontiguous_gemm =====================
M N K =    128    128   1024, Max Error = 0.0000916, Time =   0.00076493   0.00076662   0.00076931 s, AVG Performance =    40.7631 Gflops
M N K =    192    192   1024, Max Error = 0.0000610, Time =   0.00076698   0.00076780   0.00076906 s, AVG Performance =    91.5764 Gflops
M N K =    256    256   1024, Max Error = 0.0000610, Time =   0.00152576   0.00152660   0.00152893 s, AVG Performance =    81.8814 Gflops
M N K =    384    384   1024, Max Error = 0.0000916, Time =   0.00228659   0.00228770   0.00229008 s, AVG Performance =   122.9401 Gflops
M N K =    512    512   1024, Max Error = 0.0000916, Time =   0.00380314   0.00380520   0.00380733 s, AVG Performance =   131.3990 Gflops
M N K =    768    768   1024, Max Error = 0.0000916, Time =   0.01294746   0.01294880   0.01295213 s, AVG Performance =    86.8806 Gflops
M N K =   1024   1024   1024, Max Error = Skip(太慢) Time =   0.02234061   0.02234143   0.02234266 s, AVG Performance =    89.5198 Gflops
M N K =   1536   1536   1024, Max Error = Skip(太慢) Time =   0.03191296   0.03210220   0.03284889 s, AVG Performance =   140.1773 Gflops
M N K =   2048   2048   1024, Max Error = Skip(太慢) Time =   0.05617766   0.05617869   0.05617971 s, AVG Performance =   142.4028 Gflops
M N K =   3072   3072   1024, Max Error = Skip(太慢) Time =   0.12535807   0.12535910   0.12536013 s, AVG Performance =   143.5875 Gflops
M N K =   4096   4096   1024, Max Error = Skip(太慢) Time =   0.22240563   0.22240666   0.22240768 s, AVG Performance =   143.8806 Gflops
M N K =   6144   6144   1024, Max Error = Skip(太慢) Time =   0.50062031   0.50062376   0.50063050 s, AVG Performance =   143.8206 Gflops
M N K =   8192   8192   1024, Max Error = Skip(太慢) Time =   0.88881969   0.88882298   0.88882691 s, AVG Performance =   144.0107 Gflops
M N K =  12288  12288   1024, Max Error = Skip(太慢) Time =   2.00096869   2.00097609   2.00098300 s, AVG Performance =   143.9298 Gflops
M N K =  16384  16384   1024, Max Error = Skip(太慢) Time =   3.55376339   3.55379081   3.55383396 s, AVG Performance =   144.0715 Gflops
```

分析代码差异，唯一区别在于m、n的计算方式不一样：
```cpp
//原始版本
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

//我的版本
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
```
由于分块方式为`dim3 blockDim(BN/*blockDim.x*/, BM/*blockDim.y*/)`; 所以`blockIdx.x * blockDim.x + threadIdx.x;`对应维度为n，而`blockIdx.y * blockDim.y + threadIdx.y;`对应维度为m。原始版本正确，而我的版本错误。

不过，由于测试规模中 M == N，这种规模小我的版本是正确的，但性能下降了10倍。接下来尝试分析原因。

## ncu分析

我尝试基于PyTorch的Docker下载&安装nsight compute，但失败了。linux推荐使用nvidia的docker，`nvidia/cuda:11.7.1-devel-ubuntu20.04`，自带ncu。

mac GUI从官网下载：https://developer.nvidia.com/tools-overview/nsight-compute/get-started

#### 计算、访存 吞吐
蓝色为我的版本，绿色为原始版本。我的版本计算吞吐很低，而访存很高，说明访存瓶颈。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818203952.png)

#### ncu中的相同之处：Instruction/Requests

固定`M,N,K=1024`, 用ncu在linux下抓出，在mac上的NVIDIA Nsight Compute中分析。


先分析两个版本的相同之处：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818183458.png)

1. L1 cache上的Global Load Instruction/Requests 都为67108864。回顾kernel代码，只有这一行涉及了Global Load:
```cpp
sum += a[OFFSET(m,k,K)] * b[OFFSET(k,n,N)]; //这里有2条Global Load指令
```
由于M=N=K=1024，所以Global Load（Instruction/Requests）一共发生了$M*N*K/(num Thread In Warp)=1024*1024*1024*2/32=67108864$。

统计Instruction时要除以32，是因为warp内的32个线程共享同一条Instruction；统计Request时，也要除以32，因为无论warp内的32个thread的内存访问模式（顺序，strided，random），1个warp只会发出一条request给到L1 cache。所以Instruction和Requests都是67108864。

2. L1 cache上的Global Store Instruction/Requests 都为32768
回顾kernel代码，只有这一行涉及了Global Load:
```cpp
c[OFFSET(m,n,N)] = sum;
```
由于M=N=K=1024，所以Global Store（Instruction/Requests）一共发生了$M*N/(num Thread In Warp)=1024*1024/32=32768$。

#### ncu中的不同之处：Wavefront
接着分析不同之处，并试图找到性能差异点，这里先看一个概念：Wavefront
> Wavefront: Number of unique “work packages” generated at the end of the processing stage for requests. All work items of a wavefront are processed in parallel, while work items of different wavefronts are serialized and processed on different cycles.

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818184754.png)
L1 cache中存在一条global load流水线，当Tag Stage接收到global load Request时，会将Request拆解为不同的Wavefront，load流水线1个cycle只能处理一个wavefront，TagStage会逐个向流水线后续部分发出拆解后的Wavefront；如果request中的memory没有coalescing，那么会被拆分为多个wavefront，每个wavefront需要1个cycle，那么这个warp的request会花费更多的cycle会完成，自然会变慢。（Wavefront的视频解释见[video](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s32089/) 9:15处）

先看下原始版本的Wavefront：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818190041.png)

Store/Load的Wavefront和Request数目一样，说明每个Request做到了memory coalesce足够简单，不需要拆分。

再看我的版本的Wavefront：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818190251.png)

Store的Wavefront是Request数目的3.5倍，说明平均每个Store Request拆分成了3.5个Wavefront。

Load的Wavefront是Request数目72倍，说明平均每个Load Request拆分成了7个Wavefront。

而L1 cache一次只能处理一个Wavefront，这解释了我的版本显著慢于原始版本的原因。

另一处不同为Sectors/Req，我的版本为16.5而原始版本为2.5，这里的原因稍后再说。

#### 分析代码
从ncu的分析结果，我们发现warp的Request的质量不高，每个Request产生了多个Wavefront，导致L1 cache需要更多的cycle来处理。那接下来分析代码。

还是以`M=N=K=1024`，为了简化分析，我们只分析第0个block的第0个warp。先分析我的版本:
```cpp
m = blockIdx.x * blockDim.x + threadIdx.x;
n = blockIdx.y * blockDim.y + threadIdx.y;
```

由于同一个block内的thread按照`threadIdx.x`递增来划分（[参考](https://stackoverflow.com/a/6177473)），所以warp内的32个thread的m、n分别为：
- thread 0: m=0, n=0
- thread 1: m=1, n=0
- thread 2: m=2, n=0
- ......
- thread 31: m=31, n=0

2处Global Load分别为：
- `a[OFFSET(m,k,K)]=a[m*K+k]` 
- `b[OFFSET(k,n,N)]=b[k*N+n]`

第0个block的第0个warp在k=0时，load访问的Global Memory地址为:
- thread 0: m=0, n=0, global load的地址为`a[0], b[0]`
- thread 1: m=1, n=0, global load的地址为`a[K], b[0]`
- thread 2: m=2, n=0, global load的地址为`a[2K], b[0]`
- ......
- thread 31: m=31, n=0, global load的地址为`a[31K], b[0]`

第0个block的第0个warp在k=1时，load访问的Global Memory地址为:
- thread 0: m=0, n=0, global load的地址为`a[1], b[1]`
- thread 1: m=1, n=0, global load的地址为`a[K+1], b[1]`
- thread 2: m=2, n=0, global load的地址为`a[2K+1], b[1]`
- ......
- thread 31: m=31, n=0, global load的地址为`a[31K+1], b[1]`

同一时刻，warp内的32个thread对数组a的访问的stride为K，对数组b的访问stride为0。当stride=0时，这种最简单的内存访问pattern只需要一个wavefront即可完成。而剩余的wavefront为 `(total_load_wavefront - stride_0_wavefront) = 301990397 - (1024*1024*1024/32) = 301990397 = 301990397 - 33554432 = 268435965`。而 268435965/33554432=8.00001517，33554432为数组a的load的次数，（这里不太清楚为什么不能完全整除，但不影响结论）。

即平均而言，a的每个load请求需要8个wavefront。

Global Store的分析类似:
- thread 0: m=0, n=0, global store的地址为`c[0]`
- thread 1: m=1, n=0, global store的地址为`c[N]`
- thread 2: m=2, n=0, global store的地址为`c[2N]`
- ......
- thread 31: m=31, n=0, global store的地址为`c[32N]`

平均而言，每个store request需要`total_store_wavefront/store_request=262156/32768=8.00036621`个wavefront。

#### ncu中的不同之处：Sector, Sectors/Req
再看一个概念：Sector。GM->L2, L2->L1, L1->SM（存疑？，参考https://stackoverflow.com/a/63499878）的最小单位为sector=32B

> 疑惑，L1 L2 cacheline为128B，cacheline如何再细分为sector？如何区分cacheline中的4个sector？

当SM发起一次load request时，warp里的每个thread访问的地址会对齐到32B，并汇总为32B的sector。比如我的版本，第0个block的第0个warp在k=0时，load访问的Global Memory地址为:
- thread 0: m=0, n=0, global load a的sector为`a[0]`开始的32B, global load b的sector为`b[0]`开始的32B, 
- thread 1: m=1, n=0, global load a的sector为`a[1]`开始的32B, global load b的sector为`b[0]`开始的32B,
- ......

global load a的一次request包含32个sector，共计`(1024*1024*1024/32) * 32 = 1073741824`，global load b的一次request包含1个sector，共计`(1024*1024*1024/32) = 33554432`。总global load sectors为`1073741824 + 33554432 = 1107296256`：

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818202617.png)

再来看下Sectors/Req这个指标 `1107296256 / (1024*1024*1024/32*2)[load a + load b次数]= 16.5`。因为global load a 该指标为32，global load b该指标为1，平均一下就是16.5.

> Sectors/Req 除了通过GUI来看，也可以命令行中用`ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio ./ut.exe`查看，其中`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`为ncu/硬件实现时这项统计的代号）


一般而言，一个warp 32个thread，假设每个thread需要4B(1个float)，那么1个warp需要32*4B=128B，即4个sector。所以Sectors/Req一般最佳为4。而我的版本为16.5，这也侧面反映了global load没有coalesce。


再分析下原始版本的代码，第0个block的第0个warp在k=0时，load访问的Global Memory地址为:
- thread 0: m=0, n=0, global load的地址为`a[0], b[0]`, global store的地址为`c[0]`
- thread 1: m=0, n=1, global load的地址为`a[0], b[1]`, global store的地址为`c[1]`
- thread 2: m=0, n=2, global load的地址为`a[0], b[2]`, global store的地址为`c[2]`
- ......
- thread 31: m=0, n=31, global load的地址为`a[0], b[31]`, global store的地址为`c[31]`

b[0], b[1], b[2], ..., b[31]是物理连续的地址，所以load b的Sectors/Req为4，load a的Sectors/Req为1，平均为2.5。Request也不用拆分为多个wavefront，Request和wavefront数量相等。

global store的分析也类似，c[0], c[1], c[2], ..., c[31]是物理连续的地址，所以store的Sectors/Req为4。Request也不用拆分为多个wavefront，Request和wavefront数量相等。

详细结果见下图

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818203410.png)


## 图示
通过2张图可以更清楚的看到原始版本和我的版本的访存pattern的区别

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818215545.png)

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240818215603.png)

## 其他问题：printf limit
在kernel中进行打印的时候，期望每个thread打印32行log，但编译执行后发现，只有后14行；而前面18行丢失了。向GPT询问：

> 在CUDA kernel中使用printf进行调试时，可能会遇到输出不完全的情况，特别是当输出量较大时。这通常是因为CUDA的printf缓存大小有限制，超过这个限制的输出可能会被截断或丢失。以下是一些解决此问题的建议：
> 1. 增加printf缓存大小
> 
> NVIDIA CUDA提供了一个运行时API函数cudaDeviceSetLimit，可以用来调整printf缓存的大小。你可以在程序中增加这个调用来尝试解决问题：
> ```
> #include <cuda_runtime.h>
> cudaError_t error = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024); #10MB
> if (error != cudaSuccess) {
>     printf("Failed to set printf FIFO size: %s\n", cudaGetErrorString(error));
> }
> ```
> 其中size_in_bytes是你希望设置的缓存大小，单位是字节。你可以尝试将其设置得更大，例如10MB（10485760字节）。
> 
> 4. 同步设备
> 确保在读取输出之前，所有CUDA kernel都已经完成执行。可以在打印之后调用cudaDeviceSynchronize()来同步设备：
> 