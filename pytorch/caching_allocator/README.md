https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html

几点有意思的：
1. cacheingalloctor申请不出大块的内存了，但是将碎片化的没在用的显存cudaFree后，cudaMalloc还是能分配到内存，这是因为cudaMalloc会被驱动虚拟化，就算是物理上不连续的内存，也能被分配成连续的大块内存。
但是不是所有碎片化的显存都能够被释放，比如申请了一块10M内存，其中前5M正在使用，另外5M被split出来等待后续使用；这5M不能单独释放，cudaFree只接受原始的10M内存
2. 


## cudaMalloc cudaFree耗时分析

`nvcc -arch=sm_80 profile.cpp -o profile.exe; ./profile.exe`

分别以以下单位分配、释放100次：

1. 4B

2. 4K

3. 4M

4. 240M

|  | cudaMalloc(ms) | cudaFree(ms) |
|----------|----------|----------|
| 1B | 0.00411471| 0.00000036 |
| 1K | 0.00696423| 0.00000027 |
| 1M | 0.13539373 | 0.00000036 |
| 240M | 0.37568712 | 0.00000054 |

当申请的内存较大时，每次cudaMalloc的时间达到毫秒级别，如果没有cachingallocator，则性能开销还是比较大的。

需要注意的是，第一次cudaMalloc时还会触发lazyinit，大概需要2～5秒的时间。

## Megatron怎么做sharding?