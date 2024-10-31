

## 1.编译mpirun

> 如果是测单机的机内带宽，可以不用安装mpirun

### 安装依赖
每个人的环境不同，遵循一个基本的原则，缺什么装什么。比如[1]只用装第一行，但是我的环境还需要装event hwloc ucx
```
sudo apt-get update
sudo apt-get install libpmix-dev libfabric-dev libev-dev zlib1g-dev
sudo apt-get install libevent-dev libhwloc-dev
```
接下来要安装ucx，否则configure会报这个错：
```
checking for ucp/api/ucp.h… no
configure: WARNING: UCX version is too old, please upgrade to 1.9 or higher.
configure: error: UCX support requested but not found. Aborting
```

这个依赖比较奇怪，apt源上找不到（也可能是我没仔细找），所以我跑到[github主页](https://github.com/openucx/ucx/releases/tag/v1.17.0)下了一个预编译好的deb包：
```sh
wget https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0-ubuntu20.04-mofed5-cuda12-x86_64.tar.bz2
tar xvjf ucx-1.17.0-ubuntu20.04-mofed5-cuda12-x86_64.tar.bz2 
#产生了如下3个包，我只安装了前面2个
ucx-cuda-1.17.0.deb ucx-1.17.0.deb  ucx-xpmem-1.17.0.deb
dpkg -i ucx-cuda-1.17.0.deb
dpkg -i ucx-1.17.0.deb
```

### 编译

```
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
tar xzf openmpi-5.0.5.tar.gz
cd openmpi-5.0.5/
```

这里使用了[1]中的configure命令：
```
# Note: Installing to /opt/openmpi in case we need to remove later.
#       This will require us to export $PATH and $LD_LIBRARY_PATH next
./configure --prefix=/opt/openmpi \
   --enable-mpi-ext=cuda \
   --with-cuda=/usr/local/cuda \
   --with-cuda-libdir=/usr/local/cuda/lib64 \
   --with-ucx --with-libfabric --enable-builtin-atomics \
   --without-cma --with-libevent=external \
   --with-hwloc=external --disable-silent-rules \
   --enable-ipv6 --with-devel-headers \
   --with-slurm --with-sge --without-tm --with-zlib \
   --enable-heterogeneous 2>&1 | tee config.out

make -j 72 all 2>&1 | tee make.out
sudo make install 2>&1 | tee install.out
```

```
export PATH=/opt/openmpi/bin/:$PATH
export LD_LIBRARY_PATH=/opt/openmpi/lib/:$LD_LIBRARY_PATH
```


## 2. 编译nccl-tests

### 安装依赖

nccl-tests依赖nccl，目前有3种办法安装nccl：
1. 源码编译
2. `sudo apt-get install libnccl-dev`
3. `conda create -n nccl python=3.9; ...; pip install nvidia-nccl-cu12`

我选择了第3种方式，可以用conda环境来隔离不同版本的nccl。如果是用这种方式，还有额外的一步需要做：
```
cd site-packages/nvidia/nccl
ln -s libnccl.so.2 libnccl.so
```
因为nccl编译时使用`-L/root/miniconda3/envs/nccl/lib/python3.9/site-packages/nvidia/nccl/lib -lnccl` 查找`libnccl.so`，如果是libnccl.so.2，则找不到。

### 编译

```sh
make -j 72 MPI=1 MPI_HOME=/opt/openmpi CUDA_HOME=/usr/local/cuda/ NCCL_HOME=/root/miniconda3/envs/nccl/lib/python3.9/site-packages/nvidia/nccl 
```

这一步如果遇到问题，可以直接在Makefile中通过echo来debug，简单直观。感谢nccl，没用又臭又长的cmake来配置。

## 3. benchmark

配置nccl环境变量
```
conda activate nccl (conda自动设置，等价于下面的方法)
或
export LD_LIBRARY_PATH=/root/miniconda3/envs/nccl/lib/python3.9/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

需要保证GPU空闲，否则测出来的带宽会受严重影响


### 理论带宽
通过`nvidia-smi nvlink -s`能看到GPU的理论带宽为 25GB * 12 = 300GB/s
```
GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-7eeee076-1b59-1cb1-6ac0-6070eca44f7d)
         Link 0: 25 GB/s
         Link 1: 25 GB/s
         Link 2: 25 GB/s
         Link 3: 25 GB/s
         Link 4: 25 GB/s
         Link 5: 25 GB/s
         Link 6: 25 GB/s
         Link 7: 25 GB/s
         Link 8: 25 GB/s
         Link 9: 25 GB/s
         Link 10: 25 GB/s
         Link 11: 25 GB/s
...
```
根据https://github.com/NVIDIA/nccl/issues/1054#issuecomment-1794557164 去除NVLink overhead，实际busbw约为300 * 80% = 240GB/s。

什么是算法带宽(algobw)和硬件带宽(busbw):https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

### 实测带宽

单机8卡A100，随着数据量上升，逐渐接近理论上限240GB/s

```
./build/all_reduce_perf -b 8 -e 4G -f 2 -g 8      
# nThread 1 nGpus 8 minBytes 8 maxBytes 4294967296 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  54740 on     thanos device  0 [0x07] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid  54740 on     thanos device  1 [0x0a] NVIDIA A100-SXM4-80GB
#  Rank  2 Group  0 Pid  54740 on     thanos device  2 [0x45] NVIDIA A100-SXM4-80GB
#  Rank  3 Group  0 Pid  54740 on     thanos device  3 [0x4b] NVIDIA A100-SXM4-80GB
#  Rank  4 Group  0 Pid  54740 on     thanos device  4 [0x84] NVIDIA A100-SXM4-80GB
#  Rank  5 Group  0 Pid  54740 on     thanos device  5 [0x8a] NVIDIA A100-SXM4-80GB
#  Rank  6 Group  0 Pid  54740 on     thanos device  6 [0xc0] NVIDIA A100-SXM4-80GB
#  Rank  7 Group  0 Pid  54740 on     thanos device  7 [0xc3] NVIDIA A100-SXM4-80GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    40.12    0.00    0.00      0    38.85    0.00    0.00      0
          16             4     float     sum      -1    39.64    0.00    0.00      0    39.19    0.00    0.00      0
          32             8     float     sum      -1    39.06    0.00    0.00      0    38.67    0.00    0.00      0
          64            16     float     sum      -1    38.89    0.00    0.00      0    39.05    0.00    0.00      0
         128            32     float     sum      -1    39.53    0.00    0.01      0    39.73    0.00    0.01      0
         256            64     float     sum      -1    39.08    0.01    0.01      0    39.48    0.01    0.01      0
         512           128     float     sum      -1    39.17    0.01    0.02      0    39.43    0.01    0.02      0
        1024           256     float     sum      -1    42.05    0.02    0.04      0    39.65    0.03    0.05      0
        2048           512     float     sum      -1    38.39    0.05    0.09      0    39.41    0.05    0.09      0
        4096          1024     float     sum      -1    39.12    0.10    0.18      0    39.46    0.10    0.18      0
        8192          2048     float     sum      -1    39.39    0.21    0.36      0    40.05    0.20    0.36      0
       16384          4096     float     sum      -1    39.47    0.42    0.73      0    39.43    0.42    0.73      0
       32768          8192     float     sum      -1    40.93    0.80    1.40      0    40.26    0.81    1.42      0
       65536         16384     float     sum      -1    40.97    1.60    2.80      0    41.18    1.59    2.79      0
      131072         32768     float     sum      -1    42.26    3.10    5.43      0    42.48    3.09    5.40      0
      262144         65536     float     sum      -1    43.97    5.96   10.43      0    44.58    5.88   10.29      0
      524288        131072     float     sum      -1    48.31   10.85   18.99      0    48.73   10.76   18.83      0
     1048576        262144     float     sum      -1    52.61   19.93   34.88      0    53.03   19.77   34.60      0
     2097152        524288     float     sum      -1    59.30   35.37   61.89      0    71.33   29.40   51.45      0
     4194304       1048576     float     sum      -1    103.8   40.40   70.70      0    102.8   40.81   71.41      0
     8388608       2097152     float     sum      -1    162.6   51.58   90.27      0    161.5   51.93   90.88      0
    16777216       4194304     float     sum      -1    260.5   64.41  112.72      0    260.0   64.53  112.92      0
    33554432       8388608     float     sum      -1    439.2   76.41  133.71      0    441.9   75.94  132.89      0
    67108864      16777216     float     sum      -1    695.5   96.49  168.86      0    695.2   96.53  168.93      0
   134217728      33554432     float     sum      -1   1305.0  102.85  179.99      0   1312.3  102.27  178.98      0
   268435456      67108864     float     sum      -1   2392.6  112.19  196.34      0   2393.0  112.18  196.31      0
   536870912     134217728     float     sum      -1   4627.5  116.02  203.03      0   4250.9  126.30  221.02      0
  1073741824     268435456     float     sum      -1   8247.6  130.19  227.83      0   8240.7  130.30  228.02      0
  2147483648     536870912     float     sum      -1    16330  131.51  230.13      0    16349  131.35  229.87      0
  4294967296    1073741824     float     sum      -1    32336  132.82  232.44      0    32455  132.34  231.59      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 66.2058 
#
```

也可以使用mpirun来进行单机测试（mpirun多用于多机测试），和上面的结果相近：

```
mpirun --oversubscribe --allow-run-as-root -np 8 -N 8 -host localhost ./build/all_reduce_perf -b 8 -e 4G -f 2 -g 1
# nThread 1 nGpus 1 minBytes 8 maxBytes 4294967296 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid  56543 on     thanos device  0 [0x07] NVIDIA A100-SXM4-80GB
#  Rank  1 Group  0 Pid  56544 on     thanos device  1 [0x0a] NVIDIA A100-SXM4-80GB
#  Rank  2 Group  0 Pid  56545 on     thanos device  2 [0x45] NVIDIA A100-SXM4-80GB
#  Rank  3 Group  0 Pid  56546 on     thanos device  3 [0x4b] NVIDIA A100-SXM4-80GB
#  Rank  4 Group  0 Pid  56547 on     thanos device  4 [0x84] NVIDIA A100-SXM4-80GB
#  Rank  5 Group  0 Pid  56548 on     thanos device  5 [0x8a] NVIDIA A100-SXM4-80GB
#  Rank  6 Group  0 Pid  56549 on     thanos device  6 [0xc0] NVIDIA A100-SXM4-80GB
#  Rank  7 Group  0 Pid  56550 on     thanos device  7 [0xc3] NVIDIA A100-SXM4-80GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
8 more processes have sent help message help-mtl-ofi.txt / OFI call fail
           8             2     float     sum      -1    20.26    0.00    0.00      0    17.74    0.00    0.00      0
          16             4     float     sum      -1    17.90    0.00    0.00      0    17.77    0.00    0.00      0
          32             8     float     sum      -1    20.75    0.00    0.00      0    20.29    0.00    0.00      0
          64            16     float     sum      -1    22.30    0.00    0.01      0    23.21    0.00    0.00      0
         128            32     float     sum      -1    23.91    0.01    0.01      0    24.08    0.01    0.01      0
         256            64     float     sum      -1    23.61    0.01    0.02      0    23.37    0.01    0.02      0
         512           128     float     sum      -1    24.36    0.02    0.04      0    23.51    0.02    0.04      0
        1024           256     float     sum      -1    24.43    0.04    0.07      0    23.63    0.04    0.08      0
        2048           512     float     sum      -1    24.32    0.08    0.15      0    23.38    0.09    0.15      0
        4096          1024     float     sum      -1    25.29    0.16    0.28      0    24.85    0.16    0.29      0
        8192          2048     float     sum      -1    25.84    0.32    0.55      0    24.30    0.34    0.59      0
       16384          4096     float     sum      -1    26.15    0.63    1.10      0    25.67    0.64    1.12      0
       32768          8192     float     sum      -1    30.25    1.08    1.90      0    29.67    1.10    1.93      0
       65536         16384     float     sum      -1    31.82    2.06    3.60      0    30.24    2.17    3.79      0
      131072         32768     float     sum      -1    31.90    4.11    7.19      0    28.97    4.52    7.92      0
      262144         65536     float     sum      -1    27.65    9.48   16.59      0    26.18   10.01   17.53      0
      524288        131072     float     sum      -1    28.11   18.65   32.64      0    26.79   19.57   34.24      0
     1048576        262144     float     sum      -1    37.89   27.67   48.43      0    38.57   27.19   47.58      0
     2097152        524288     float     sum      -1    55.39   37.86   66.26      0    55.32   37.91   66.34      0
     4194304       1048576     float     sum      -1    81.66   51.36   89.89      0    94.98   44.16   77.28      0
     8388608       2097152     float     sum      -1    130.6   64.21  112.37      0    127.8   65.62  114.84      0
    16777216       4194304     float     sum      -1    213.3   78.67  137.68      0    216.2   77.60  135.81      0
    33554432       8388608     float     sum      -1    377.6   88.87  155.53      0    377.7   88.85  155.48      0
    67108864      16777216     float     sum      -1    615.7  108.99  190.73      0    614.8  109.16  191.03      0
   134217728      33554432     float     sum      -1   1181.6  113.59  198.78      0   1187.8  113.00  197.75      0
   268435456      67108864     float     sum      -1   2218.2  121.02  211.78      0   2217.6  121.05  211.84      0
   536870912     134217728     float     sum      -1   4248.8  126.36  221.13      0   4246.2  126.44  221.26      0
  1073741824     268435456     float     sum      -1   8244.5  130.24  227.91      0   8252.7  130.11  227.69      0
  2147483648     536870912     float     sum      -1    16288  131.84  230.72      0    16294  131.79  230.64      0
1 more process has sent help message help-mtl-ofi.txt / OFI call fail
  4294967296    1073741824     float     sum      -1    32285  133.03  232.81      0    32268  133.10  232.93      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 72.7724 
#
```

多机环境缺失，暂不测试。

## 参考资料

1. https://medium.com/@ed.sealing/multi-node-gh200-nccl-testing-dc2fc64d97a0
