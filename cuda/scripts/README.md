
## make block_schedule

当Block的个数小于SM个数时，或者Block个数不为SM个数的整数倍时，观察Block在各个SM上的分布情况。

## make block_limit_per_sm

由于deviceQuery缺少每个SM同时执行的最大Block数量，所以这里通过在每个SM上执行不同数量的Block，来观察这个值。

## cd deviceQuery; make run

Copy from [Nvidia/cuda-samples](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/README.md)，做了以下修改：
1. 去除头文件依赖，全部代码都放在deviceQuery.cpp这一个standalone的文件中
2. 去除sm_89 sm_90编译选项，因为我用的nvcc版本为11.7，还不支持。
3. deviceQuery 重命名为 deviceQuery.exe，能够被gitignore分辨出来