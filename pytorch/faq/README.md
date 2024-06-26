# Error of PyTorch


## Compile & Link Error

#### 1. 
```
Traceback (most recent call last):
  File "leaky.py", line 8, in <module>
    import torch
  File "/home/github/pytorch/torch/__init__.py", line 237, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/github/pytorch/torch/lib/libtorch_python.so: undefined symbol: PyObject_GC_IsTracked
```
使用的python版本和torch编译时的版本不一致，跨版本有些符号会增减。

#### 2
**Error:**
CMake Error at cmake/public/cuda.cmake:47 (enable_language):
  No CMAKE_CUDA_COMPILER could be found.

  Tell CMake where to find the compiler by setting either the environment
  variable "CUDACXX" or the CMake cache entry CMAKE_CUDA_COMPILER to the full
  path to the compiler, or to the compiler name if it is in the PATH.
Call Stack (most recent call first):
  cmake/Dependencies.cmake:44 (include)
  CMakeLists.txt:761 (include)

**Why:**

Missing nvcc:
```
(torch_src) ➜  pytorch nvcc      
zsh: command not found: nvcc
```

**Solution:**
add nvcc to PATH; add lib to LD_LIBRARY_PATH:
```
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

## 2. torch, torchvision 版本对应关系
大版本的对应关系见：https://github.com/pytorch/vision?tab=readme-ov-file#installation

比如安装了torch 2.1，那么torchvision 应该安装对应的0.16；否则torchvision会自动升级/降级torch。

小版本没有列在表中，只能尝试，比如torch 2.0.1对应torchvision 0.15.2；0.15.1 0.15.0 都对应torch 2.0.0，不能使用