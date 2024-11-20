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

举例，当前conda环境中torch版本为2.1.2:
```
Name: torch
Version: 2.1.2+cu121
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /root/miniconda3/envs/old_mega/lib/python3.8/site-packages
Requires: filelock, fsspec, jinja2, networkx, sympy, triton, typing-extensions
Required-by: accelerate, flash_attn
```
此时安装pip install torchvision==0.16，pip会帮我们安装2.1.0，而不是使用已有的torch 2.1.2，这说明torchvision版本不匹配。
```
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting torchvision==0.16
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/c9/52/d3f1c4253ad17e4ab08a2230fb184a3a180e2348db6c144c64977335b654/torchvision-0.16.0-cp38-cp38-manylinux1_x86_64.whl (6.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.9/6.9 MB 3.6 MB/s eta 0:00:00
Requirement already satisfied: numpy in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16) (1.24.4)
Requirement already satisfied: requests in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16) (2.32.3)
Collecting torch==2.1.0 (from torchvision==0.16)
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/e1/24/f7fe3fe82583e6891cc3fceeb390f192f6c7f1d87e5a99a949ed33c96167/torch-2.1.0-cp38-cp38-manylinux1_x86_64.whl (670.2 MB)
     ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.2/670.2 MB 3.6 MB/s eta 0:02:59
ERROR: Operation cancelled by user
```
那换成安装torchvision 0.16.2，则torch不会被重装，说明torchvision版本匹配：
```
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting torchvision==0.16.2
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/3c/4e/8b8783baaa4dfef96de31fa4243b6367f931489ed840a38e0fa32230c5e4/torchvision-0.16.2-cp38-cp38-manylinux1_x86_64.whl (6.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 MB 5.9 MB/s eta 0:00:00
Requirement already satisfied: numpy in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16.2) (1.24.4)
Requirement already satisfied: requests in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16.2) (2.32.3)
Requirement already satisfied: torch==2.1.2 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16.2) (2.1.2+cu121)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torchvision==0.16.2) (10.4.0)
Requirement already satisfied: filelock in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (3.16.1)
Requirement already satisfied: typing-extensions in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (4.12.2)
Requirement already satisfied: sympy in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (1.13.3)
Requirement already satisfied: networkx in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (3.1)
Requirement already satisfied: jinja2 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (3.1.4)
Requirement already satisfied: fsspec in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (2023.10.0)
Requirement already satisfied: triton==2.1.0 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from torch==2.1.2->torchvision==0.16.2) (2.1.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from requests->torchvision==0.16.2) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from requests->torchvision==0.16.2) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from requests->torchvision==0.16.2) (2.2.3)
Requirement already satisfied: certifi>=2017.4.17 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from requests->torchvision==0.16.2) (2024.8.30)
Requirement already satisfied: MarkupSafe>=2.0 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from jinja2->torch==2.1.2->torchvision==0.16.2) (2.1.5)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /root/miniconda3/envs/old_mega/lib/python3.8/site-packages (from sympy->torch==2.1.2->torchvision==0.16.2) (1.3.0)
Installing collected packages: torchvision
Successfully installed torchvision-0.16.2
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
```


#### 3. 在conda中安装pytorch相关依赖时，保证没有设置CUDA_HOME，且环境中已有的/usr/local/cuda不在PATH LD_LIBRARY_PATH中

因为通过pip安装的pytorch 依赖的cudart、cusparse通过RPATH写死为conda安装的，如果LD_LIBRARY_PATH PATH中又定义了其他的cuda路径，虽然不会影响pytorch的安装，但可能影响其他pytorch相关包的安装，比如flash-attn grouped-gemm