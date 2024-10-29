Pytorch是怎样链接NCCL的？一般人的想法肯定是动态链接，因为NCCL的接口一般固定，ncclAllReduce，ncclAllGather等。但其实具体要分情况讨论，不同途径安装的torch，so的依赖方式不同。

## 1. pip install torch==2.1.2
假设直接安装pytorch
```sh
>  pip install torch==2.1.2
Collecting torch==2.1.2
  Downloading torch-2.1.2-cp39-cp39-manylinux1_x86_64.whl.metadata (25 kB)
Requirement already satisfied: filelock in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (3.16.1)
Requirement already satisfied: typing-extensions in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (4.12.2)
Requirement already satisfied: sympy in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (1.13.3)
Requirement already satisfied: networkx in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (3.2.1)
Requirement already satisfied: jinja2 in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (3.1.4)
Requirement already satisfied: fsspec in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (2024.10.0)
Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch==2.1.2)
  Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch==2.1.2)
  Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch==2.1.2)
  Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch==2.1.2)
  Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cublas-cu12==12.1.3.1 (from torch==2.1.2)
  Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cufft-cu12==11.0.2.54 (from torch==2.1.2)
  Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-curand-cu12==10.3.2.106 (from torch==2.1.2)
  Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch==2.1.2)
  Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch==2.1.2)
  Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
Collecting nvidia-nccl-cu12==2.18.1 (from torch==2.1.2)
  Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl.metadata (1.8 kB)
Collecting nvidia-nvtx-cu12==12.1.105 (from torch==2.1.2)
  Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
Requirement already satisfied: triton==2.1.0 in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from torch==2.1.2) (2.1.0)
Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch==2.1.2)
  Using cached nvidia_nvjitlink_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
Requirement already satisfied: MarkupSafe>=2.0 in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from jinja2->torch==2.1.2) (3.0.2)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /root/miniconda3/envs/test2/lib/python3.9/site-packages (from sympy->torch==2.1.2) (1.3.0)
Downloading torch-2.1.2-cp39-cp39-manylinux1_x86_64.whl (670.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 670.2/670.2 MB 33.9 MB/s eta 0:00:00
Using cached nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
Using cached nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
Using cached nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
Using cached nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 34.9 MB/s eta 0:00:00
Using cached nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
Using cached nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
Using cached nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
Using cached nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
Downloading nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl (209.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.8/209.8 MB 39.5 MB/s eta 0:00:00
Using cached nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
Using cached nvidia_nvjitlink_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl (19.7 MB)
Installing collected packages: nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch
Successfully installed nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.18.1 nvidia-nvjitlink-cu12-12.6.77 nvidia-nvtx-cu12-12.1.105 torch-2.1.2
```

注意到安装了依赖库 `nvidia-nccl-cu12`
```
 ~/miniconda3/envs  pip show nvidia-nccl-cu12   
Name: nvidia-nccl-cu12
Version: 2.18.1
Summary: NVIDIA Collective Communication Library (NCCL) Runtime
Home-page: https://developer.nvidia.com/cuda-zone
Author: Nvidia CUDA Installer Team
Author-email: cuda_installer@nvidia.com
License: NVIDIA Proprietary Software
Location: /root/miniconda3/envs/test2/lib/python3.9/site-packages
Requires: 
Required-by: torch
```

对应的libnccl.so不在`site-packages/nvidia_nccl_cu12-2.21.5.dist-info`，而是和其他nvidia的依赖库一样，被统一放在`site-packages/nvidia`目录下下管理，比如nccl被安装到了 `site-packages/nvidia/nccl` 中
```sh
 ~/miniconda3/envs/test/lib/python3.9/site-packages/nvidia  ll -h 
total 52K
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cublas
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cuda_cupti
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cuda_nvrtc
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cuda_runtime
drwxr-xr-x 5 root root 4.0K Oct 30 00:27 cudnn
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cufft
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 curand
drwxr-xr-x 5 root root 4.0K Oct 30 00:27 cusolver
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 cusparse
-rw-r--r-- 1 root root    0 Oct 30 00:27 __init__.py
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 nccl
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 nvjitlink
drwxr-xr-x 5 root root 4.0K Oct 30 00:26 nvtx
drwxr-xr-x 2 root root 4.0K Oct 30 00:27 __pycache_
```

让我们用ldd验证一下torch是否链接了nccl:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030004233.png)

还值得注意的一点是，尽管LD_LIBRARY_PATH里包含了`libcublas.so.12`，但是还是指向了conda环境中作为torch依赖安装的`libcublas.so.12`，这是因为`RPATH`的优先级大于`LD_LIBRARY_PATH`:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030011257.png)


## 2. pip install https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp39-cp39-linux_x86_64.whl

```sh
 ~  pip install https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp39-cp39-linux_x86_64.whl  
Collecting torch==2.1.2+cu121
  Using cached https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp39-cp39-linux_x86_64.whl (2200.7 MB)
Collecting filelock (from torch==2.1.2+cu121)
  Using cached filelock-3.16.1-py3-none-any.whl.metadata (2.9 kB)
Collecting typing-extensions (from torch==2.1.2+cu121)
  Using cached typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)
Collecting sympy (from torch==2.1.2+cu121)
  Using cached sympy-1.13.3-py3-none-any.whl.metadata (12 kB)
Collecting networkx (from torch==2.1.2+cu121)
  Using cached networkx-3.2.1-py3-none-any.whl.metadata (5.2 kB)
Collecting jinja2 (from torch==2.1.2+cu121)
  Using cached jinja2-3.1.4-py3-none-any.whl.metadata (2.6 kB)
Collecting fsspec (from torch==2.1.2+cu121)
  Using cached fsspec-2024.10.0-py3-none-any.whl.metadata (11 kB)
Collecting triton==2.1.0 (from torch==2.1.2+cu121)
  Using cached triton-2.1.0-0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.3 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch==2.1.2+cu121)
  Using cached MarkupSafe-3.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy->torch==2.1.2+cu121)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Using cached triton-2.1.0-0-cp39-cp39-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89.3 MB)
Using cached filelock-3.16.1-py3-none-any.whl (16 kB)
Using cached fsspec-2024.10.0-py3-none-any.whl (179 kB)
Using cached jinja2-3.1.4-py3-none-any.whl (133 kB)
Using cached networkx-3.2.1-py3-none-any.whl (1.6 MB)
Using cached sympy-1.13.3-py3-none-any.whl (6.2 MB)
Using cached typing_extensions-4.12.2-py3-none-any.whl (37 kB)
Using cached MarkupSafe-3.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (20 kB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Installing collected packages: mpmath, typing-extensions, sympy, networkx, MarkupSafe, fsspec, filelock, triton, jinja2, torch
Successfully installed MarkupSafe-3.0.2 filelock-3.16.1 fsspec-2024.10.0 jinja2-3.1.4 mpmath-1.3.0 networkx-3.2.1 sympy-1.13.3 torch-2.1.2+cu121 triton-2.1.0 typing-extensions-4.12.2
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
 ~                                                                                                                                            
```
注意到，没有安装一大坨的nvidia的依赖，nccl也没有安装。但是torch中是能使用nccl的：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030005504.png)

ldd也看不到`libnccl.so`了：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030005557.png)
这里有点割裂：
- 虽然torch也带了cublas，但cublas 使用本地cublas，因为设置了LD_LIBRARY_PATH，并且这个版本的so没有使用RPATH而是使用了RUNPATH，RUNPATH优先级小于`LD_LIBRARY_PATH`:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030011437.png)
- cudnn使用torch的cudnn，因为本地的cuda toolkit中没有cudnn
- cudart使用写死commit号的版本。。这就不知道为啥要写死了

甚至gdb进去也看不到
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/c7a3a02b6a59a84cc9b19bf39ef5fd2a.png)

查阅了一些资料：
1. https://discuss.pytorch.org/t/how-to-link-a-custom-nccl-version/107464/7
2. https://cloud.tencent.com/document/product/1646/93319#b405fcc2-aeac-43b7-9a89-9273194775fa
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030005759.png)
其中，2还给出了nccl静态链接的情况下如何使用自定义版本的nccl：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241030005850.png)

如果觉得编译pytorch比较麻烦，2还给出了nccl插件的办法。


---

为什么会有这种区别呢，我的猜测是：
- pip install torch==2.1.2时，假设本地没有任何cuda环境，所以torch倾向于把一整套都装上
- 而使用`pip install https://download.pytorch.org/whl/cu121/torch-xxx`时，代表本地有cu121环境，所以torch不会试图安装整个cuda环境，但nccl为什么要静态链接呢？还是不理解。