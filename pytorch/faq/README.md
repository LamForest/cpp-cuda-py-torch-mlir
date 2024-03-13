# Error of PyTorch


1. 
```
Traceback (most recent call last):
  File "leaky.py", line 8, in <module>
    import torch
  File "/home/github/pytorch/torch/__init__.py", line 237, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/github/pytorch/torch/lib/libtorch_python.so: undefined symbol: PyObject_GC_IsTracked
```
使用的python版本和torch编译时的版本不一致，跨版本有些符号会增减。

