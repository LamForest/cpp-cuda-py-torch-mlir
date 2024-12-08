
这篇文章用一个工作中遇到的例子来介绍2个工具的使用：

- py-spy 对程序的堆栈、性能进行实时的监控，具有dump top record 3个能力，具体参考 [用py-spy分析Python代码](https://zhuanlan.zhihu.com/p/358466229)
- madbg 使用ipdb attach到一个正在运行的python进程上，进行调试

那为什么不直接使用ipdb，vscode调试或者其他的python性能分析工具？相比于其他工具，这两个工具的特点是无需修改python代码，任意一个正在运行的python进程，都可以通过py-spy madbg进行调试；其次，它们很易用，一行命令即可使用，在服务器上不需要复杂的配置。

典型的场景有：比如生产环境突然卡死了，想要修改源码，加入pdb调试代码很麻烦；或者是一个偶现的问题，跑了几百次才能复现一次（比如[多进程同步](https://stackoverflow.com/q/25308847)）。

这里用一个我实际遇到的场景来举例。

## 举例：megatron 启动 hang/卡死

#### 实际问题
有A、B两台机器，A机器上跑一个单机8卡的Megatron-LM模型，一切正常；B机器上跑一个单机8卡的Megatron-LM模型，训练在启动阶段长时间卡死，显存占用只有40M左右，两边的硬件、软件版本完全一致。


#### 定位过程
之前遇到这种问题，我可能采取的思路是二分在代码里打断点，最终缩小到某一行python代码里。但对于Megatron-LM，这个方法可行，但有点麻烦。因为Megatron-LM的的启动比较复杂，包含通信库初始化、模型初始化、数据集cache加载等等。

所以这里我尝试使用py-spy:
```python
> pip install py-spy
> py-spy dump --pid 79655
Process 79655: /root/miniconda/envs/torch201_cuda/bin/python -u /workspace/Megatron-LM/examples/../pretrain_llama.py --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-layers 8 --tokenizer-type HFTokenizer --hf-tokenizer-path /workspace/hf_tokenizer ... --distributed-backend nccl
Python v3.8.18 (/root/miniconda/envs/torch201_cuda/bin/python3.8)

Thread 79655 (idle): "MainThread"
    wait (torch/utils/file_baton.py:42)
    _jit_compile (torch/utils/cpp_extension.py:1523)
    load (torch/utils/cpp_extension.py:1284)
    _c_extention_load_helper (megatron/accelerator/custom_ops/xpu_ops/__init__.py:42)
    load (megatron/accelerator/custom_ops/xpu_ops/__init__.py:54)
    register_ops (megatron/accelerator/custom_ops/xpu_custom_operator_manager.py:167)
    load_custom_ops (megatron/accelerator/xpu_accelerator.py:406)
    _compile_dependencies (megatron/initialize.py:210)
    initialize_megatron (megatron/initialize.py:90)
    pretrain (megatron/training.py:95)
    <module> (pretrain_llama.py:117)
```

py-spy直接定位到了程序hang的地方。有了这个信息后，我们按图索骥，直接看torch这行代码：[wait (torch/utils/file_baton.py:42)](https://github.com/pytorch/pytorch/blob/c8c669ce749dec7c8cc448b5e0de15d38023fa78/torch/utils/file_baton.py#L42):
```python
    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor.
        """
        while os.path.exists(self.lock_file_path): #堆栈指向这里
            time.sleep(self.wait_seconds)
```

再往上看一层堆栈，[_jit_compile (torch/utils/cpp_extension.py:1523)](https://github.com/pytorch/pytorch/blob/e9ebda29d87ce0916ab08c06ab26fd3766a870e5/torch/utils/cpp_extension.py#L1486C18-L1486C29)：：

torch在加载cpp_extension时，为了避免多个进程对同一份C++代码进行编译造成冲突，使用了基于文件的同步机制，多进程同时进入时，第一个进程通过FileBaton try_acquire()创建一个lock文件，并开始编译cpp_externsion；其他进程发现lock文件已被创建，则在FileBaton的wait中自旋等待；当第一个进程加载完成后，释放FileBaton（其实就是把lock文件删除）；最后所有进程加载编译好的so。

尽管torch使用了finally来确保任何情况下，FileBaton创建的lock文件都能够释放
```python 
            finally:
                baton.release()
```

但还是有不少情况，python进程会来不及走到finally，比如被killed。那么此时该lock文件会残留。当再启动程序时，由于lock文件存在，所有进程都try_acquire()失败，开始wait等待，但永远等不到wait结束，于是产生了死锁。

> Q：这种基于文件的同步机制是最优的吗？如何避免这个问题?

那下一个问题就是找到lock文件并将其删除。这其实挺简单的，在FileBaton wait里打印即可：`print(self.lock_file_path)`。

有没有更简单的方式呢？如果能用pdb attach到hang的进程中，因为正好hang在wait()中，那就能在pdb中直接`print(self.lock_file_path)`，不用修改代码。

使用pdb attach到指定进程中有许多方法，这里使用的是madbg:
```bash
pip install madbg
madbg attach <pid>
```
效果如下：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241207014859.png)

删除这个文件后，问题解决，模型不再hang。

