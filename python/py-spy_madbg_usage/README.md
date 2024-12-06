
这次介绍2个工具：

- py-spy 对程序的堆栈、性能进行实时的监控，具有dump top record 3个能力，具体参考 [用py-spy分析Python代码](https://zhuanlan.zhihu.com/p/358466229)
- madbg 使用ipdb attach到一个正在运行的python进程上，进行调试

可能有人会产生疑问，为什么不直接使用ipdb，vscode调试或者其他的python性能分析工具？相比于其他工具，这两个工具的特点是无需修改python代码，任意一个正在运行的python进程，都可以通过py-spy madbg进行调试；其次，它们很易用，一行命令即可使用，在服务器上不需要复杂的配置。

有的场景是无法预测的，比如生产环境突然卡死了，想要修改源码，加入pdb调试代码很麻烦；或者是一个偶现的问题，跑了几百次才能复现一次（比如[多进程同步](https://stackoverflow.com/q/25308847)），这时候，py-spy madbg的优势就体现出来了。

这里用一个我实际遇到的场景来举例。

## 举例：megatron 启动 hang/卡死

背景：有A、B两台机器，A机器上跑一个单机8卡的Megatron-LM模型，一切正常；B机器上跑一个单机8卡的Megatron-LM模型，训练在启动阶段长时间卡死，显存占用只有10M左右，两边的硬件、软件版本完全一致。

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

py-spy直接定位到了程序hang的地方。有了这个信息后，我们按图索骥，直接看torch这行[代码](https://github.com/pytorch/pytorch/blob/c8c669ce749dec7c8cc448b5e0de15d38023fa78/torch/utils/file_baton.py#L42):
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

torch使用了基于文件的同步机制，并自spin等待。