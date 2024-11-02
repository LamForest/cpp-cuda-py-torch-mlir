新的通信算子的dispatch方式

## 如何根据tensor device调用Backend的通信算子
使用方式后向兼容，之前的初始化方式仍然可用：
```py
init_process_group("nccl", ...)
init_process_group("mpi", ...)
```
这种单一Backend的写法，在2.0.1中仍然可以继续使用，但比如nccl不支持cpu tensor，所以得到的效果倒是和之前一样，唯一的区别是从dispatcher走了一趟。


如果要在同一个pg上根据tensor device，使用不同的backend进行通信，那需要为每个device指定backend
```py
init_process_group(...) #不传入backend，默认等价于下面的形式，即CPU使用gloo，cuda使用nccl
init_process_group("cpu:gloo,cuda:nccl", ...)
```

效果如下
```py
import torch
import torch.distributed as dist
import os
import argparse
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"{os.getpid()=}, {rank=}, {world_size=}, init process group")
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def op_all_reduce(rank, tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'[{rank}] all_reduce result: {tensor}')


def distributed_ops_demo(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    tensor_cuda = torch.tensor([1,2,3], device="cuda")
    op_all_reduce(rank, tensor_cuda) #nccl backend

    tensor_cpu = torch.tensor([10,20,30], device="cpu")
    op_all_reduce(rank, tensor_cpu) #gloo backend

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed operations")
    parser.add_argument("--world_size", type=int, default=2, help="Number of processes (world size)")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(distributed_ops_demo, args=(world_size,), nprocs=world_size, join=True)

```


## 具体实现

### 初始化：init_process_group

init_process_group只是一个wrapper，大部份的实现逻辑在_new_process_group_helper

1. pg = ProcessGroup(...)

2. 在_new_process_group_helper的for循环中，对于传入的每一组`device:backend`:

- 如果backend是built-in的backend，比如gloo nccl mpi，则代码中hardcode了backend -> ProcessGroupXXX的对应关系，否则在distributed_c10d.Backend中查找backend对应的构造函数（或者叫工厂函数）：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102202702.png)
这是通过distributed_c10d.Backend.register_backend注册的，所有非built-in的backend，都要进行这个注册操作。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102202900.png)



- 查询到工厂函数/构造函数后，进行实例化backend_class = ProcessGroupXXX(...)

Note: 其实ProcessGroupXXX已经是过时的名字了，现在ProcessGroupXXX不再继承ProcessGroup，而是继承Backend，并且受ProcessGroup管理；更好的名字是BackendGloo，BackendNCCL，只是为了后向兼容，所以还保留ProcessGroupXXX这个名字：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102202015.png)

- 检查backend_class是否继承了Backend；如果没有，说明该后端继承的还是ProcessGroup，于是pg = backend_class
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102202248.png)

- 向pg中注册 device -> backend_class的对应关系


### 调用通信算子
当使用同一个pg在cpu、cuda的tensor上进行通信的时候：
```py
import torch.distributed as dist

dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)

dist.all_reduce(tensor_cuda)

dist.all_reduce(tensor_cpu)
```

1. 由于all_reduce没有传入process_group，所以使用默认的pg = default_process_group
2. pg.allreduce(tensor_cuda)
3. 进入C++中定义的 ProcessGroup::allreduce，调用dispatcher注册的`c10d::allreduce`:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102204359.png)
4. 由于tensor device为cuda，所以调用的是`c10d::allreduce_cuda`；
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102204458.png)
5. 不过`allreduce_cuda`中并没有与cuda相关的代码，而是通过pg->getBackend查询负责处理cuda的是哪个backend_class，然后调用backend_class的allreduce
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241102204711.png)

至此，通过dispatcher调用Backend的通信算子完成。

## 后续
上述逻辑&代码来自于pytorch 2.0.1，从2.0.1到2.5.0有一些小变化，比如会使用一些macro来减少重复数据、c10d的通信算子的参数改变了，但是总体来说，上述的逻辑&代码依旧保持一致。

## 参考

1. 这个特性的需求文档：https://github.com/pytorch/pytorch/issues/86225

