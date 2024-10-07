这个文章的主要说明：
1. nsys的基本使用
2. torch wait_stream实现

## nsys下载

最新版本在[nsight systems](https://developer.nvidia.com/nsight-systems/get-started#latest-version)上下载。

Linux:
```sh
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/NsightSystems-linux-public-2024.6.1.90-3490548.run
bash NsightSystems-linux-public-2024.6.1.90-3490548.run
echo "export PATH=/opt/nvidia/nsight-systems/2024.6.1/bin:$PATH" >> ~/.bashrc
```

Mac
```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/NsightSystems-macos-public-2024.6.1.90-3490548.dmg
```

在Linux上获取profiling结果后，放到mac上查看。

## nsys使用

```sh
nsys profile \
-w true \
-t cuda,nvtx,osrt,cudnn,cublas \
-s process-tree \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
--cudabacktrace=all \
-x true \
-o wait \
-f true \
python wait_stream.py
```
参数意义请参考文档：https://docs.nvidia.com/nsight-systems/UserGuide/index.html
有几个参数比较重要:
- `--capture-range=cudaProfilerApi --capture-range-end=stop` 这两个结合起来，可以通过`torch.cuda.profiler.start() torch.cuda.profiler.stop()` 控制nsys的范围，防止初始化、warmup的代码干扰分析。
- `-f true` overwrite已有的`.nsys-rep`文件

这里给出的参数参考了：
1. https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
2. https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59

## 举例：分析torch Stream.wait_stream()的实现

```py
import torch
import time
from torch.cuda import nvtx

def main():
    #init and warmup
    cuda = torch.device('cuda')
    s = torch.cuda.Stream()  # Create a new stream.
    default_stream = torch.cuda.default_stream(cuda)
    a = torch.empty(8192, 8192, device=cuda).normal_(0.0, 1.0)
    for _ in range(10):
        b = torch.matmul(a, a.T).sum()
        with torch.cuda.stream(s):
            c = torch.matmul(a, a + 1).sum()
        torch.cuda.synchronize()
        
    # start profiling
    torch.cuda.profiler.start()
    nvtx.range_push("run")
    c = torch.matmul(a, a.T)
    
    s.wait_stream(default_stream)
    
    with torch.cuda.stream(s):
        d = torch.matmul(c, c+1)

    torch.cuda.synchronize()
    a.record_stream(s)
    nvtx.range_pop()
    
    
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()

```

在mac上的nsight system中打开后，如下：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007180623.png)

从图中可以看出来，CPU进行这段代码中的cuda api调用、kernel launch都是异步的，我们继续zoom in：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007180851.png)

## 举例 resnet18

```sh
nsys profile \
-w true \
-t cuda,nvtx,osrt,cudnn,cublas, \
-s cpu \
--capture-range=cudaProfilerApi \
--capture-range-end=stop \
--cudabacktrace=all \
-x true \
-o resnet18 \
-f true \
python resnet18.py
```


```py
import torch
import torch.nn as nn
import torchvision.models as models

# setup
device = 'cuda:0'
model = models.resnet18().to(device)
data = torch.randn(64, 3, 224, 224, device=device)
target = torch.randint(0, 1000, (64,), device=device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nb_iters = 12
warmup_iters = 10

for i in range(nb_iters):
    optimizer.zero_grad()

    # start profiling after 10 warmup iterations
    if i == warmup_iters:
        torch.cuda.profiler.start()
        ctx = torch.autograd.profiler.emit_nvtx()
        ctx.__enter__()

    # push range for current iteration
    if i >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(i))

    # push range for forward
    if i >= warmup_iters: torch.cuda.nvtx.range_push("forward")
    output = model(data)
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    loss = criterion(output, target)

    if i >= warmup_iters: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    if i >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
    optimizer.step()
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

    # pop iteration range
    if i >= warmup_iters: torch.cuda.nvtx.range_pop()

ctx.__exit__(None,None, None)
torch.cuda.profiler.stop()
```

这里新增了`torch.autograd.profiler.emit_nvtx()` 每个aten算子都会被一个nvtx所包含，如下图所示：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007181959.png)


如何将fwd bwd算子对应起来？
可以通过seq，比如前向cross_entropy的logsoftmax算子，前向如下：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007182117.png)
那么在nsight中搜索790，就能找到对应的反向算子：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007182404.png)

详细说明见：https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.emit_nvtx