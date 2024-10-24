
```py
torch.manual_seed(0)
a = torch.randn(10, device="cpu").to(torch.float16)
b = 0.5154787418555804
c = a.mul(b)
```
c如何计算呢？有2种情况：
1. b 转换为 fp16 与 fp16的tensor a进行计算
2. b 转换为 fp32，a转换为fp32，计算，写回时再转换为fp16。精度更高

pytorch 2.0.1采取的是方法1，而2.1.2采取的是方法2。可使用如下程序验证：
```py
import torch
print(torch.__version__)
torch.manual_seed(0)
a = torch.randn(10, device="cpu").to(torch.float16)
b = 0.5154787418555804
c = a.mul(b)

print(f"{a[1].item()=:.10f}, {c[1].item()=:.10f}")

print("==== 方法1: cast scalar to fp16 ====")
c = a.mul(
    torch.tensor([b], dtype=torch.float16)
)

print(f"{a[1].item()=:.10f}, {c[1].item()=:.10f}")


print("==== 方法2: cast a to fp32 and then cast result to fp16====")
c = a.mul(
    torch.tensor([b], dtype=torch.float32)
).to(torch.float16)

print(f"{a[1].item()=:.10f}, {c[1].item()=:.10f}")
```

方法1的结果为：`c[1].item()=-0.1513671875`，这也正是python2.0.1的结果
方法2的结果为：`c[1].item()=-0.1512451172`，这也正是python2.1.2的结果
