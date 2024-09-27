## gradient-accumulation-fusion 希望解决的问题

考虑LLM里面的ffn的参数param，每次反向求param梯度时会进行如下操作：
1. 此时已经得到了grad_output
2. param.grad = grad_output.t().matmul(x)
3. param.main_grad += param.grad; param.grad = None

2、3的matmul 和 梯度累加能否合并呢？其实是可以的，也不复杂，就是在matmul算完后，不写入一个新的tensor，而是直接加到param.main_grad上。

那megatron是如何实现的？


## Megatron 如何实现gradient-accumulation-fusion


```py
# https://github.com/NVIDIA/Megatron-LM/blob/d55d61ad5ab67d64283f4543601abd47b190602c/megatron/core/tensor_parallel/layers.py#L495
        if ctx.gradient_accumulation_fusion:
            if wgrad_compute: #一般为True
                if weight.main_grad.dtype == torch.float32:
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        total_input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        total_input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

```

这里比较困惑的是fused_weight_gradient_mlp_cuda 从何而来？首先第一眼看上去这就是一个c扩展，但是Megatron源码里搜不到。

其实这个实现在apex中:
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240928011430.png)

wgrad_gemm_accum_fp32在这
https://github.com/NVIDIA/apex/blob/b7a4acc1c8599f9306b519c9a88c044f1b280a07/csrc/megatron/fused_weight_gradient_dense_cuda.cu#L132

源码比较好理解，这里简要说明一下。我们需要的是这么一个操作：`param.main_grad += grad_output.t().matmul(x)`，而正好cublas存在这么一个接口 [cublasGemmEx()](https://docs.nvidia.com/cuda/cublas/#cublasgemmex)：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240928012055.png)

式中的alpha beta被设置成了 1.0 1.0：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240928012003.png)