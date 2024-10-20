

```
4 Addr: b7f1000000000_0, Size: 1.0GiB (1073741824 bytes) allocation, Total memory used after allocation: 2.0GiB (2147483648 bytes), stream 0, timestamp Invalid Date
CUDACachingAllocator.cpp:0:c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(int, unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void**, int, unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::allocate(unsigned long) const
:0:at::TensorBase at::detail::_empty_generic<long>(c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType, c10::optional<c10::MemoryFormat>)
??:0:at::detail::empty_generic(c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType, c10::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, c10::ScalarType, c10::optional<c10::Device>, c10::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)
??:0:at::detail::empty_cuda(c10::ArrayRef<long>, c10::TensorOptions const&)
RegisterCUDA.cpp:0:at::(anonymous namespace)::create_out(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions const&)
RegisterCUDA.cpp:0:at::(anonymous namespace)::structured_cat_out_cuda_functional::set_output_raw_strided(long, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::TensorOptions, c10::ArrayRef<at::Dimname>)
??:0:at::meta::structured_cat::meta(c10::IListRef<at::Tensor> const&, long)
RegisterCUDA.cpp:0:at::(anonymous namespace)::wrapper_CUDA_cat(c10::IListRef<at::Tensor> const&, long)
RegisterCUDA.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::IListRef<at::Tensor> const&, long), &at::(anonymous namespace)::wrapper_CUDA_cat>, at::Tensor, c10::guts::typelist::typelist<c10::IListRef<at::Tensor> const&, long> >, at::Tensor (c10::IListRef<at::Tensor> const&, long)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long)
??:0:at::_ops::cat::redispatch(c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long)
VariableType_0.cpp:0:torch::autograd::VariableType::(anonymous namespace)::cat(c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long)
VariableType_0.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long), &torch::autograd::VariableType::(anonymous namespace)::cat>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long> >, at::Tensor (c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::IListRef<at::Tensor> const&, long)
??:0:at::_ops::cat::call(c10::IListRef<at::Tensor> const&, long)
python_torch_functions_2.cpp:0:torch::autograd::THPVariable_cat(_object*, _object*, _object*)
??:0:PyCFunction_Call
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:792:flatten_tensors
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:800:flatten_tensors_into_flat_param
:0:method_vectorcall
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:687:_init_flat_param_and_metadata
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:537:__init__
:0:slot_tp_init
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/_init_utils.py:531:_init_param_handle_from_params
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/_init_utils.py:519:_init_param_handle_from_module
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:487:__init__
:0:slot_tp_init
```

## FSDP (继承nn.Module, FSDPState)
成员变量(FSDPState)
- `backward_prefetch: BackwardPrefetch`
- `forward_prefetch: ForwardPrefetch`
成员变量(其他)
- `self._fsdp_wrapper_module` 初始化FSDP时，传入的nn.Module
- 



## FlatParameter
```
    """
    This is the flat parameter used by :class:`FullyShardedDataParallel`. It is
    comprised of one or more original parameters, which are flattened and
    concatenated to construct the flat parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.
```

- .data: 
```
self.flat_param.data = padded_unsharded_flat_param[
            : unsharded_size.numel()
        ].view(
            unsharded_size
        ) 
```
- .grad: 由于flat_param是nn.Parameter，requires_grad=True，所以反向时会自动计算grad
- _post_backward_hook_state: 注册了flat_param注册了post_backward_hook后，该项为flat_param._post_backward_hook_state = (acc_grad, hook_handle)，我以为只用注册一次，但是源码里为了严谨，每次pre_forward的时候都会去注册：
```py
        # Register post-backward hooks to reshard the parameters and reduce-scatter
        # their gradients. They must be re-registered every forward pass in case
        # the `grad_fn` is mutated.
        _register_post_backward_hook(state, handle)
```

## `FSDP.__init__`

_init_ignored_module_states 略

_init_device_handle 通过以下优先级，确定当前rank的device：
1. FSDP的参数`device_id`
2. module如果在某个后端上（cpu meta）除外，则使用该后端，如果在cpu上，会先把fsdp module to到cuda上。
如果没有使用wrap_policy，那么整个模型实际上是一个fsdp_module, 会在已有的显存占用上增加整个模型显存+fsdp sharded显存，虽然会在FSDP之后马上降回到fsdp sharded的显存大小，但还是会容易爆显存，不推荐使用
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241003173801.png)
如果用了wrap_policy，那么每个fsdp module会逐个to cuda并进行初始化，并不会导致过高的内存峰值
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241016170932.png)
3. 否则使用`torch.cuda.current_device()`（最低优先级）
该函数的结果会存在`state._device_handle = _FSDPDeviceHandle.from_device(determined_device)` 中，这里为啥不用普通的torch.device呢？

注意到，这里的state其实就是FSDP self，因为FSDP继承了 nn.Module和FSDPState


_init_process_group_state. 暂时不讨论hybrid的话，就是default pg；一般的模型会在调用FSDP()之前初始化好default_pg，所以这里直接get_default_pg就行，没什么操作。

FSDP的参数也有process group，不用hybrid，也一般不传。

在_init_process_group_state中，还设置了gradient_pre(post)divide_factor：
> gradient_predivide_factor (float, default=1.0) – Allows perfoming the average of gradients over processes partially before and partially after the allreduce. Before allreduce: grads.mul_(1.0/gradient_predivide_factor). After allreduce: grads.mul_(gradient_predivide_factor/world size). This can reduce the stress on the dynamic range of FP16 allreduces for widely scaled-out runs.
> Source https://nvidia.github.io/apex/parallel.html

在DP中，各个rank的梯度要做个平均，即sum(grads)/world size，这个除以world size的操作被分成了predivide和postdivide，predivide在allreduce之前，postdivide在之后。比如pytorch中的设置为：
```
(Pdb) default_hooks.DefaultState._get_gradient_predivide_factor(64)
8.0

64个dp rank，pre=8.0，则post=world size/8.0=8.0

(Pdb) default_hooks.DefaultState._get_gradient_predivide_factor(4)
2.0
4个dp rank，pre=2.0，则post=world size/2.0=2.0

(Pdb) default_hooks.DefaultState._get_gradient_predivide_factor(2)
2.0
2个dp rank，pre=2.0，则post=world size/1.0=1.0
```

_init_core_state 初始化以下状态，只列出比较关键的
- sharding_strategy
- limit_all_gathers(RateLimiter) 
```
limit_all_gathers (bool) – If True, then FSDP explicitly synchronizes the CPU thread to ensure GPU memory usage from only two consecutive FSDP instances (the current instance running computation and the next instance whose all-gather is prefetched). If False, then FSDP allows the CPU thread to issue all-gathers without any extra synchronization. (Default: True) We often refer to this feature as the “rate limiter”. This flag should only be set to False for specific CPU-bound workloads with low memory pressure in which case the CPU thread can aggressively issue all kernels without concern for the GPU memory usage.
```
- use_orig_params:用户是否能够通过nn.Module.named_parameters()访问模型原始的params？设置为True感觉用户更友好点？为啥不是默认True，影响性能？
```
use_orig_params (bool) – Setting this to True has FSDP use module ‘s original parameters. FSDP exposes those original parameters to the user via nn.Module.named_parameters() instead of FSDP’s internal FlatParameter s. This means that the optimizer step runs on the original parameters, enabling per-original-parameter hyperparameters. FSDP preserves the original parameter variables and manipulates their data between unsharded and sharded forms, where they are always views into the underlying unsharded or sharded FlatParameter, respectively. With the current algorithm, the sharded form is always 1D, losing the original tensor structure. An original parameter may have all, some, or none of its data present for a given rank. In the none case, its data will be like a size-0 empty tensor. Users should not author programs relying on what data is present for a given original parameter in its sharded form. True is required to use torch.compile(). Setting this to False exposes FSDP’s internal FlatParameter s to the user via nn.Module.named_parameters(). (Default: False)
```
- ...


_init_runtime_state 略


_init_buffer_state：什么是buffer？https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266 。据我观察buffer没有纳入flatparameter的管理范围

_init_prefetching_state:
初始化bwd prefetching，有3种取值，怎么做的到？cpu会等待？
```
    - ``BACKWARD_PRE``: This enables the most overlap but increases memory
      usage the most. This prefetches the next set of parameters *before* the
      current set of parameters' gradient computation. This overlaps the *next
      all-gather* and the *current gradient computation*, and at the peak, it
      holds the current set of parameters, next set of parameters, and current
      set of gradients in memory. （FSDP默认值）不对，此时下一个fsdp unit的反向已经开始算了，所以还要包含下一fsdp的反向，可以看下内存时序
    - ``BACKWARD_POST``: This enables less overlap but requires less memory
      usage. This prefetches the next set of parameters *after* the current
      set of parameters' gradient computation. This overlaps the *current
      reduce-scatter* and the *next gradient computation*, and it frees the
      current set of parameters before allocating memory for the next set of
      parameters, only holding the next set of parameters and current set of
      gradients in memory at the peak.
    - FSDP's ``backward_prefetch`` argument accepts ``None``, which disables
      the backward prefetching altogether. This has no overlap and does not
      increase memory usage. In general, we do not recommend this setting since
      it may degrade throughput significantly.
```
以及fwd prefetching，只有true/false2种取值，默认为false。
```
forward_prefetch (bool) – If True, then FSDP explicitly prefetches the next forward-pass all-gather before the current forward computation. This is only useful for CPU-bound workloads, in which case issuing the next all-gather earlier may improve overlap. This should only be used for static-graph models since the prefetching follows the first iteration’s execution order. (Default: False)
```


### _init_param_handle_from_module


_move_module_to_device： 如果FSDP传入了device_id，那么会先把module移到device_id上；否则加入module在cpu上，则在cpu上做shard init，torch会warn比较慢。那如果module超过了显存大小呢？defer init怎么使用？似乎要安装一个另外的库 https://pytorch.org/torchdistx/latest/deferred_init.html
```
_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False
```
还有fsdp会引入超过module大小的显存占用吗？


没太理解，默认为False，先不管。sync_module_states: sync_module_states (bool) – If True, then each FSDP module will broadcast module parameters and buffers from rank 0 to ensure that they are replicated across ranks (adding communication overhead to this constructor). This can help load state_dict checkpoints via load_state_dict in a memory efficient way. See FullStateDictConfig for an example of this. (Default: False)


padding:什么情况需要padding？

padding方式
```py

def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE
    )
```


extensions: 目前用不上，记录一下。类型为FSDPExtensions，好像hook作用


##### padding:

```
        # Only align addresses for `use_orig_params=True` (for now)
        align_addresses = use_orig_params
```
在padding在2处处理了（都只在use_orig_params=True时生效）
- for 循环中，保证总的numel的字节数整除 16 字节，比如fp32，则numel个数要整除4。
但是由于padding的逻辑写在前面，所以最后一个tensor如果没有对齐，那这段逻辑并不会在最后得到一个 整除4 的numel：

```
submodule_name='fc1', param_name='weight', aligned_numel=4, numel_to_pad=4, total_numel=0
len(params_to_flatten)=1, total_numel=67125249, total_numel_without_padding=67125249, param_infos[-1]=ParamInfo(param_name='weight', module=Linear(in_features=8193, out_features=8193, bias=False), module_name='fc1'), shapes[-1]=torch.Size([8193, 8193]), numels[-1]=67125249, fqn='fc1.weight'
submodule_name='fc2', param_name='weight', aligned_numel=4, numel_to_pad=3, total_numel=67125249
len(params_to_flatten)=3, total_numel=134250501, total_numel_without_padding=134250498, param_infos[-1]=ParamInfo(param_name='weight', module=Linear(in_features=8193, out_features=8193, bias=False), module_name='fc2'), shapes[-1]=torch.Size([8193, 8193]), numels[-1]=67125249, fqn='fc2.weight' #最后得到的total_numel不能整除4.
len(params_to_flatten)=3, len(numels)=3, len(shapes)=2, len(is_padding_mask)=3
```

我的修改建议是移到后面。

- 第二种是保证总的total_numel整除self.world_size，这个好理解，megatron也是如此，这样每个dp能分到完全一样的ele数量。
Q：这个我认为不管有没有开use_orig_params=True，也都需要padding到world_size的倍数。错误。这个逻辑不是这里， 而是在shard时的chunk来处理，那这里处理的意义是什么？


这里特殊的有几点是
- padding tensor的值为42
```
def _construct_padding_tensor(
    padding_numel: int, dtype: torch.dtype, requires_grad: bool, device: torch.device
):
    # NOTE: Set the padding value as a magic number for debuggability. The
    # value itself should never be used in any user-facing computation.
    return (
        torch.ones(
            (padding_numel,), dtype=dtype, requires_grad=requires_grad, device=device
        )
        * _FLAT_PARAM_PADDING_VALUE #该常量为42
    )
```
是否可以改成empty().fill_(42) 会更快些？

- FlatParameter的metadata存了一些元数据，在不考虑padding的情况下，是相等的，但是考虑padding的情况下，有些就不相等了。其中param_infos, shapes fqns(full qualified name)，param_extensions的长度等于非padding tensor的数量，即param的数量，但是numels、is_padding_mask的长度等于所有tensor的数量，即param + padding tensor的数量。



#### flatten flatten_tensors_into_flat_param flatten_tensors
终于到这一步了，不用考虑aligned_numel > 0，因为_init_flat_param_and_metadata中已经考虑了。


```python
  def flatten_tensors(...):
    ...
    else:
        flat_tensors = [
            torch.flatten(_detach_if_needed(tensor)) for tensor in tensors
        ]
    return torch.cat(flat_tensors, dim=0)
```


#### flatten -> sharded _init_param_handle_from_params 

通过cat得到一个flatten的tensor后，下一步就是shard：
```
    handle = FlatParamHandle(
        params,
        fully_sharded_module,
        state.compute_device,
        SHARDING_STRATEGY_MAP[state.sharding_strategy],
        state.cpu_offload.offload_params,
        state.mixed_precision.param_dtype,
        state.mixed_precision.reduce_dtype,
        state.mixed_precision.keep_low_precision_grads,
        state.process_group,
        state._use_orig_params,
    ) #flatten
    handle.shard() #shard
```

这块还会申请新的显存，堆栈如下：
```
5 Addr: b7f0fe0000000_0, Size: 512.0MiB (536870912 bytes) allocation, Total memory used after allocation: 2.5GiB (2684354560 bytes), stream 0, timestamp Invalid Date
CUDACachingAllocator.cpp:0:c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::malloc(int, unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::malloc(void**, int, unsigned long, CUstream_st*)
:0:c10::cuda::CUDACachingAllocator::Native::NativeCachingAllocator::allocate(unsigned long) const
:0:at::TensorBase at::detail::_empty_strided_generic<c10::ArrayRef<long> >(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType)
??:0:at::detail::empty_strided_generic(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType)
??:0:at::detail::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ScalarType, c10::optional<c10::Device>)
??:0:at::detail::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
??:0:at::native::empty_strided_cuda(c10::ArrayRef<long>, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
RegisterCUDA.cpp:0:at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
RegisterCUDA.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CUDA__empty_strided>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> > >, at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
??:0:at::_ops::empty_strided::redispatch(c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
RegisterBackendSelect.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>), &at::(anonymous namespace)::empty_strided>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> > >, at::Tensor (c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
??:0:at::_ops::empty_strided::call(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)
??:0:at::native::clone(at::Tensor const&, c10::optional<c10::MemoryFormat>)
RegisterCompositeExplicitAutograd.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (at::Tensor const&, c10::optional<c10::MemoryFormat>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeExplicitAutograd__clone>, at::Tensor, c10::guts::typelist::typelist<at::Tensor const&, c10::optional<c10::MemoryFormat> > >, at::Tensor (at::Tensor const&, c10::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)
??:0:at::_ops::clone::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)
VariableType_1.cpp:0:torch::autograd::VariableType::(anonymous namespace)::clone(c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)
VariableType_1.cpp:0:c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>), &torch::autograd::VariableType::(anonymous namespace)::clone>, at::Tensor, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat> > >, at::Tensor (c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)
??:0:at::_ops::clone::call(at::Tensor const&, c10::optional<c10::MemoryFormat>)
python_variable_methods.cpp:0:torch::autograd::THPVariable_clone(_object*, _object*, _object*)
:0:method_vectorcall_VARARGS_KEYWORDS
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:1018:_get_shard
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/flat_param.py:858:shard
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/utils/_contextlib.py:115:decorate_context
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/_init_utils.py:543:_init_param_handle_from_params
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/_init_utils.py:519:_init_param_handle_from_module
/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py:487:__init__
```


handle.shard()代码如下，具体的逻辑为：
1. flat tensor chunk 为 world size份，此时有2种情况：
 - flat_tensor.numel() < worldsize，此时有的rank分不到chunk结果，那么为这个rank构造一个全0的tensor
 - flat_tensor.numel() 无法整除 worldsize，那么最后一个rank用0补齐
2. chunk是一个view算子，通过shard = chunk.clone() 产生新的shard tensor
3. `orig_storage._resize_(0)` 释放 flat tensor
```py
    @torch.no_grad()
    def shard(self):
        """
        Shards the handle's ``FlatParameter``. This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
        flat_param = self.flat_param
        if not self.uses_sharded_strategy:
            self._init_shard_metadata(0, 0, flat_param.numel() - 1)
        else:
            _p_assert(
                flat_param.storage_offset() == 0,
                "The `FlatParameter` is not the sole occupant of its storage",
            )
            orig_storage = flat_param._typed_storage()
            sharded_flat_param, numel_padded = FlatParamHandle._get_shard(
                flat_param, self.rank, self.world_size
            )
            flat_param.set_(sharded_flat_param)  # type: ignore[call-overload]
            start_idx = sharded_flat_param.numel() * self.rank
            end_idx = sharded_flat_param.numel() * (self.rank + 1) - 1  # inclusive
            self._init_shard_metadata(numel_padded, start_idx, end_idx)
            if orig_storage._size() > 0: #释放
                orig_storage._resize_(0)
        if self._use_orig_params:
            self._use_sharded_views()
```

## FSDP 前后的parameters变化

FSDP前model的named_parameters如下：
```
==== start traversing model: <class '__mp_main__.Net'>
Parameter name: fc1.weight
Parameter shape: torch.Size([512, 512])
Parameter requires_grad: True
Parameter data: tensor([[-0.0250, -0.0079,  0.0352,  ...,  0.0413,  0.0277, -0.0034],
        [ 0.0161, -0.0359, -0.0036,  ...,  0.0315, -0.0187,  0.0373],
        [ 0.0396,  0.0287,  0.0362,  ..., -0.0221, -0.0027, -0.0115],
        ...,
        [-0.0214,  0.0118,  0.0288,  ...,  0.0174, -0.0162,  0.0322],
        [ 0.0351, -0.0083,  0.0174,  ...,  0.0202, -0.0343,  0.0416],
        [ 0.0236,  0.0312,  0.0221,  ..., -0.0305,  0.0203, -0.0321]])

Parameter name: fc2.weight
Parameter shape: torch.Size([512, 512])
Parameter requires_grad: True
Parameter data: tensor([[ 1.7403e-02, -3.0526e-03,  5.7124e-03,  ...,  4.0722e-02,
          2.8896e-02,  2.3451e-02],
        [ 4.2621e-05,  2.8816e-03,  2.2567e-02,  ...,  3.1377e-02,
         -4.3220e-02,  2.9821e-02],
        [ 2.2272e-02,  5.1505e-04, -3.2796e-02,  ..., -3.6828e-02,
         -2.5609e-02,  1.4576e-02],
        ...,
        [ 3.3194e-02,  3.2103e-02,  2.5785e-02,  ..., -6.7504e-03,
         -3.6203e-02,  2.3436e-02],
        [-3.9682e-02, -1.6967e-04, -3.4863e-02,  ..., -1.9913e-02,
         -3.5508e-02,  2.0140e-02],
        [ 4.0525e-02,  1.1450e-02,  2.8087e-02,  ..., -1.3271e-02,
         -2.9887e-02,  8.6481e-03]])
```

FSDP后的named_parameters如下（use_orig_params=False：
```
==== start traversing model: <class '__mp_main__.Net'>
Parameter name: _flat_param #FLAT_PARAM字符串常量
Parameter shape: torch.Size([262144])
Parameter requires_grad: True
Parameter data: tensor([ 0.0032,  0.0300, -0.0418,  ...,  0.0342, -0.0338,  0.0169],
       device='cuda:0')
```

如果use_orig_params=True
则和原始的param一样，而没有_flat_param
```
==== start traversing model: <class '__mp_main__.Net'>
Parameter name: fc1.weight
Parameter shape: torch.Size([262144])
Parameter requires_grad: True
Parameter data: tensor([-0.0058,  0.0339,  0.0137,  ...,  0.0230,  0.0136, -0.0122],
       device='cuda:0')

Parameter name: fc2.weight
Parameter shape: torch.Size([0])
Parameter requires_grad: True
Parameter data: tensor([], device='cuda:0')
```

可以看到记录在_parameters中的param变为了一个shard flatparam，


## BUG? `_use_unsharded_views` 后 print(model.fc1.weight报错)

```
model = Net(512)
fsdp_model = FSDP(model, use_orig_params=False, device_id=torch.cuda.current_device())
print(f"{model.fc1.weight=}")#报错如下
```

```
Traceback (most recent call last):
  File "fsdp_simplenn.py", line 230, in <module>
    mp.spawn(fsdp_main,
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 246, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 202, in start_processes
    while not context.join():
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 163, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 1 terminated with the following error:
Traceback (most recent call last):
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 74, in _wrap
    fn(i, *args)
  File "/ssd1/gaotianlin/baidu/hac-aiacc/Megatron/old_scripts/fsdp/fsdp_simplenn.py", line 153, in fsdp_main
    print(f"{model.fc1.weight=}")
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor.py", line 431, in __repr__
    return torch._tensor_str._str(self, tensor_contents=tensor_contents)
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor_str.py", line 664, in _str
    return _str_intern(self, tensor_contents=tensor_contents)
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor_str.py", line 595, in _str_intern
    tensor_str = _tensor_str(self, indent)
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor_str.py", line 347, in _tensor_str
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor_str.py", line 381, in get_summarized_data
    start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/_tensor_str.py", line 381, in <listcomp>
    start = [self[i] for i in range(0, PRINT_OPTS.edgeitems)]
RuntimeError: setStorage: sizes [512], strides [1], storage offset 0, and itemsize 4 requiring a storage size of 2048 are out of bounds for storage of size 0

```


原因为在_use_unsharded_views设置了model.fc1.weight为flatparam的view tensor，但在shard()中flatparam在shard后，storage被resize_(0)了。此时model.fc1.weight底层是一个0 size storage，所以报错。

我尝试了注释掉FlatParamHandle `__init__` 最后的 `self._use_unsharded_views(as_params=False)#TODO 改为仅删除原始的param，但是不设置`，但是这又使得model.fc1.weight这个原始param没有被删除，导致报错
```
-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 74, in _wrap
    fn(i, *args)
  File "/ssd1/gaotianlin/baidu/hac-aiacc/Megatron/old_scripts/fsdp/fsdp_simplenn.py", line 139, in fsdp_main
    fsdp_model = FSDP(model, use_orig_params=False, device_id=torch.cuda.current_device())
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 497, in __init__
    _check_orig_params_flattened(self, self._ignored_params)
  File "/root/miniconda3/envs/old_mega/lib/python3.8/site-packages/torch/distributed/fsdp/_init_utils.py", line 1056, in _check_orig_params_flattened
    raise RuntimeError(
RuntimeError: Found an unflattened parameter: _fsdp_wrapped_module.fc1.weight; torch.Size([512, 512]) <class 'torch.nn.parameter.Parameter'>
```
这个报错是fsdp在检查有没有还未被fsdp处理的parameter。

所以，修复尝试可能是FlatParamHandle `__init__` 最后不调用`_use_unsharded_views`而是仅把所有parameter都的delattr，但是不setattr


然而，如果FSDP初始化时传入了 use_orig_params=True，则不会报错：
```
model.fc1.weight=Parameter containing:
tensor([-0.0058,  0.0339,  0.0137,  ...,  0.0230,  0.0136, -0.0122],
       device='cuda:0', requires_grad=True) 
model.fc2.weight=Parameter containing:
tensor([], device='cuda:0', requires_grad=True) #因为shard了，所以size==0
```
所以，这算是use_orig_params=False这个情况下有bug吗？


## typo
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241005140650.png)


## init_flat_param_attributes 为什么申请tensor后马上又释放storage？

```py
            flat_param._full_param_padded = torch.empty(
                padded_unsharded_numel,
                device=self.device,
                dtype=unsharded_param_dtype,
            )
            flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
            _free_storage(flat_param._full_param_padded)
```

猜测？这种做法可能看起来有些奇怪，因为我们刚刚分配了空间。这里的关键是理解 _free_storage 函数的作用：它并不是销毁张量，而是释放张量的内存存储，但保留张量的形状和数据类型等信息。这样做的目的是优化内存使用，避免在不需要使用该张量存储数据时占用内存资源。在需要时（例如在全局聚集过程中），相关代码可以重新分配内存并填充数据。


在unshard()时，取回了 _full_param_padded并检查是否storage==0，然后resize到`flat_param._padded_unsharded_size`，这不是又来一遍吗？


哈哈哈，现在我知道了是因为record_streams的作用，

## stream
第一次forward之前会进行lazy init，这时候会初始化流。假设没有用hybrid_sharding，会用到几个流：
1. 默认流（计算流）
2. unshard_stream: allocate all-gather的tensor，以及进行allgather（反向也算吗？，那反向就有2个通信流，分别用于RS和AG，为啥要搞2个？）
3. post_backward_stream: 反向reduce scatter的流
4. _pre_unshard_stream 不开混精应该用不上
```py
@no_type_check
def _init_streams(
    state: _FSDPState,
) -> None:
    """
    Initializes CUDA streams for overlapping communication, computation, and
    data transfers. The streams should be shared across FSDP instances.
    """
    assert state._is_root
    assert state._device_handle.is_available()
    uses_hybrid_sharding = any(
        fsdp_state.sharding_strategy in HYBRID_SHARDING_STRATEGIES
        for fsdp_state in state._all_fsdp_states
    )
    # Prioritize all-gathers/reduce-scatters over async all-reduce for HSDP and
    # preserve the default priority of 0 otherwise
    high_priority = -1 if state.limit_all_gathers and uses_hybrid_sharding else 0
    # Default stream for computation
    state._default_stream = state._device_handle.current_stream()
    # Stream for unshard logic, including allocating the all-gather destination
    # tensors and the all-gathers themselves
    state._unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for overlapping gradient reduction with the backward pass gradient
    # computation
    state._post_backward_stream = state._device_handle.Stream(priority=high_priority)
    # Stream for pre-unshard logic, namely allocations and writes for CPU
    # offloading (H2D copy) and mixed precision (low precision cast)
    state._pre_unshard_stream = state._device_handle.Stream(priority=high_priority)
    # Stream to run HSDP's all-reduce as async (if using HSDP)
    state._all_reduce_stream = (
        state._device_handle.Stream() if uses_hybrid_sharding else state._default_stream
    )

```


## runtime_utils中的unshard()由pre_unshard unshard post_unshared组成，分别干嘛的




## 为什么要在前向注册hook？
因为反向的时候，干涉不了，必须得在前向注册hook，才能干预反向的

为什么在_pre_forward进行注册，因为只有在前向的输入tensor上注册Accu hook，才能在反向结束后，立刻调用reduce scatter

同理，只有在post_forward注册tensor pre?hook，才能在反向开始之前，调用all_gather

## FlatParameterHandle

```py
    def unshard(self):
        """
        Runs the unshard logic. This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
        if not self.needs_unshard():
            # Even when not needing an unshard, we should switch to using
            # the unsharded flat parameter
            unsharded_flat_param = (
                self._get_padded_unsharded_flat_param()
                if self.uses_sharded_strategy
                else self.flat_param
            )
            self._use_unsharded_flat_param(unsharded_flat_param)
            return
        unsharded_flat_param = self._alloc_padded_unsharded_flat_param() #将flat_param._full_param_padded resize为all-gather target size。如果混精了，切
        padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param) #调用dist.all_gather_into_tensor进行unsharded
        self._use_unsharded_flat_param(padded_unsharded_flat_param)
```


## summon（召集）

临时将所有的param and grad（如果设置了with_grads=True）all-gather到所有rank上，如果设置了rank0_only=True，则只在rank_0上gather。

一个例子是收集所有grad计算 grad_norm: https://github.com/Lightning-AI/pytorch-lightning/issues/17600


## HandleTrainingState TrainingState

```py

class TrainingState(Enum):
    """
    An enum that indicates the state of a ``FullyShardedDataParallel` instance.
    """

    IDLE = auto() #刚初始化
    FORWARD_BACKWARD = auto() #一般情况？
    SUMMON_FULL_PARAMS = auto()


class HandleTrainingState(Enum):
    """
    An enum that indicates the state of a ``FlatParamHandle`.
    """

    IDLE = auto()
    FORWARD = auto()
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
    SUMMON_FULL_PARAMS = auto()

```


## flat_param.data的动态变化

```
    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.
```

1. root_pre_forward lazy init时，在 init_flat_param_attributes 中 flat_param._local_shard = flat_param.data
从此以后，flat_param._local_shard 一直持有shard param，一直不释放。

2. pre_forward时，进行unshard, flat_param.data = padded_unsharded_flat_param[:xxx]

3. 

## MixPrecision dtype https://pytorch.org/docs/2.1/fsdp.html#torch.distributed.fsdp.MixedPrecision

FSDP的参数mixedprecision 

一般情况下，没有设置，那么
FlatParamHandle._fwd_bwd_param_dtype = FlatParamHandle._orig_param_dtype
FlatParamHandle._reduce_dtype = FlatParamHandle._orig_param_dtype
而_orig_param_dtype=params[0]，即假设module的参数类型都一样的情况下，以第一个的dtype为准（需不需要检查所有的dtype是否一致？）

keep_low_precision_grads默认为False，那么何时将bf16 grad upcast到fp32？

#### _uses_param_mixed_precision 的实现

这还挺坑的，看上去是个属性，实际上不是。

一般不会模型初始化时参数为bf16，计算时用fp32吧。。。所以一般为false

```
    @property
    def _uses_param_mixed_precision(self) -> bool:
        return self._fwd_bwd_param_dtype != self._orig_param_dtype
```

## 优化器
优化器在fsdp后面attach上面，那么自然而然只会管理这个rank shard的一部分，我的猜测


## auto wrap

```
FullyShardedDataParallel(
  (_fsdp_wrapped_module): Net(
    (fc1): FullyShardedDataParallel(
      (_fsdp_wrapped_module): Linear(in_features=512, out_features=512, bias=False)
    )
    (fc2): FullyShardedDataParallel(
      (_fsdp_wrapped_module): Linear(in_features=512, out_features=512, bias=False)
    )
    (fc3): FullyShardedDataParallel(
      (_fsdp_wrapped_module): Linear(in_features=512, out_features=512, bias=False)
    )
  )
)
```

## too many preforward
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007233740.png)

当exec_order_data第一次执行时，会进行一些初始化逻辑；
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007233821.png)
这里的判断条件是`self.is_first_iter`即`self._iter == 0`，每次post_backward时，`self._iter += 1`，所以理论上，初始化只会执行一次，但是由于机智的我，没有调用backward，所以不会走到post_backward。。。。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007233950.png)

所以每次前向都会初始化，然后在timeline中就是占据了大量时间

## with torch.profiler.record_function("FullyShardedDataParallel._post_forward"): 包含了 nvtx.range

实验发现的

## root_pre_forward只会在root module调用，子module直接返回
可以从NS中看到：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241007235102.png)

## SHARD_GRAD_OP
该情况下，前向不会reshard flat param，因为前向判断是否需要reshard的条件是：
```py
@no_type_check
@nvtx.range("_runtime_utils._post_forward_reshard")
def _post_forward_reshard(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> None:
    """Reshards parameters in the post-forward."""
    if not handle:
        return
    # Do not free the root's parameters in the post-forward for `FULL_SHARD`
    # with the intention that they are immediately used for backward
    # computation (though this may not be true)
    free_unsharded_flat_param = (
        not state._is_root
        and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES #包含了FULL_SHARD, HYBRID_SHARD
    )
    print(f"{state._is_root=}, {handle._sharding_strategy=}")
    _reshard(state, handle, free_unsharded_flat_param)
```

但是反向会，因为反向的条件是
```py
@no_type_check
def _should_free_in_backward(
    state: _FSDPState,
    handle: FlatParamHandle,
) -> bool:
    """
    Returns whether FSDP should free the unsharded flat parameter in the
    post-backward or not.
    """
    if not handle.uses_sharded_strategy:
        return False
    # If not syncing gradients, then we do not free for strategies that do not
    # reshard after forward as a *heuristic* to tradeoff higher memory for
    # higher throughput.
    print_r(f"_should_free_in_backward {handle._sharding_strategy=}, {state._sync_gradients=}")
    return (
        state._sync_gradients #默认为True，除非no_sync
        or handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES #包含了FULL_SHARD, HYBRID_SHARD
    )
```

所以如果开了gradient accum，那么下一次前向的时候，需要再次unshard flat param。除非sync_gradient=False，这种情况下的行为还没仔细看，感觉也很坑。



## rate limiter如何实现

```

class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[torch.cuda.Event] = collections.deque()
        self._max_num_inflight_all_gathers = 2  # empirically chosen

    def enqueue(self, free_event: torch.cuda.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)

    def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()
        return None

    def _dequeue(self) -> Optional[torch.cuda.Event]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()
            return event
        return None

  

    if state.limit_all_gathers and free_unsharded_flat_param:
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            # We don't run a even queue for freeing under torch compile atm
            # But maybe we need to? TODO(voz): Look into this
            free_event = state._device_handle.Event()
            free_event.record()
            state._free_event_queue.enqueue(free_event)
```

当遇到root fsdp module是，因为root_fsdp module没有参数，所以直接返回，root fsdp module.handle is None。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241008002816.png)
所以在_pre_forward中，root_fsdp module直接返回，不会从queue中dequeue。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241008002855.png)
同理，也不会reshard，也是直接返回了：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241008003112.png)
当然这是后话了，root module的reshard在其他module之后。还早着呢

第一个linear层：
队列长度为0时，返回None，不做等待，直接unshard
当reshard发生之后，record，然后enqueue。
当队列长度为2时，返回event，直到event完成后，才能继续unshard。

这保证了第0个fsdp module reshard之后，第2个fsdp module才能unshard。

nsys中看到第2个fsdp module unshard会调用cudaEventSynchonize

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009140051.png)


## _register_post_backward_reshard_only_hook _register_post_backward_hook的区别

`_pre_forward`中会去尝试注册2个hook，从代码来看，一般2个hook不会同时注册上，两者是互补的。
首先 2.1的fsdp假设一个fsdp unit中，同时都是可学习的参数，或者同时都是不可学习的参数。

所以handle.flat_param.require_grads有2个状态，True False。

如果为True，则`_register_post_backward_hook`能够注册成功。这个hook会做grad RS和flat param reshard。

如果为False，则`_register_post_backward_reshard_only_hook`能够注册成功。这个hook会做flat param reshard。

## event.synchronize()阻塞CPU
因为unshard分配mem由于CCA的存在，是个纯CPU的事情，必须在CPU侧block住

## dist async_op=False在CUDA上是不生效的，返回不意味着完成

https://pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations

Synchronous operation - the default mode, when async_op is set to False. When the function returns, it is guaranteed that the collective operation is performed. **In the case of CUDA operations, it is not guaranteed that the CUDA operation is completed, since CUDA operations are asynchronous.** For CPU collectives, any further function calls utilizing the output of the collective call will behave as expected. For CUDA collectives, function calls utilizing the output on the same CUDA stream will behave as expected. Users must take care of synchronization under the scenario of running under different streams. For details on CUDA semantics such as stream synchronization, see CUDA Semantics. See the below script to see examples of differences in these semantics for CPU and CUDA operations.

## 每个fsdp module的第一个iter的 record_pre_forward在干嘛？占了好多时间，还有流操作
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009135046.png)


## allgather包含了这么多绿色的stream操作？在干嘛
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009135643.png)

## 可视化event.synchronize()带来的CPU wait和stream bubble
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009214705.png)
为啥fc0 fc1 fc2的event.synchronize()不需要等呢？因为我在step之间加了D2H，产生sync？所以fc0 fc1 fc2的event所记录的操作都执行完了，不需要等待。而fc3需要等待fc0执行完。。。因为我把cuda任务加重了，此时fc0还没做完，所以需要等待。

为啥fc1 AG这么长呢？不知道。。。

## record_params干啥用的？
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241010162927.png)

## 前向对于fc5做了 unshard fwd reshard，反向的话fc5怎么做？

也是unshard fwd reshard，此时无所谓fc5 fwd是否完成或者还没开始，分配一个新的storage进行AG，然后用这个全新的参数做AG。

而此时前向的fc5，用前向的AG的参数做前向，反正recordstream了，不会被释放掉。

## 内存分析

### 1. 各个fsdp 初始化

1. to cuda
2. cat to flat_param
3. shard: view and clone
4. free flat_param
5. original param free when function exit

### 2. root_pre_forward初始化_full_param_padded _padded_unsharded_size

在root_pre_forward的lazy_init中，会对每个fsdp module的flat_param进行init_flat_param_attributes操作。

init_flat_param_attributes中有下图的操作，使用torch.empty初始化_full_param_padded，并设置`flat_param._padded_unsharded_size = flat_param._full_param_padded.size()`，然后马上free flat_param。注意到这里没有什么event, record_stream；所以真的是初始化后直接释放了。


![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241016174719.png)

为什么要这么做呢？因为后续flat_param._full_param_padded不再创建tensor，而是直接操作flat_param._full_param_padded.storage()，所以需要在lazy_init中把tensor这个壳子创建出来，这样后续可以直接操作storage了！！！


因为是循环逐个调用每个fsdp_module的lazy_init，所以memory是逐个分配释放，如下图：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241016174204.png)

### 3. 每一层unshard

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241016184108.png)


## _reduce_grad的细节
1. 会用一个临时变量unshard_grad保存grad,然后grad置为None，这考虑到可能下一个grad计算时，当前RS还没完成（属于极端情况，但pytorch也考虑到了
2. 构造RS的output tensor, 即 torch.empty_like(unshard_grad.chunk(state.world_size))
2. pre_divide
3. RS(sharded_grad, unsharded_grad)
4. post_divide
5. 梯度累加（_accumulate_sharded_grad(...)）flat_param._saved_grad_shard += sharded_grad

unshard_grad shard_grad不会在退出时释放，因为nccl里还会持有tensor的引用。

### reduce_scatter的input output 的生命周期

首先output(sharded_grad)的生命周期不用担心，因为这是在post_backward stream上分配 + 使用的，只有分配和使用不同stream的情况下，才会有冲突

再看input的生命周期，有2个tensor：
1. flat_param.grad 这个tensor在默认流中计算梯度时分配；在post_backward stream上使用；出现了**跨流分配+使用**，所以需要record_stream
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241018205419.png)
2. 如果是需要pad的情况，那么padded_unsharded_grad 是在 post_backward stream上分配+使用的；此时flat_param.grad在默认流上分配，但是在post_backward stream上构造padded_unsharded_grad时使用，也在上图的record_stream中考虑了这种情况


太多细节了，麻了。

## optimizer

### 1. 初始化

由于是在fsdp初始化完成之后，挂上优化器，所以优化器只会看到sharded_param，优化器状态自然而然时sharded的。

根据
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241018214102.png)
adam的优化器状态都是0初始化，所以不需要考虑分布式初始化的问题。

### 2. step
因为

