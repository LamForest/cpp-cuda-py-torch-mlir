`CUDA_VISIBLE_DEVICES=0,1 python use_orig_params.py`的在rank0上的运行结果如下：

```
====== use_orig_params=False ======
==== start traversing model: <class 'torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel'>
Parameter _fsdp_wrapped_module._flat_param: shape=torch.Size([163840])


====== use_orig_params=False ======
==== start traversing model: <class 'torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel'>
Parameter _fsdp_wrapped_module.fc0.weight: shape=torch.Size([65536])
Parameter _fsdp_wrapped_module.fc1.weight: shape=torch.Size([65536])
Parameter _fsdp_wrapped_module.fc2.weight: shape=torch.Size([32768])
Parameter _fsdp_wrapped_module.fc3.weight: shape=torch.Size([0])
Parameter _fsdp_wrapped_module.fc4.weight: shape=torch.Size([0])
```

在False情况下，fsdp初始化后named_parameters只能看到每个fsdp module的sharded flat_param。这时候使用optimizer绑定model时，optimizer看到的是module的sharded flat_param。那么优化器自然也shard了

而为True的情况下，module的named_parameters中还是有fc0 ~ fc4这些parameter，但变成了**sharded flat_param的view**；有些view是完整的，即fc0 ~ fc1，说明fc0 ~ fc1的参数完全在rank0的shard flat_param上；同理，fc2只有一半在rank0上，而fc3~fc4全在rank1的shard flat_param上。再考虑下optimizer绑定model时，optimizer看到的是fc0 ~ fc4 parameter，但是有的完整，有的不完整，自然而然优化器也shard了。
