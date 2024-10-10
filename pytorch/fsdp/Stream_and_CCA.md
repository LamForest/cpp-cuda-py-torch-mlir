

https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486


## CCA每个block与某个stream绑定
如果某个stream A的block用完了，而其他stream B还有富余，B上富余的block是不能转让的，因为不知道B是否还有未完成的操作使用了已释放的block。

这时候，CCA会wait所有stream完成，然后emptycache。这也算是一种soft OOM

One effect of the CUDACachingAllocator tagging blocks by stream is that the blocks will keep that tag during its entire lifetime. When a request for memory comes in from a certain stream, the CUDACachingAllocator will first give out blocks with the same stream tag if available. If not available…well…this is another reason the CUDACachingAllocator would need to cudaFree() and reallocate memory (which, if you recall, is slow). Even if you have enough memory on the machine, if it’s tied to a different stream, our CUDACachingAllocator will cudaFree() its reserved memory to reallocate for your particular stream. Again, this problem is only relevant when multiple streams are used. In the default PyTorch case, you should not run into this as everything uses a single stream.

## record_stream() is the only reason why a requested free is not immediately blockFreed. 
(Does this remind you of a profile we had mentioned earlier?) It is important to note that, with our CUDACachingAllocator, del doesn’t literally blockFree anything. **It is only during a later malloc where our CUDACachingAllocator has to evaluate whether a block of memory can be reused.** Consider the following allocations of C and E.

为啥要等到之后的malloc而不是在record的task完成后立刻释放呢？拜托，这得有中断给到CCA，所以只能到下一次进入CCA的时候，才有机会做这个检查。

## Tensor.recordStream(s)实现 (torch 2.1.2)
https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
Marks the tensor as having been used by this stream. When the tensor is deallocated, ensure the tensor memory is not reused for another tensor until all work queued on stream at the time of deallocation is complete.
根据文档，当t.recordStream(s)时，不会实际records，当t释放时，才会开始record s 上的task。这点为什么要这么设计？

1. t.record_stream(s)
这是一个aten算子，只有cuda的实现

```yaml
- func: record_stream(Tensor(a!) self, Stream s) -> ()
  variants: method
  dispatch:
    CUDA: record_stream_cuda

```

```cpp
//RecordStream.cu
namespace at::native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  c10::cuda::CUDACachingAllocator::recordStream(self.storage().data_ptr(), at::cuda::CUDAStream::unpack3(data.stream_id, data.device_index, data.device_type));
}
}
```

recordStream在block->stream_uses打了一个标记，稍后在释放时，会在insert_events中record所有stream_uses中的流
```cpp
  void recordStream(Block* block, cuda::CUDAStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }
```
> main最新的实现还变了，直接从EventPool创建了event？但是record_stream event record不是应该在释放的时候吗（按照文档）

从nsys也能看出来，没有调用cuda api，是一个纯torch行为。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009134103.png)

2. 当t释放时，比如引用计数0，或者手动resize storage 为 0；此时进入`c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::free(...)`，会检查这块内存是否record了其他stream

如果有，会调用`c10::cuda::CUDACachingAllocator::Native::DeviceCachingAllocator::insert_events(...)`，处理所有stream_uses中的流：创建event并record；然后append到cuda_events中。

```cpp
  void insert_events(Block* block) {
    int prev_device = 0;
    C10_CUDA_CHECK(c10::cuda::GetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_CUDA_CHECK(c10::cuda::SetDevice(stream.device_index()));

      EventPool::Event event =
          create_event_internal(static_cast<int>(stream.device_index()));
      C10_CUDA_CHECK(cudaEventRecord(*event, stream.stream()));

      block->event_count++;
      cuda_events[stream].emplace_back(std::move(event), block);
    }

    C10_CUDA_CHECK(c10::cuda::MaybeSetDevice(prev_device));
  }
```

从nsys看出，resize(0)会触发3次cudaEventRecord。为啥是3次？stream_uses中应该只有一个默认流吧？？？
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009134422.png)


3. 此次释放及以后每一次malloc，都会在`process_events`对cuda_events的所有流上的所有event进行cudaEventQuery，event是否已经完成。

一个block上的所有event都完成了，则free_block，此时这个block才是真正被释放了。

```cpp
  void process_events(const std::shared_ptr<GatheredContext>& context) {
    insert_events_deferred_until_no_capture();

    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = cuda_events.begin(); it != cuda_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        cudaError_t err = C10_CUDA_ERROR_HANDLED(cudaEventQuery(*event));
        if (err == cudaErrorNotReady) {
          // ignore and clear the error if not ready
          (void)cudaGetLastError();
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != cudaSuccess) {
          C10_CUDA_CHECK(err);
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = cuda_events.erase(it);
      } else {
        it++;
      }
    }
  }

  Block* malloc(int device, size_t orig_size, cudaStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway == 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their GPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cudaEventQueries, illegal during CUDA graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events(context);
    }
    ...
```

这可以从第二个fsdp module的alloc中看到，有多个cudaEventQuery，这就是在查第1个fsdp module reshard时的record_stream的event，是否完成了。
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241009135309.png)

### 而fsdp是如何使用record_stream的呢？
是在free storage之前才record_stream，记录计算上的所有任务，保证unsharded_flat_param(一般情况下就是AG param)在计算完成之后才会被释放。
```py
    def _free_unsharded_flat_param(self):
        """
        Frees the padded unsharded flat parameter. The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
        self._check_sharded_strategy()
        unsharded_flat_param = self._get_padded_unsharded_flat_param()
        self._check_storage_allocated(unsharded_flat_param)
        self._check_on_compute_device(unsharded_flat_param)
        # Do not free the memory until all ops in the current stream finish
        _no_dispatch_record_stream(
            unsharded_flat_param, self._device_handle.current_stream()
        )
        _free_storage(unsharded_flat_param)
```

根据这个discuss，https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486 会去掉recordStream的使用？