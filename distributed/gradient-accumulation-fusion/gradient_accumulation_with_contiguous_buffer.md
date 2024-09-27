# 相关的几个参数

local(default): 为使用megatron的`class DistributedDataParallel(DistributedDataParallelBase):`类
torch: 使用torch的DDP类
```

    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')

```

使用megatron的DDP类时，所有的grad会放在一个连续的buffer里面
```
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
```
具体的实现方式是：
1. 将DDP中的所有的参数的按照dtype分类，bf16, fp32, fp16各分为1类。
2. 统计每类参数需要占据的显存大小
3. 每类参数申请一块连续的显存空间，并按照每个参数需要的显存大小，分配在这块连续空间中的起始地址，并赋值给param.main_grad(和torch使用的param.grad以示区别)
> Note: 如果有DP，那么由于DP使用的reduce_scatter需要各个dp的tensor长度一样，这意味着这块连续的显存空间需要整除 `#DP * sizeof(dtype)`
4. 为每个param通过奇葩方式[Now is officiallly supported by pytorch](https://github.com/pytorch/pytorch/issues/76464)注册AccumulateGrad的hook，从而在param.grad被计算出来之后，param.main_grad += param.grad，然后param.grad = None，让python GC回收param.grad。

使用contiguousbuffer的流程是：
1. 每次param.gard is None（因为每次 param.main_grad += param.gard，都会param.gard = None，所以每次param.grad都是None
2. 正常的pytorch反向计算图，并得到param.grad（详见pytorch/AccumulateGrad，param.grad为None情况下，param.gard = std:move(grad)，所以无开销
3. `param.main_grad.add_(param.grad.data), param.grad = None`

不使用的话，即一般情况下（假设不是第一次反向，param.grad已经存在） 则param.grad += grad

两者相比，使用连续的buffer的峰值内存为 `all_grad_size + 某个param.grad`，多了某个param.grad这一部分显存；所以峰值内存可能会稍高；但考虑到内存碎片减少了，所以实际内存可能会小，这个要具体分析。


## 可否用param.register_hook替代AccumulateGrad hook？
是否等价？很接近，自己跑2次都不一样，应该是等价，只是随机性没消除（实验配置见最后），根据flash_attn的说法，我用的版本还不能够消除随机性
| post_grad_accu run 1   | post_grad_accu run 2   | tensor_hook run 1   | tensor_hook run 2   |
|-------|-------|-------|-------|
| 1.227534E+01 | 1.227534E+01  | 1.227534E+01  |  1.227534E+01 |
| 1.225588E+01 | 1.225588E+01  | 1.225588E+01  |  1.225588E+01 |
| 1.226881E+01 | 1.226881E+01  | 1.226881E+01  |  1.226881E+01 |
| 1.214663E+01 | 1.214662E+01  | 1.214662E+01  |  1.214658E+01 |
| 1.182831E+01 | 1.182830E+01  | 1.182837E+01  |  1.182834E+01 |

但是为什么不用呢param.register_hook？可能因为AccumulateGrad hook是拿到最终param.grad结果之后，而param.register_hook拿到的还不是，可能用户又加了些grad hook。这是我的猜想。

如何替代？diff如下：
```
diff --git a/megatron/model/distributed.py b/megatron/model/distributed.py
index 382b307..69f0bc3 100644
--- a/megatron/model/distributed.py
+++ b/megatron/model/distributed.py
@@ -172,23 +172,34 @@ class DistributedDataParallel(DistributedDataParallelBase):
             for param in self.module.parameters():
                 if param.requires_grad:
                     # Expand so we get access to grad_fn.
-                    param_tmp = param.expand_as(param)
+                    if not hasattr(param, 'grad_fn'):
+                        print("param doesnot has grad_fn, so we use expand as")
+                    # param_tmp = param.expand_as(param)
                     # Get the gradient accumulator functtion.
-                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
-                    grad_acc.register_hook(self._make_param_hook(param))
-                    self.grad_accs.append(grad_acc)
+                    # grad_acc = param_tmp.grad_fn.next_functions[0][0]
+                    # print(f"{grad_acc=} {param_tmp.grad_fn=} {param_tmp.grad_fn.next_functions=} {param_tmp.grad_fn.next_functions[0]=}")
+                    # grad_acc.register_hook(self._make_param_hook(param))
+                    param.register_hook(self._make_param_hook(param))
+                    # self.grad_accs.append(grad_acc)
 
 
     def _make_param_hook(self, param):
         """Create the all-reduce hook for backprop."""
         # Hook used for back-prop.
-        def param_hook(*unused):
+        # def param_hook(*unused):
+        #     # Add the gradient to the buffer.
+        #     if param.grad is not None:
+        #         # The gradient function of linear layers is fused with GEMMs
+        #         param.main_grad.add_(param.grad.data)
+        #         # Now we can deallocate grad memory.
+        #         param.grad = None
+        def param_hook(grad):
             # Add the gradient to the buffer.
-            if param.grad is not None:
+            if grad is not None:
                 # The gradient function of linear layers is fused with GEMMs
-                param.main_grad.add_(param.grad.data)
-                # Now we can deallocate grad memory.
-                param.grad = None
+                param.main_grad.add_(grad.data)
+                return None
+            return grad
         return param_hook
```





配置如下：
```sh
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA3_70B_ARGS="--tensor-model-parallel-size 1 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 1 \
                 --tokenizer-type HFTokenizer \
                 --hf-tokenizer-path $HF_TOKENIZER_PATH \
                 --hidden-size 4096 \
                 --ffn-hidden-size 8192 \
                 --num-attention-heads 32 \
                 --init-method-std 0.01 \
                 --micro-batch-size 1 \
                 --global-batch-size 4 \
                 --seq-length 4096 \
                 --max-position-embeddings 8192 \
                 --lr 0.0002 \
                 --min-lr 1.0e-5 \
                 --lr-decay-style cosine \
                 --weight-decay 1e-2 \
                 --clip-grad 1.0 \
                 --initial-loss-scale 16 \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --adam-eps 1e-05 \
                 --train-iters 50000 \
                 --eval-iters 0 \
                 --lr-decay-iters 50000 \
                 --lr-warmup-fraction 0.002 \
                 --sequence-parallel \
                 --rmsnorm-epsilon 1e-5 \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --untie-embeddings-and-output-weights \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --disable-bias-linear \
                 --no-bias-gelu-fusion \
                 --no-position-embedding \
                 --no-masked-softmax-fusion \
                 --attention-softmax-in-fp32 \
                 --no-query-key-layer-scaling \
                 --multi-query-attention \
                 --multi-query-group-num 8 \
                 --use-flash-attn \
                 --memory-saving \
                 --use-distributed-optimizer \
                 --fused-rmsnorm \
                 --rotary-emb-base 500000 \
                 --no-gradient-accumulation-fusion \
                 --apply-rope-native \
                 --use-cpu-initialization \
                 --bf16"


OUTPUT_ARGS="--log-interval 1 \
             --detail-log-interval 1 \
             --exit-interval 5 \
              --timing-log-option all \
             --timing-log-level 2 \
             --save-interval 100 \
             --eval-interval 10000000"


OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --distributed-backend nccl"

```