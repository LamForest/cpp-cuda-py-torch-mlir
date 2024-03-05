# 理解argmax的dim参数

## 2维
先考虑一个简单的tensor：`a = torch.randn(2,3)`

`torch.argmax(a, dim=0)，dim=0`的语义为，固定其他dim，在dim 0上进行argmax：
- `output[0] = argmax(a[0, 0], a[1, 0])`
- `output[1] = argmax(a[0, 1], a[1, 1])`
- `output[2] = argmax(a[0, 2], a[1, 2])`

可以看作把a分为3组，每组2个元素，组内进行argmax，最后将分组argmax的结果拼接起来，得到一个shape=(3,)的output tensor

## 通用情况
再考虑一个tensor：`a = torch.randn(s1, s2, ..., si, ..., sn)`，其中s1, s2, ..., si, ..., sn是shape的各个维度。

torch.argmax(a, dim=i)的语义为，固定其他dim，在dim i上进行argmax；
- `output[0, ..., 0] = argmax(a[0, ...,0(dim=i), ..., 0], a[0, ...,1(dim=i), ..., 0], a[0, ...,2(dim=i), ..., 0], ..., a[0, ...,si(dim=i), ..., 0])`
- `output[1, ..., 0] = argmax(a[1, ...,0(dim=i), ..., 0], a[1, ...,1(dim=i), ..., 0], a[1, ...,2(dim=i), ..., 0], ..., a[1, ...,si(dim=i), ..., 0])`
- ...
- `output[s1, ..., sn] = argmax(a[s1, ...,0(dim=i), ..., sn], a[s1, ...,1(dim=i), ..., sn], a[s1, ...,2(dim=i), ..., sn], ..., a[s1, ...,si(dim=i), ..., sn])`

可以看作把a分为`s1 * s2 * ... * si-1 * si+1 * ... * sn`组，每组si个元素，组内进行argmax，最后将分组argmax的结果拼接起来，得到一个shape=`(s1, s2, ..., si-1, si+1, ..., sn)`的tensor。

## keepdim的含义
根据通用情况，可以将argmax(dim=i)，看作在i维度上，进行reduce操作，将`(s1, s2, ..., si, ..., sn)`形状的tensor，转换为`(s1, s2, ..., si-1, si+1, ..., sn)`形状的tensor。argmax完成后，dim=i的维度消失了。

有没有办法保留第i个维度呢？将keepdim设为True，此时输出Tensor的形状为`(s1, s2, ..., si-1, 1, si+1, ..., sn)`，本来第i个维度消失，现在第i个维度变为了1（因为本质上是reduce操作，所以维度为1）。

所以，keepdim对输出没有任何影响，仅仅保留了dim维度，并始终为1。

## dim参数的使用场景
设想一个推理任务，网络输出的output的形状为BxC，其中B是batch size，C是类别数。此时，需要使用argmax来得到每个样本的类别：`torch.argmax(output, dim=1)`，得到的结果形状为`(B,)`。

## 在其他算子上，dim参数的意义
除了argmax算子，还有其他算子的dim参数，也是类似的，比如mean, sum。在这些算子中，dim参数的含义，也是在dim维度上进行reduce操作。