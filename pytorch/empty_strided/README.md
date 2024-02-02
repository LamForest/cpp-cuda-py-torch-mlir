# 理解empty_strided算子含义

## 算子文档
https://pytorch.org/docs/stable/generated/torch.empty_strided.html

```python
torch.empty_strided(size, stride, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False) → Tensor
```
> Creates a tensor with the specified size and stride and filled with undefined data.
> If the constructed tensor is “overlapped” (with multiple indices referring to the same element in memory) its behavior is undefined.

## 以torch.empty_strided(shape=(5,5), stride=(2,3))

