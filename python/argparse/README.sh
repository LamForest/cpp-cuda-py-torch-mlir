
所有True/False参数的默认值，在未显式指定的情况下为False；
所以tp_gemm_parallel的默认值为False；
而seq_par是store_false，所以取反了，默认值是True

```python
python store_true_false.py 

echo "-------"
python store_true_false.py --tp-gemm-parallel

echo "-------"
python store_true_false.py --tp-gemm-parallel --no-seq-par
```

```shell
The value of --tp-gemm-parallel is: False
The value of --seq-par is: True
-------
The value of --tp-gemm-parallel is: True
The value of --seq-par is: True
-------
The value of --tp-gemm-parallel is: True
The value of --seq-par is: False
```
