# basic gdb usage
> gdb --args python your_script.py
> b xxx
> r
> info b
(gdb) info b
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   <MULTIPLE>
1.1                         y   0x00000000004249ed in printf at /croot/python-split_1694437953337/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include/bits/stdio2.h:104
1.2                         y   0x00000000004426e8 in printf at /croot/python-split_1694437953337/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include/bits/stdio2.h:104
1.3                         y   0x0000000000442727 in printf at /croot/python-split_1694437953337/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include/bits/stdio2.h:104

> delete 1 # 1.1 1.2也会跟着被删掉

# symbol belongs to which shared library
```
(gdb) info sharedlibrary
From                To                  Syms Read   Shared Object Library
0x00007ffff7fd0100  0x00007ffff7ff2684  Yes         /lib64/ld-linux-x86-64.so.2
0x00007ffff7fa7ae0  0x00007ffff7fb7535  Yes         /lib/x86_64-linux-gnu/libpthread.so.0
0x00007ffff7f9c220  0x00007ffff7f9d179  Yes         /lib/x86_64-linux-gnu/libdl.so.2
0x00007ffff7f973e0  0x00007ffff7f97d90  Yes         /lib/x86_64-linux-gnu/libutil.so.1
0x00007ffff7e543c0  0x00007ffff7efafa8  Yes         /lib/x86_64-linux-gnu/libm.so.6
0x00007ffff7c77630  0x00007ffff7dec29d  Yes         /lib/x86_64-linux-gnu/libc.so.6 #__fprintf
0x00007ffff773d050  0x00007ffff773f961  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_heapq.cpython-38-x86_64-linux-gnu.so
0x00007ffff7722050  0x00007ffff772f671  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_ctypes.cpython-38-x86_64-linux-gnu.so
0x00007ffff770d000  0x00007ffff7715791  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/../../libffi.so.8
0x00007ffff76fe050  0x00007ffff77020d1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_struct.cpython-38-x86_64-linux-gnu.so
0x00007ffff76ad050  0x00007ffff76b4ba1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/math.cpython-38-x86_64-linux-gnu.so
0x00007ffff7826050  0x00007ffff78263d1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_opcode.cpython-38-x86_64-linux-gnu.so
0x00007ffff781d050  0x00007ffff781fb11  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/zlib.cpython-38-x86_64-linux-gnu.so
0x00007ffff7505050  0x00007ffff7516241  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/../../libz.so.1
0x00007ffff74ad050  0x00007ffff74bbe41  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_bz2.cpython-38-x86_64-linux-gnu.so
0x00007ffff7812050  0x00007ffff7815311  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_lzma.cpython-38-x86_64-linux-gnu.so
0x00007ffff747e050  0x00007ffff749abb1  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/../../liblzma.so.5
0x00007ffff780b050  0x00007ffff780b871  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/grp.cpython-38-x86_64-linux-gnu.so
0x00007ffff7476050  0x00007ffff7476861  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_bisect.cpython-38-x86_64-linux-gnu.so
0x00007ffff746d050  0x00007ffff7471ca1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_sha512.cpython-38-x86_64-linux-gnu.so
0x00007ffff7468050  0x00007ffff7468ee1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_random.cpython-38-x86_64-linux-gnu.so
0x00007ffff7421050  0x00007ffff74221e1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/_posixsubprocess.cpython-38-x86_64-linux-gnu.so
0x00007ffff7419050  0x00007ffff741afb1  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/lib-dynload/select.cpython-38-x86_64-linux-gnu.so
0x00007ffff7412040  0x00007ffff74120f9  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_global_deps.so
0x00007ffff18234a0  0x00007ffff192b2da  Yes (*)     /usr/local/cuda-11.7/lib64/libcurand.so.10
0x00007fffe8db5c40  0x00007fffe91b13aa  Yes (*)     /usr/local/cuda-11.7/lib64/libcufft.so.10
0x00007fffdfbea0a0  0x00007fffe055d48a  Yes (*)     /usr/local/cuda-11.7/lib64/libcublas.so.11
...
0x00007fffcb672060  0x00007fffcb672129  Yes         /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/_C.cpython-38-x86_64-linux-gnu.so
0x00007fffc9506bd0  0x00007fffc9fab450  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_python.so
0x00007fffc92f3880  0x00007fffc92f5cc0  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libshm.so
0x00007fffc92ec040  0x00007fffc92ec0f9  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch.so
0x00007fffbe0d3f60  0x00007fffc72ac380  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so #at::_ops::clone::redispatch
0x00007fff979f0530  0x00007fff99dfa672  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so
0x00007fff96c18420  0x00007fff96c5b53a  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libc10_cuda.so
0x00007fff96b86be0  0x00007fff96be0c01  Yes (*)     /root/miniconda/envs/python38_torch201_cuda/lib/python3.8/site-packages/torch/lib/libc10.so

#根据范围来看，0x7ffff7cb6bd0在libc.so.6的范围内

(gdb) p __fprintf
$7 = {int (FILE *, const char *, ...)} 0x7ffff7cb6bd0 <__fprintf>


#C++的符号怎么判断？
(gdb) p at::_ops::clone::redispatch
No type "_ops" within class or namespace "at".
#要加个引号 gdb会去将''内的作为代码执行，参考：https://stackoverflow.com/a/24958777
#根据范围来看，在libtorch_cpu.so中
(gdb) p 'at::_ops::clone::redispatch' 
$12 = (<text from jump slot in .got.plt, no debug info>) 0x7fffbf65af00 <at::_ops::clone::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::optional<c10::MemoryFormat>)>

#为什么有些符号的地址明显不对，地址偏移太小了:
(gdb) p _PyEval_EvalFrameDefault
$9 = {PyObject *(PyFrameObject *, int)} 0x4d8110 <_PyEval_EvalFrameDefault>
(gdb) p Py_BytesMain
$11 = {int (int, char **)} 0x579e50 <Py_BytesMain>
```

