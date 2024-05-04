
最近正在使用[gpustat](https://github.com/wookayin/gpustat)这个工具。

![gpustat示例](https://github.com/wookayin/gpustat/blob/master/screenshot.png)

这个工具存在一个限制，刷新的间隔存在一个**下限**，假如设置`gpustat -i 0.1`，刷新间隔并不是0.1s，而是约0.9s。这是一个挺奇怪的事，查询gpu的状态不至于有这么大的开销。

一开始我打算使用最笨的方法，`time.time()`打点。即使对于`gpustat`这个相对较小的工具，也是相当麻烦，要一层层的在函数调用栈中打点，并且要修改源码。于是我尝试搜了下类似的工具，发现有一个叫[cProfile](https://docs.python.org/3/library/profile.html)的库。

这个库有2种使用方式：

1. 直接使用`python -m cProfile xxx.py`，
2. 在代码中导入`cProfile`，然后使用`pr = cProfile.Profile()`来创建对象，然后调用`pr.enable(); ...; pr.disable();` 手动设定profile范围。

这里主要介绍第一种方式。

## 表格输出
`python -m cProfile -s cumtime  -m gpustat`

其中的 -s 的意思是 sort。常用的 sort 类型有两个：

1. tottime，指的是函数本身的运行时间，扣除了子函数的运行时间
2. cumtime，指的是函数的累计运行时间，包含了子函数的运行时间

这两个时间都相当重要，都需要关注。

```
106115 function calls (102804 primitive calls) in 1.008 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     64/1    0.000    0.000    1.009    1.009 {built-in method builtins.exec}
        1    0.000    0.000    1.008    1.008 <string>:1(<module>)
        1    0.000    0.000    1.008    1.008 runpy.py:199(run_module)
        1    0.000    0.000    0.944    0.944 runpy.py:63(_run_code)
        1    0.000    0.000    0.944    0.944 __main__.py:1(<module>)
        1    0.000    0.000    0.944    0.944 cli.py:108(main)
        1    0.000    0.000    0.941    0.941 cli.py:57(print_gpustat)
        1    0.000    0.000    0.922    0.922 core.py:445(new_query)
        8    0.000    0.000    0.876    0.110 core.py:458(get_gpu_info)
        8    0.801    0.100    0.801    0.100 {built-in method time.sleep}
        7    0.000    0.000    0.093    0.013 __init__.py:1(<module>)
       72    0.000    0.000    0.073    0.001 core.py:505(_wrapped)
```

在上面的表格中，可以看到`{built-in method time.sleep}`共占了0.8s的时间。在gpustat的源码中，全局搜一下time.sleep，发现在`core.py:458(get_gpu_info)`中调用了time.sleep：

![image.png](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240504153959.png)

[PR](https://github.com/wookayin/gpustat/pull/65/files)

原因没有细看，这大概是psutil的一个限制，gpustat通过psutil来获取cpu进程的信息。

## 火焰图输出

```
pip install snakeviz
python -m cProfile -o a.prof  -m gpustat
snakeviz a.prof
```

![image.png](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20240504155047.png)

## 参考
https://zhuanlan.zhihu.com/p/53760922