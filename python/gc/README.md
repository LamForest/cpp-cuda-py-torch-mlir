python中的内存管理分2种, 一种是gc, 另一种是引用计数


## 引用计数
引用计数通过在每个PyObject中都添加 ob_refcnt 
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241004230325.png)

当python的函数退出作用域时，会将local对象的ob_refcnt减1，如果为0，则调用析构函数。（这点是我猜测的，具体要看下cpython源码）。

## gc

引用计数众所周知存在瑕疵，所以CPython通过gc弥补了引用计数存在的缺陷。具体细节不展开。

对我而言，和引用计数最大的区别是：引用计数时即时生效的，函数退出作用域后，local对象立刻析构；而gc是不定时的，什么时候内存被释放，python用户无法感知。

## 例子

```py
#gc_test.py
"""
Reference: https://devguide.python.org/internals/garbage-collector/index.html
输出如下：
====== reference count works ========
malloc a : allocated: 4.0 memory (MB)
fa return : allocated: 0.0 memory (MB) #引用计数可以处理的tensor，在fa退出后，立刻归还给CCA(CudaCachingAllocator)
====== reference count fail, reference cycles ========
fb malloc self-ref list : allocated: 4.0 memory (MB)
fb return : allocated: 4.0 memory (MB) #而循环引用则等待垃圾回收
manually call gc.collect() : allocated: 0.0 memory (MB) #手动调用gc后，显存被释放
"""
import torch
import gc
def report_memory(name):
    mega_bytes = 1024.0 * 1024.0
    string = name + " : "
    cur_alloc = torch.cuda.memory_allocated()
    string += "allocated: {:.1f} memory (MB)".format(cur_alloc / mega_bytes)
    print("{}".format(string), flush=True)
    return cur_alloc


def main():
    print("====== reference count works ========")
    
    fa()
    cur_alloc = report_memory("fa return")
    assert cur_alloc == 0
    
    print("====== reference count fail, reference cycles ========")
    fb()
    cur_alloc = report_memory("fb return")
    assert cur_alloc == 4 * 1024 * 1024
    
    gc.collect()
    cur_alloc = report_memory("manually call gc.collect()")
    assert cur_alloc == 0
    
    
    

def fa():
    a:torch.Tensor = torch.empty(1024, 1024).cuda() # 4M
    report_memory("malloc a")

def fb():
    li = [torch.empty(1024, 1024).cuda()] #4M
    li.append(li) #make a reference cycle object
    
    report_memory("fb malloc self-ref list")
    #fb return, but li remain alive, cannot be detected by refcnt
    
if __name__ == '__main__':
    # for _ in range(1000):
    main()
```
