# restrict 关键字的作用

简而言之，指针 `T *ptr` 加上了 `__restrict__` 关键字后，编译器会假设不会有其他的指针和`ptr`指向相同的地址，从而进行更大幅度的优化。

## 例子

源码见main.cpp，运行结果如下

```sh
> g++ main.cpp -o main -O2
> ./main
updatePtrs                             a: 101, b: 110, c: 100
updatePtrs_restrict                    a: 3, b: 10, c: 100
updatePtrs_restrict(aliasing pointers) a: 30
updatePtrs         (aliasing pointers) a: 40
```

通过objdump反汇编结果:
```sh
00000000000011e0 <_Z10updatePtrsPiS_S_>:
    11e0:       f3 0f 1e fa             endbr64 
    11e4:       8b 02                   mov    (%rdx),%eax
    11e6:       01 07                   add    %eax,(%rdi)
    11e8:       8b 02                   mov    (%rdx),%eax
    11ea:       01 06                   add    %eax,(%rsi)
    11ec:       c3                      retq   

```
没有使用`__restrict__`时，编译器会认为`a`, `b`, `c`可能指向相同的地址，所以编译器产生的汇编中两次加法操作是串行的。这是因为这4条mov add指令都使用了 `%eax` 寄存器，存在结构冲突（这里CPU会通过寄存器重命名排除这个冲突吗？）


```sh
00000000000011f0 <_Z19updatePtrs_restrictPiS_S_>:
    11f0:       f3 0f 1e fa             endbr64 
    11f4:       8b 02                   mov    (%rdx),%eax
    11f6:       01 07                   add    %eax,(%rdi)
    11f8:       01 06                   add    %eax,(%rsi)
    11fa:       c3                      retq   
    11fb:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
```
使用了`__restrict__`后，编译器会认为`a`, `b`, `c`不会指向相同的地址，所以编译器产生的汇编中两次加法操作是并行的。


#### 使用了`__restrict__`后，仍然传入aliasing的地址，则undefine behavior

如果违反了`__restrict__`的使用约定，则结果是undefine behavior，可能出错：
```c++
    a = 10;
    updatePtrs_restrict(&a, &a, &a);
    printf("updatePtrs_restrict(aliasing pointers) a: %d\n", a); // a: 30，而正确答案为40
```
根据汇编代码，可以推出30是这样得来的：
```
mov    (%rdx),%eax ;eax = 10
add    %eax,(%rdi) ;(%rdi)即*ptrA=10+10=20
add    %eax,(%rsi) ;(%rsi)即*ptrB=20+10=30,此时由于上一条add已经将a += 10了，所以*ptrB此时为20，add %eax后为30
```
