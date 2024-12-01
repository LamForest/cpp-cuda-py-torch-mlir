使用gcc版本：
```
╰─ gcc --version
gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```


## plt + got
在hello_c目录下：
```sh
╰─ gcc -shared -fPIC -o lib/libhello.so lib/src/hello.c -Ilib/include #编译so
    gcc -o main main.c lib/libhello.so -I lib/include #编译main
```

```
╰─ readelf -S main | egrep  '.plt|.got'
  [11] .rela.plt         RELA             0000000000000650  00000650
  [13] .plt              PROGBITS         0000000000001020  00001020
  [14] .plt.got          PROGBITS         0000000000001050  00001050
  [15] .plt.sec          PROGBITS         0000000000001060  00001060
  [24] .got              PROGBITS         0000000000003fb0  00002fb0
```
这里看其中的`.plt.sec .got`，其他的我暂时也不太理解含义
```sh
╰─ objdump -d main -j .plt.sec
main:     file format elf64-x86-64

Disassembly of section .plt.sec:

0000000000001060 <printf@plt>:
    1060:       f3 0f 1e fa             endbr64 
    1064:       f2 ff 25 5d 2f 00 00    bnd jmpq *0x2f5d(%rip)        # 3fc8 <printf@GLIBC_2.2.5>
    106b:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)

0000000000001070 <hello@plt>:
    1070:       f3 0f 1e fa             endbr64 
    1074:       f2 ff 25 55 2f 00 00    bnd jmpq *0x2f55(%rip)        # 3fd0 <hello>
    107b:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)


╰─ objdump -d main -j .got.plt
main:     file format elf64-x86-64


Disassembly of section .got:

0000000000003fb0 <_GLOBAL_OFFSET_TABLE_>:
    3fb0:       b0 3d 00 00 00 00 00 00 00 00 00 00 00 00 00 00     .=..............
        ...
    3fc8:       30 10 00 00 00 00 00 00 40 10 00 00 00 00 00 00     0.......@.......
        ...
```
观察`.plt.sec`的内容，共有2项，每一项并不是一个指针，而是一段代码，`endbr64`是intel防止缓冲区溢出等攻击提出的指令，`nopl   0x0(%rax,%rax,1)`为空操作；唯一有用的是`bnd jmpq *0x2f5d(%rip) `，含义为从GOT表(`0x2f5d(%rip)`)中取出目标地址并跳转，其实就是`0x3fc8`指向了got表中的第3项。


观察`.got`即GLOBAL_OFFSET_TABLE(GOT)，从0x3fb0起始，到0x3fc8结束，共10项 80个字节：
- 其中0x3fc8和0x3fd0这两项将要填充的是printf和hello的函数地址。前面3项的作用根据《程序员的自我修养》，可能是：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241201143156.png)
- GOT中目前是无意义的数据，因为只有elf被加载到内存中时，真正函数地址才会被写到GOT中。这个时机可以是elf加载时，或者符号使用时（延迟绑定），详见后文。


继续objdump代码段，观察**代码段**如何和plt、got交互的：

>这里省略了.text其他部分，只保留了main函数。其他部分不关注

```sh
╰─ objdump -d main -j .text
main:     file format elf64-x86-64
Disassembly of section .text:

0000000000001169 <main>:
    1169:       f3 0f 1e fa             endbr64 
    116d:       55                      push   %rbp
    116e:       48 89 e5                mov    %rsp,%rbp
    1171:       b8 00 00 00 00          mov    $0x0,%eax
    1176:       e8 f5 fe ff ff          callq  1070 <hello@plt>
    117b:       8b 05 8f 2e 00 00       mov    0x2e8f(%rip),%eax        # 4010 <time>
    1181:       89 c6                   mov    %eax,%esi
    1183:       48 8d 3d 7a 0e 00 00    lea    0xe7a(%rip),%rdi        # 2004 <_IO_stdin_used+0x4>
    118a:       b8 00 00 00 00          mov    $0x0,%eax
    118f:       e8 cc fe ff ff          callq  1060 <printf@plt>
    1194:       b8 00 00 00 00          mov    $0x0,%eax
    1199:       5d                      pop    %rbp
    119a:       c3                      retq   
    119b:       0f 1f 44 00 00          nopl   0x0(%rax,%rax,1)
```

由于hello和printf都是外部函数，所以这里会调用`.plt.sec`的代码，而`.plt.sec`会查GOT表，跳转到正确的地址。

## Q：为什么要使用plt和got间接跳转？可以直接挑战吗
可不可以不用plt+got，而是在加载动态库时，根据加载的地址，改写.text中的callq的目标地址？

不行，有3点弊端：
- `.text`一般不允许改写，存在安全威胁和漏洞。
- 在不同进程中，动态库的地址都是不同的并且动态库在所有进程中共享，直接改写`.text`会对所有进程的`.text`造成影响，这就违背了动态库共享的目的。
- 假设一个so中对printf有100k次调用，如果直接改写`.text`，那么100k次调用都需要改写；而采用plt+got，只用改写一次。



## Q：一个进程中，不同so和main之间的got、plt是共享的吗？
这个我觉得是可以合并的，一个进程只有一张merged的plt、got表。但是，我通过gdb看到的，每个so有自己的plt和got。而且，如果不是独立的，cuda_mock就无法根据so的名字筛选要被替换的got表了。

![plt、got是否共享](image.png)
图源：https://zhuanlan.zhihu.com/p/558522498
原因在这个zhihu文章中解释的比较清楚，text中对plt的callq指令是在编译成so时，就已经确定了，之后不再更改，所以text到plt的的相对位置是固定的，plt的起始地址不可变。

> 先说一个引子:
问: 我们最初成立.got段的目的是什么?
>
> 答: 把代码中引用一个不确定的地址的情况下,需要重定位修改代码部分的重定位过程导致代码段不可变不成立的:地址引用,将其从代码中剥离出来.单独放到数据段. 就像代码段中对一个变量的绝对地址引用,修改为从数据段的一个地址指针的间接引用. 这样才能保证代码段的内容始终保持不变. 然后数据段.GOT的内容动态的重定位到相应的目标符号.
>
> 其实有了上面一段描述的推理后, 那.got是否属于进程中所有共享对象统一使用.这个结论已经很明确了.那就是.got段是属于模块自身的一个数据段.而不是所有的模块共享. 从上面的.got 段会被一个代码段的地址相对地址引用. 那就决定了.got必须与相对应的代码段保持稳定的相对距离. 这样才能被代码用相对地址定位的方式进行访问到. 而另外一方面,不同的代码段之间的相对位置又是不确定的. 这就间接说明了, 不同共享对象的之间的代码段的相对地址不确定. 所以不同的代码段(不同的模块)之间使用的.got不是同一个表.虽然.got被称作全局偏移表: global offset table. 但是他只属于模块本身.

## Q：不同进程中，同一个so的got、plt是共享的吗？
got肯定不是共享的，so在不同的进程中，加载的位置不一样，got中存放的地址也不一样。但是plt呢？不确定。

## Q：为什么要使用2层跳转，而不是.text直接从GOT中查表，只一层跳转？
不清楚，TODO

## Q：延迟绑定生效吗？
其实默认gcc是不启用lazy binding的，需要手动开启。

#### 实验1 gcc默认不启用lazy binding

```sh
cd hello_c
#正常编译libhello.so
gcc -shared -fPIC -o lib/libhello.so lib/src/hello.c -Ilib/include
#正常编译main，-g是为了调试
gcc  -g  main.c -o main -Llib  -lhello -Ilib/include
#编译一个没有hello符号的so，名字也叫libhello.so
gcc -shared -fPIC -o lib_renamed/libhello.so lib_renamed/src/hello.c -Ilib_renamed/include
#指定使用没有hello符号的so，进入gdb
LD_LIBRARY_PATH=`pwd`/lib_renamed:$LD_LIBRARY_PATH gdb ./main
(gdb) b main
Breakpoint 1 at 0x1169: file main.c, line 5.
(gdb) r
Starting program: /home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main 
/home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main: symbol lookup error: /home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main: undefined symbol: hello
```

可以看到，还没有进入到main，更没有调用hello，就报错了；这说明不是延迟绑定

#### 实验2 gcc启用lazy binding

```sh
# ... 前面步骤一样
#正常编译main，-g是为了调试
gcc -z lazy -g  main.c -o main -Llib  -lhello -Ilib/include
# ... 后面步骤一样，同样使用错误的so，启动gdb
LD_LIBRARY_PATH=`pwd`/lib_renamed:$LD_LIBRARY_PATH gdb ./main 
(gdb) b main
Breakpoint 1 at 0x1169: file main.c, line 5.
(gdb) r
Starting program: /home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main 

Breakpoint 1, main () at main.c:5
5       int main() {
```
这次在进入main之后，也没有报错。继续单步，调用hello：
```sh
(gdb) n
6           hello(); // 调用库中的函数
(gdb) n
/home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main: symbol lookup error: /home/github/cpp-cuda-py-torch-mlir/cpp/compile_link/plt_got_lazy_binding/hello_c/main: undefined symbol: hello
[Inferior 1 (process 106339) exited with code 0177]
```
当真正调用到hello时，触发延迟绑定，才报错了。

可以通过`LD_BIND_NOW=1 LD_LIBRARY_PATH=`pwd`/lib:$LD_LIBRARY_PATH gdb ./main`强制不使用延迟绑定。但是不能通过`LD_BIND_NOW=0 LD_LIBRARY_PATH=`pwd`/lib:$LD_LIBRARY_PATH gdb ./main` 强制不使用延迟绑定。


#### 参考
为什么默认不启用lazy binding？可能是安全因素

https://stackoverflow.com/questions/62527697/why-does-gcc-link-with-z-now-by-default-although-lazy-binding-is-the-default

> When BIND_NOW is enabled, all symbols will be resolved before executing the program code. The .got.plt section is merged into .got section by ld. The ld.so changes the .got section to read-only before calling program entry point.



## Q：编译、链接时，需要动态库的头文件和so吗？

头文件编译时需要，通过 -I 指定查找目录。

so链接时需要，通过 -L 指定查找目录，通过-l指定so名字。但是为什么？如果没有的话可以链接吗？感觉也可以。。。但根据程序员的自我修养，确实链接时是需要对应的so的：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241201125307.png)

## `-shared` 与 `-fPIC`

通常，我们都是同时使用这两个参数，比如编译动态库时：`gcc -shared -fPIC -o libhello.so hello.c`。那这两个参数的含义是什么？可以不同时使用吗？

1. `-shared` 参数告诉编译器，我们正在编译一个共享库（Shared Library）。在so被装载时，会进行装载时重定位（Load-time relocation），即在运行时，动态链接器（ld.so）会将so中的代码段、数据段的符号地址替换为实际的运行时地址。

但这会引入不必要的内存消耗，因为理论上，动态链接库的代码段是可以在多个进程中共享的。

![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241201130345.png)

2. 所以这引入了 `-fPIC` 参数，即生成位置无关代码（Position Independent Code）。这样，动态库的代码段可以被多个进程共享，而不需要为每个进程单独加载一份代码。这通过GOT实现。

