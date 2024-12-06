# RPATH RUNPATH patchelf

## 查找顺序：RPATH > LD_LIBRARY_PATH > RUNPATH > 系统lib目录

> https://blogs.oracle.com/solaris/post/changing-elf-runpaths-code-included 文章有点老，前面的部分可以看看。

The runtime linker looks in the following places, in the order listed, to find the sharable objects it loads into a process at startup time:

- If LD_LIBRARY_PATH (or the related LD_LIBRARY_PATH_32 and LD_LIBRARY_PATH_64) environment variables are defined, the directories they specify are searched to resolve any remaining dependencies.

- If the executable, or any sharable objects that are loaded, contain a runpath, the directories it specifies are searched to resolve dependencies for those objects. （所以一个在前面加载的so，如果so中存在runpath/rpatch，会影响到后面的so的查找，这种不确定性可能会导致一些潜在的bug）

- Finally, it searches two default locations for any remaining dependencies: /lib and /usr/lib (or /lib/64 and /usr/lib/64 for 64-bit code).

The above scheme offers a great deal of flexibility, and it usually works well. There is however one notable exception — the "Runpath Problem". The problem is that many objects are not built with a correct runpath, and once an object has been built, it has not been possible to change it. It is common to find objects where the runpath **is correct on the system the object was built on, but not on the system where it is installed**.

总的来说，RPATH > LD_LIBRARY_PATH > RUNPATH > 系统lib目录（/lib, /usr/lib，...）

## 例子：通过readelf -d 查看so使用的是RPATH还是RUNPATH
pytorch 使用的是RPATH，而不是RUNPATH：
```sh
root@thanos:/opt/conda/lib/python3.10/site-packages/torch/lib# readelf -d libc10_cuda.so

Dynamic section at offset 0x62a68 contains 32 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libc10.so]
 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.11.0]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libc10_cuda.so]
 0x000000000000000f (RPATH)              Library rpath: [$ORIGIN:$ORIGIN/../../../..]
 0x000000000000000c (INIT)               0x10000
 0x000000000000000d (FINI)               0x51460
```
所以这导致了torch是用的cuda library，永远是conda里面安装的，而不是/usr/local/cuda*/，不管怎么修改LD_LIBRARY_PATH，都是无效的。  



再看一个一般的so，用的则是RUNPATH:

```sh
root@thanos:/usr/local/cuda-11.7/lib64# readelf -d libcurand.so

Dynamic section at offset 0x3f94fb8 contains 36 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [librt.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libcurand.so.10]
 0x0000000000000010 (SYMBOLIC)           0x0
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]
```

可以通过`patchelf --set-rpath '$ORIGIN' *.so`来修改RUNPATH/RPATH，虽然命令是set-rpath，但实际上也可以修改RUNPATH。 

> patchelf其他用法：
```sh
syntax: patchelf
  [--set-interpreter FILENAME]
  [--page-size SIZE]
  [--print-interpreter]
  [--print-soname]              Prints 'DT_SONAME' entry of .dynamic section. Raises an error if DT_SONAME doesn't exist
  [--set-soname SONAME]         Sets 'DT_SONAME' entry to SONAME.
  [--set-rpath RPATH]
  [--remove-rpath]
  [--shrink-rpath]
  [--allowed-rpath-prefixes PREFIXES]           With '--shrink-rpath', reject rpath entries not starting with the allowed prefix
  [--print-rpath]
  [--force-rpath]
  [--add-needed LIBRARY]
  [--remove-needed LIBRARY]
  [--replace-needed LIBRARY NEW_LIBRARY]
  [--print-needed]
  [--no-default-lib]
  [--debug]
  [--version]
  FILENAME
```

## patchelf的原理：RPATH/RUNPATH指向.dynstr中新的字符串

elf中所有使用的字符串都位于 .dynstr段中，可以通过readelf -S *.so查看：`[]`中的代表在dynstr段中的偏移地址。
```sh
String dump of section '.dynstr':
  [     1]  __gmon_start__
  [    10]  _ITM_deregisterTMCloneTable
  [    2c]  _ITM_registerTMCloneTable
  [    46]  __cxa_finalize
  [    55]  _Unwind_Resume
  [    64]  _ZN5MLIR9LogStreamC1Ev
  [    7c]  _ZN5MLIR9LogWritereoERKNS_9LogStreamE
  [    a3]  _ZN8MLIR_rt15xpuWorkspaceGetEii
  [    c4]  _ZN8MLIR_rt17xpuL3WorkspaceGetEi
  [    e6]  _ZN8MLIR_rt19xpuWorkspaceReserveEim
  [   10b]  _ZN8MLIR_rt21xpuL3WorkspaceReserveEim
  ...
  [  4ddd]  $ORIGIN
  [  4de5]  XXXXXXXXXXXXXXXXX
```
比如所有c++ mangle的符号都位于.dynstr中，

patchelf在修改RUNPATH/RPATH时，实际上是**将RUNPATH/RPATH中指向.dynstr中的一串字符串**；问题在于，如果dynstr不存在新的RUNPATH的字符串时，patchelf还能成功吗？比如`patchelf --set-rpath 'hellohellohellohello' *.so`。答案是可以的，我们先记录下patchelf之前的 so size和.dynstr大小：
```sh
> readelf -d ./libMLIRRuntime.so
Dynamic section at offset 0xe5c90 contains 39 entries:
  Tag        Type                         Name/Value
 ...
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libMLIRRuntime.so]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN]

> ls -al libMLIRRuntime.so
-rwx--x--x 1 root root 1061664 Jan 29 13:59 libMLIRRuntime.so

> readelf -S libMLIRRuntime.so               
There are 31 section headers, starting at offset 0xfca90:
Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  ...
  [26] .strtab           STRTAB           0000000000000000  000ee110
       000000000000e869  0000000000000000           0     0     1
  [27] .shstrtab         STRTAB           0000000000000000  000fc979
       0000000000000116  0000000000000000           0     0     1
  [28] .dynstr           STRTAB           00000000000e8000  000fe000
       0000000000004df7  0000000000000000   A       0     0     8
  [29] .gnu.hash         GNU_HASH         00000000000ecdf8  00102df8
       00000000000004fc  0000000000000000   A       1     0     8
  [30] .note.gnu.build-i NOTE             00000000000ed2f8  001032f8
       0000000000000024  0000000000000000   A       0     0     8

> readelf -p .dynstr libMLIRRuntime.so       
String dump of section '.dynstr':
  ...
  [  4d59]  GLIBCXX_3.4
  [  4d65]  GLIBCXX_3.4.20
  [  4d74]  CXXABI_1.3.2
  [  4d81]  XXXXXXX
  [  4d89]  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  [  4ddd]  $ORIGIN
  [  4de5]  XXXXXXXXXXXXXXXXX
```



然后修改RUNPATH/RPATH，再查看so size和.dynstr大小：
```sh
> patchelf --set-rpath 'hellohellohello' libMLIRRuntime.so

> readelf -d ./libMLIRRuntime.so

Dynamic section at offset 0xe5c90 contains 39 entries:
  Tag        Type                         Name/Value
 ...
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libMLIRRuntime.so]
 0x000000000000001d (RUNPATH)            Library runpath: [hellohellohello] #修改成功

> ls -al libMLIRRuntime.so
-rwx--x--x 1 root root 1084936 Jan 29 14:11 libMLIRRuntime.so

> readelf -S libMLIRRuntime.so                    
There are 31 section headers, starting at offset 0xfca90:

Section Headers:
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  ...
  [27] .shstrtab         STRTAB           0000000000000000  000fc979
       0000000000000116  0000000000000000           0     0     1
  [28] .gnu.hash         GNU_HASH         00000000000ecdf8  00102df8 （offset不变）.dynstr原来位于.shstrtab 和 .gun.bash之间，现在移到了最后，但是.gun.bash和.note.gnu.build-i的offset不变，说明原来的.dynstr段的位置现在没人用，浪费了。
       00000000000004fc  0000000000000000   A       1     0     8
  [29] .note.gnu.build-i NOTE             00000000000ed2f8  001032f8（offset不变）
       0000000000000024  0000000000000000   A       0     0     8
  [30] .dynstr           STRTAB           00000000000ee000  00104000（新的.dynstr段在elf后面新增的）
       0000000000004e07  0000000000000000   A       0     0     8

> readelf -p .dynstr libMLIRRuntime.so       
String dump of section '.dynstr':
  ...
  [  4d81]  XXXXXXX
  [  4d89]  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  [  4ddd]  XXXXXXX
  [  4de5]  XXXXXXXXXXXXXXXXX
  [  4df7]  hellohellohello
```

比较patch前后的文件大小、elf header、dynstr段，可以发现以下几点变化：
- patchelf之后，so变大了；dynstr的大小也变大了，从4df7 -> 4e07
- dynstr段移到了最后一段，原来放.dynstr的位置空置了，这算是一种浪费，所以尽量不要随便patchelf。
- elf文件用不到$ORIGIN这个字符串了，所以从.dynstr段中删除了$ORIGIN字符串，以节约elf空间。

所以当patchelf修改RUNPATH/RPATH时，如果RUNPATH/RPATH中不存在新的字符串，那么patchelf会在dynstr段中添加新的字符串，这会使得dynstr段的大小变大，并移至最后，同时也会使得so size变大。

#### elf加载到进程中后，只有strtab没有dynstr段了，.dynstr会合并到strtab段

事情起因是`readelf -S libhello.so` 和 `readelf -p .dynstr libhello.so` 都能确切的看到.dynstr段（以及dynsym段）；但PT_DYNAMIC段中d_type的宏枚举只有DT_STRTAB，而没有DT_DYNSTR。

```
> readelf -S libhello.so
  [ 4] .dynsym           DYNSYM           0000000000000318  00000318
       00000000000000c0  0000000000000018   A       5     1     8
  [ 5] .dynstr           STRTAB           00000000000003d8  000003d8
       000000000000007b  0000000000000000   A       0     0     1
  [ 6] .gnu.version      VERSYM           0000000000000454  00000454
       0000000000000010  0000000000000002   A       4     0     2

> readelf -p .dynstr libhello.so
String dump of section '.dynstr':
  [     1]  __gmon_start__
  [    10]  _ITM_deregisterTMCloneTable
  [    2c]  _ITM_registerTMCloneTable
  [    46]  __cxa_finalize
  [    55]  time
  [    5a]  hello
  [    60]  puts
  [    65]  libc.so.6
  [    6f]  GLIBC_2.2.5
```

在github上看到这么一段注释：
![](https://raw.githubusercontent.com/LamForest/pics/main/obsidian/20241208184849.png)
感觉挺合理的，对于.dynsym也是一样的道理。




## cmake设置rpath
相比于直接对elf文件进行修改，更好的的方式当然是在编译时就设置RUNPATH为合理的值；还有少数情况，比如torch，希望强制用户使用某个路径，而不允许通过LD_LIBRARY_PATH来修改，那么可以设置RPATH。

```cmake
set_target_properties(${LIB_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH OFF
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
```
