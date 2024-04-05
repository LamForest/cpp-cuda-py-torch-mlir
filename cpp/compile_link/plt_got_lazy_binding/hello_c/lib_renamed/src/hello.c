#include <stdio.h>
#include "hello.h"

// 导出的全局变量
int time = 10;

// 导出的函数
void hello_renamed() {
    printf("Hello, world!\n");
}