#include <stdio.h>
#include "hello.h" // 引用我们的动态库


int main() {
    hello(); // 调用库中的函数
    printf("Time is: %d\n", time); // 打印库中的变量
    return 0;
}
