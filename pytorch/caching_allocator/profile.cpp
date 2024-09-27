#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>


int main(){
    auto start_warmup = std::chrono::steady_clock::now();
    int *warmup;
    cudaMalloc((void**)&warmup, sizeof(int));
    cudaFree(warmup);//ignore the initialization
    auto end_warmup = std::chrono::steady_clock::now();



    auto start0 = std::chrono::steady_clock::now();

    
    

    int num = 100;
    int* hptr[num];
    for(int i = 0; i<num; i++){
        cudaMalloc((void**)&hptr[i], 1024 * 1024 * 240 );
    }
    auto end0 = std::chrono::steady_clock::now();

    
    // int temp;
    // scanf("%d", &temp);
    for(int i = 0; i<num; i++){
        printf("ptr[%d]=%p\n", i, hptr[i]);
        cudaFree(hptr[i]);
    }
    auto start1 = std::chrono::steady_clock::now();
    auto end1 = std::chrono::steady_clock::now();
    auto nano_warmup = std::chrono::duration_cast<std::chrono::nanoseconds>(end_warmup-start_warmup).count();
    auto nano0 = std::chrono::duration_cast<std::chrono::nanoseconds>(end0-start0).count();
    auto nano1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1-start1).count();
    // printf("time of memory clean (ms) .............................. %ld ............................ \n",nano0);
    printf("time of warmup (ms) = %f ............................ \n",(float)nano_warmup * 1e-6);
    printf("time of cudaMalloc (ms) = %f, avg (ms) = %.8f\n",(float)nano0 * 1e-6, (float)nano0 * 1e-6 / num);
    printf("time of cudaFree (ms) = %f, avg (ms) = %.8f \n",(float)nano1 * 1e-6, (float)nano1 * 1e-6 / num);
}