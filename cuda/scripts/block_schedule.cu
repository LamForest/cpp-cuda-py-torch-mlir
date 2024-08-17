/*

假设SM 2,3  4,5 6,7每2个物理上是一个group，占用一些共享资源；那么加入 blocknum < SM count，那么将block调度到不相邻的SM上，会避免相邻SM产生竞争

另外SM 0并不会优先被调度到，难道SM 0有什么特殊作用？

======= SM count: 56, launch 32 block, each block has 1 threads ========block 0 executed on SM 0
block 1 executed on SM 2
block 2 executed on SM 4
block 3 executed on SM 6
block 4 executed on SM 8
block 5 executed on SM 10
block 6 executed on SM 12
block 7 executed on SM 14
block 8 executed on SM 16
block 9 executed on SM 18
block 10 executed on SM 20
block 11 executed on SM 22
block 12 executed on SM 24
block 13 executed on SM 26
block 14 executed on SM 28
block 15 executed on SM 30
block 16 executed on SM 32
block 17 executed on SM 34
block 18 executed on SM 36
block 19 executed on SM 38
block 20 executed on SM 40
block 21 executed on SM 42
block 22 executed on SM 44
block 23 executed on SM 46
block 24 executed on SM 48
block 25 executed on SM 50
block 26 executed on SM 52
block 27 executed on SM 54
block 28 executed on SM 1
block 29 executed on SM 3
block 30 executed on SM 5
block 31 executed on SM 7

*/


#include <stdio.h>

#include <stdlib.h>

#include <cuda.h>

/* E.D. Riedijk */

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %%smid;" : "=r"(ret) );

     return ret;

}

__global__ void kern(int *sm){

   if (threadIdx.x==0)

      sm[blockIdx.x]=get_smid();

}

int main(int argc, char *argv[]){

    int N = atoi(argv[1]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int *sm, *sm_d;

    sm = (int *) malloc(N*sizeof(*sm));

    cudaMalloc((void**)&sm_d,N*sizeof(*sm_d));

    kern<<<N,1>>>( sm_d);

    cudaMemcpy(sm, sm_d, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("======= SM count: %d, launch %d block, each block has 1 threads ========", prop.multiProcessorCount, N);
    for (int i=0;i<N;i++){
        printf("block %d executed on SM %d\n",i,sm[i]);
    }

    return 0;

}
