#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define OFFSET(row, col, ld) ((row) * (ld) + (col))

using GEMMFn = void (*)(float*, float*, float*, const int, const int, const int);


void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

float testMaxError(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}

__global__ void naive_contiguous_gemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        // #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            // if(k < 5)
                // printf("B[x=%d, y=%d] T[x=%d, y=%d] [m=%d,n=%d,k=%d,M=%d,N=%d,K=%d] - a[%d], b[%d], c[%d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,m,n,k,M,N,K,OFFSET(m,k,K), OFFSET(k,n,N), OFFSET(m,n,N));
        }
        c[OFFSET(m, n, N)] = psum;
    }
}


__global__ void naive_incontiguous_gemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if(m < M && n < N){
        float sum = 0;
        for(int k = 0; k < K; ++k){
            sum += a[OFFSET(m,k,K)] * b[OFFSET(k,n,N)];
            // if(k < 5)
                // printf("B[x=%d, y=%d] T[x=%d, y=%d] -  a[%d], b[%d], c[%d]\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,OFFSET(m,k,K), OFFSET(k,n,N), OFFSET(m,n,N));
        }
        c[OFFSET(m,n,N)] = sum;
    }

}               


float testPerformance(
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}


int main() {

    cudaError_t error = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024 ); // 10MB
    if (error != cudaSuccess) {
        printf("Failed to set printf FIFO size: %s\n", cudaGetErrorString(error));
    }

    const int MN_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    // const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    const int outer_repeat = 5, inner_repeat = 1;





    {
        GEMMFn gemms[2] = {naive_contiguous_gemm, naive_incontiguous_gemm};
        std::string gemms_name[2] = {"naive_contiguous_gemm", "naive_incontiguous_gemm"};

        const int BM = 32, BN = 32; // #threads per block，CUDA主流架构都是1024


        for (int gemm_i = 0; gemm_i < 2; gemm_i++)
        {
            GEMMFn gemm = gemms[gemm_i];

            printf("\n\n===================== gemm function: %s =====================\n", gemms_name[gemm_i].c_str());

            const int TESTNUM = 15;
            for (int i = 0; i < TESTNUM; i++) {
                /***
                1.验证精度, >1024的维度验证精度太慢了，skip
                */
                const int M = MN_list[i], N = MN_list[i], K = K_list[i];
                dim3 blockDim(BN, BM);
                dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
                if(M < 1024){
                    float max_error = testMaxError(gemm, gridDim, blockDim, M, N, K);
                    printf("M N K = %6d %6d %6d, Max Error = %.7f, ", M, N, K, max_error);
                }else{
                    printf("M N K = %6d %6d %6d, Max Error = Skip(太慢) ", M, N, K);
                }



                /***
                2. 计算性能
                */

                double max_sec = 0.0;
                double min_sec = DBL_MAX;
                double total_sec = 0.0;

                for (int j = 0; j < outer_repeat; j++) {
                    double this_sec = testPerformance(gemm, gridDim, blockDim, M, N, K, inner_repeat);
                    max_sec = max(max_sec, this_sec);
                    min_sec = min(min_sec, this_sec);
                    total_sec += this_sec;
                }

                double avg_sec = total_sec / outer_repeat;
                double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

                printf("Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", min_sec, avg_sec, max_sec, avg_Gflops);
            }
        }
    }

}