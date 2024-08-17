#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

__global__ void tick_1s()
{
    for (int i = 0; i < 1e3; ++i)
    {
        __nanosleep(1e6);
    }
}

int main(int argc, char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int num_blocks = atoi(argv[1]) * prop.multiProcessorCount;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tick_1s<<<num_blocks, 1>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);

    printf("SM count: %d, use 1 SM to laucnch %d blocks, so ", prop.multiProcessorCount, num_blocks);
    printf("blocks/SM: %d", num_blocks / prop.multiProcessorCount);
    printf(", elapsed time: %.2f\n", duration / 1e3);
}
