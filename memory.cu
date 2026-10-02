#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>


#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void Coalescedcopy(float *input, float *output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //coalsced acess
    if (i < N) 
    {
        output[i] = input[i];
    }

     
}

int main(){

    int N = 1<<24;
    size_t size = N * sizeof(float);

    float *d_input, *d_output;
    //Calculate in GB
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));


    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));


    Coalescedcopy<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
     CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliSeconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliSeconds, start, stop));

    double bytesTransferred = 2.0 * N*sizeof(float); // 2 for input and output

    double seconds = milliSeconds / 1000.0;
    double bandwidth = bytesTransferred / (seconds * 1e9); // in GB/s
    

   //printf("Time taken for Coalescedcopy kernel: %f ms\n", milliSeconds);
   printf("time take for stride-2 kernal: %f ms\n", milliSeconds);
   printf("Bandwidth: %f GB/s\n", bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
   CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}