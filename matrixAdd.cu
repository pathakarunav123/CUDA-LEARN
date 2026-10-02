#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void matrixAdd(float *A, float *B, float *C, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    int rows = 1024;
    int cols = 1024;
    size_t size = rows * cols * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for(int i = 0; i < rows * cols; i++){
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );

    matrixAdd<<<blocks, threads>>>(d_A, d_B, d_C, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    for(int i = 0; i < rows * cols; i++){
        if(h_C[i] != h_A[i] + h_B[i]){
            fprintf(stderr, "Result verification failed at index %d: %f != %f\n", i, h_C[i], h_A[i] + h_B[i]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Matrix addition successful!\n");
    printf("First 10 matrix additions:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.0f + %.0f = %.0f\n", h_A[i], h_B[i], h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;


}
