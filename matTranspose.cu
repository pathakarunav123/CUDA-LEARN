#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return EXIT_FAILURE; \
    } \
} while (0)

__global__ void transposeMatrix(const float *source, float *transposed, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        transposed[col * rows + row] = source[row * cols + col];
    }
}

int main()
{
    const int rows = 3;
    const int cols = 4;
    const size_t elementCount = static_cast<size_t>(rows) * cols;
    const size_t size = elementCount * sizeof(float);

    float *hostSource = static_cast<float *>(malloc(size));
    float *hostTransposed = static_cast<float *>(malloc(size));
    if (hostSource == nullptr || hostTransposed == nullptr) {
        fprintf(stderr, "Host allocation failed\n");
        free(hostSource);
        free(hostTransposed);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < elementCount; ++i) {
        hostSource[i] = static_cast<float>(i + 1);
    }

    float *deviceSource = nullptr;
    float *deviceTransposed = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceSource, size));
    CUDA_CHECK(cudaMalloc(&deviceTransposed, size));
    CUDA_CHECK(cudaMemcpy(deviceSource, hostSource, size, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );
    transposeMatrix<<<blocks, threads>>>(deviceSource, deviceTransposed, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hostTransposed, deviceTransposed, size, cudaMemcpyDeviceToHost));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float expected = hostSource[row * cols + col];
            float actual = hostTransposed[col * rows + row];
            if (actual != expected) {
                fprintf(stderr, "Transpose failed at (%d, %d): %.0f != %.0f\n",
                        row, col, actual, expected);
                cudaFree(deviceSource);
                cudaFree(deviceTransposed);
                free(hostSource);
                free(hostTransposed);
                return EXIT_FAILURE;
            }
        }
    }

    printf("Matrix transpose successful!\n");
    printf("Original matrix:\n");
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            printf("%.0f ", hostSource[row * cols + col]);
        }
        printf("\n");
    }

    printf("Transposed matrix:\n");
    for (int row = 0; row < cols; ++row) {
        for (int col = 0; col < rows; ++col) {
            printf("%.0f ", hostTransposed[row * rows + col]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaFree(deviceSource));
    CUDA_CHECK(cudaFree(deviceTransposed));
    free(hostSource);
    free(hostTransposed);
    return EXIT_SUCCESS;
}
