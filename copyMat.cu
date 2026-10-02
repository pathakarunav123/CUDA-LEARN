#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		return EXIT_FAILURE; \
	} \
} while (0)

__global__ void copyMatrix(const float *source, float *destination, int rows, int cols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < rows && col < cols) {
		int index = row * cols + col;
		destination[index] = source[index];
	}
}

int main()
{
	const int rows = 1024;
	const int cols = 1024;
	const size_t elementCount = static_cast<size_t>(rows) * cols;
	const size_t size = elementCount * sizeof(float);

	float *hostSource = static_cast<float *>(malloc(size));
	float *hostResult = static_cast<float *>(malloc(size));
	if (hostSource == nullptr || hostResult == nullptr) {
		fprintf(stderr, "Host allocation failed\n");
		free(hostSource);
		free(hostResult);
		return EXIT_FAILURE;
	}

	for (size_t i = 0; i < elementCount; ++i) {
		hostSource[i] = static_cast<float>(i);
	}

	float *deviceSource = nullptr;
	float *deviceResult = nullptr;
	CUDA_CHECK(cudaMalloc(&deviceSource, size));
	CUDA_CHECK(cudaMalloc(&deviceResult, size));
	CUDA_CHECK(cudaMemcpy(deviceSource, hostSource, size, cudaMemcpyHostToDevice));

	dim3 threads(16, 16);
	dim3 blocks(
		(cols + threads.x - 1) / threads.x,
		(rows + threads.y - 1) / threads.y
	);
	copyMatrix<<<blocks, threads>>>(deviceSource, deviceResult, rows, cols);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemcpy(hostResult, deviceResult, size, cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < elementCount; ++i) {
		if (hostResult[i] != hostSource[i]) {
			fprintf(stderr, "Matrix copy failed at index %zu: %.0f != %.0f\n",
					i, hostResult[i], hostSource[i]);
			cudaFree(deviceSource);
			cudaFree(deviceResult);
			free(hostSource);
			free(hostResult);
			return EXIT_FAILURE;
		}
	}

	printf("Matrix copy successful!\n");
	printf("First 10 copied values:\n");
	for (int i = 0; i < 10; ++i) {
		printf("%.0f -> %.0f\n", hostSource[i], hostResult[i]);
	}

	CUDA_CHECK(cudaFree(deviceSource));
	CUDA_CHECK(cudaFree(deviceResult));
	free(hostSource);
	free(hostResult);
	return EXIT_SUCCESS;
}
