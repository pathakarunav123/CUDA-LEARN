// Tiny shared-memory example.
#include <cuda_runtime.h>
#include <iostream>

__global__ void sharedMemoryDemo(int *input, int *output)
{
	__shared__ int shared[4];

	int tid = threadIdx.x;

	// Global -> shared memory.
	shared[tid] = input[tid];

	// Ensure every thread has finished loading.
	__syncthreads();

	// Reuse the value loaded by thread 0.
	output[tid] = shared[0];
}

int main()
{
	constexpr int count = 4;
	const int inputHost[count] = {10, 20, 30, 40};
	int outputHost[count] = {};
	int *inputDevice = nullptr;
	int *outputDevice = nullptr;

	cudaMalloc(&inputDevice, count * sizeof(int));
	cudaMalloc(&outputDevice, count * sizeof(int));
	cudaMemcpy(inputDevice, inputHost, count * sizeof(int), cudaMemcpyHostToDevice);

	sharedMemoryDemo<<<1, count>>>(inputDevice, outputDevice);
	cudaDeviceSynchronize();

	cudaMemcpy(outputHost, outputDevice, count * sizeof(int), cudaMemcpyDeviceToHost);

	for (int value : outputHost)
		std::cout << value << ' ';
	std::cout << '\n';

	cudaFree(inputDevice);
	cudaFree(outputDevice);
	return 0;
}
