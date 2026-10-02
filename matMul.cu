#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

__global__ void matMul(const float *A, const float *B, float *C,
					   int M, int K, int N)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N)
	{
		float sum = 0.0f;

		for (int k = 0; k < K; k++)
		{
			sum += A[row * K + k] * B[k * N + col];
		}

		C[row * N + col] = sum;
	}
}

int main()
{
	const int M = 2;
	const int K = 3;
	const int N = 2;

	const std::vector<float> h_A = {1.0f, 2.0f, 3.0f,
									4.0f, 5.0f, 6.0f};
	const std::vector<float> h_B = {7.0f, 8.0f,
									9.0f, 10.0f,
									11.0f, 12.0f};
	std::vector<float> h_C(M * N);
	const size_t bytesA = M * K * sizeof(float);
	const size_t bytesB = K * N * sizeof(float);
	const size_t bytesC = M * N * sizeof(float);

	float *d_A;
	float *d_B;
	float *d_C;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), bytesA));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), bytesB));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), bytesC));

	CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

	dim3 threads(16, 16);

	dim3 blocks(
		(N + threads.x - 1) / threads.x,
		(M + threads.y - 1) / threads.y
	);

	matMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

	bool valid = true;
	for (int row = 0; row < M; ++row)
	{
		for (int col = 0; col < N; ++col)
		{
			float expected = 0.0f;
			for (int k = 0; k < K; ++k)
			{
				expected += h_A[row * K + k] * h_B[k * N + col];
			}

			int index = row * N + col;
			printf("C[%d][%d] = %.1f\n", row, col, h_C[index]);
			if (std::fabs(h_C[index] - expected) > 1e-5f)
			{
				fprintf(stderr, "Verification failed at C[%d][%d]: %.1f != %.1f\n",
						row, col, h_C[index], expected);
				valid = false;
			}
		}
	}

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	if (!valid)
	{
		return EXIT_FAILURE;
	}

	printf("Naive matrix multiplication successful!\n");
	return EXIT_SUCCESS;
}
