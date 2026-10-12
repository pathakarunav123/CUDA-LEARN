// A common strategy is to partition the data into subsets called tiles
//such that each tile fits into the shared memory. The term tile draws on the
//analogy that a large wall (i.e., the global memory data) can often be
//covered by tiles (i.e., subsets that each can fit into the shared memory).
//An important criterion is that the kernel computations on these tiles can
//be done independently of each other. Note that not all data structures
//can be partitioned into tiles given an arbitrary kernel function.

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

#define TILE 16

__global__ void tiledMatMul(
	const float* A,
	const float* B,
	float* C,
	int M, int K, int N)
{
	const int row = blockIdx.y * TILE + threadIdx.y;
	const int col = blockIdx.x * TILE + threadIdx.x;

	__shared__ float As[TILE][TILE];
	__shared__ float Bs[TILE][TILE];

	float sum = 0.0f;

	const int numTiles = (K + TILE - 1) / TILE;

	for (int t = 0; t < numTiles; ++t)
	{
		const int aCol = t * TILE + threadIdx.x;
		const int bRow = t * TILE + threadIdx.y;

		As[threadIdx.y][threadIdx.x] =
			(row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
		Bs[threadIdx.y][threadIdx.x] =
			(bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

		__syncthreads();

		for (int k = 0; k < TILE; ++k)
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

		__syncthreads();
	}

	if (row < M && col < N)
		C[row * N + col] = sum;
}

int main()
{
	constexpr int M = 35;
	constexpr int K = 29;
	constexpr int N = 23;

	std::vector<float> h_A(M * K);
	std::vector<float> h_B(K * N);
	std::vector<float> h_C(M * N, 0.0f);

	for (int i = 0; i < M * K; ++i)
		h_A[i] = static_cast<float>(i % 5 + 1);
	for (int i = 0; i < K * N; ++i)
		h_B[i] = static_cast<float>(i % 7 + 1);

	const size_t bytesA = h_A.size() * sizeof(float);
	const size_t bytesB = h_B.size() * sizeof(float);
	const size_t bytesC = h_C.size() * sizeof(float);
	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytesA));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytesB));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytesC));
	CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

	const dim3 threads(TILE, TILE);
	const dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
	tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
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
				expected += h_A[row * K + k] * h_B[k * N + col];

			if (std::fabs(h_C[row * N + col] - expected) > 1e-5f)
			{
				fprintf(stderr, "Verification failed at C[%d][%d]: %.1f != %.1f\n",
						row, col, h_C[row * N + col], expected);
				valid = false;
			}
		}
	}

	for (int row = 0; row < 4 && row < M; ++row)
	{
		for (int col = 0; col < 4 && col < N; ++col)
			printf("%.1f%c", h_C[row * N + col],
				   col == 3 || col == N - 1 ? '\n' : ' ');
	}

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	if (!valid)
		return EXIT_FAILURE;

	printf("Tiled matrix multiplication successful.\n");
	return EXIT_SUCCESS;
}
