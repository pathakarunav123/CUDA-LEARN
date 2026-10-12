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

__global__ void matMul(
	const float* A,
	const float* B,
	float* C,
	int M, int K, int N)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N)
	{
		float sum = 0.0f;
		for (int k = 0; k < K; ++k)
			sum += A[row * K + k] * B[k * N + col];
		C[row * N + col] = sum;
	}
}

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

bool verifyAgainstCpuReference(
	const char* kernelName,
	const std::vector<float>& output,
	const std::vector<double>& reference)
{
	constexpr double absoluteTolerance = 1e-3;
	constexpr double relativeTolerance = 1e-5;

	for (size_t i = 0; i < reference.size(); ++i)
	{
		const double actual = output[i];
		const double expected = reference[i];
		const double tolerance =
			absoluteTolerance + relativeTolerance * std::fabs(expected);

		if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
		{
			fprintf(stderr,
					"%s verification failed at C[%zu]: GPU %.6f, CPU %.6f "
					"(tolerance %.6f)\n",
					kernelName, i, actual, expected, tolerance);
			return false;
		}
	}

	printf("%s output matches the CPU reference.\n", kernelName);
	return true;
}

float benchmarkKernel(
	bool useTiled,
	const float* d_A,
	const float* d_B,
	float* d_C,
	int M, int K, int N,
	int repetitions)
{
	if (repetitions <= 0)
	{
		fprintf(stderr, "Benchmark repetitions must be greater than zero.\n");
		exit(EXIT_FAILURE);
	}

	const dim3 threads(TILE, TILE);
	const dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

	for (int i = 0; i < 20; ++i)
	{
		if (useTiled)
			tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
		else
			matMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	CUDA_CHECK(cudaEventRecord(start));

	for (int i = 0; i < repetitions; ++i)
	{
		if (useTiled)
			tiledMatMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
		else
			matMul<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
	}

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float totalMs = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	return totalMs / repetitions;
}

int main()
{
	constexpr int M = 512;
	constexpr int K = 512;
	constexpr int N = 512;
	constexpr int repetitions = 100;

	std::vector<float> h_A(M * K);
	std::vector<float> h_B(K * N);
	std::vector<float> h_naive(M * N);
	std::vector<float> h_tiled(M * N);

	for (int i = 0; i < M * K; ++i)
		h_A[i] = static_cast<float>(i % 5 + 1);
	for (int i = 0; i < K * N; ++i)
		h_B[i] = static_cast<float>(i % 7 + 1);

	std::vector<double> h_reference(M * N, 0.0);
	for (int row = 0; row < M; ++row)
	{
		for (int col = 0; col < N; ++col)
		{
			double sum = 0.0;
			for (int k = 0; k < K; ++k)
				sum += static_cast<double>(h_A[row * K + k]) *
					   static_cast<double>(h_B[k * N + col]);
			h_reference[row * N + col] = sum;
		}
	}

	const size_t bytesA = h_A.size() * sizeof(float);
	const size_t bytesB = h_B.size() * sizeof(float);
	const size_t bytesC = h_naive.size() * sizeof(float);
	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytesA));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytesB));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytesC));
	CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice));

	const float naiveMs =
		benchmarkKernel(false, d_A, d_B, d_C, M, K, N, repetitions);
	CUDA_CHECK(cudaMemcpy(h_naive.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

	const float tiledMs =
		benchmarkKernel(true, d_A, d_B, d_C, M, K, N, repetitions);
	CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_C, bytesC, cudaMemcpyDeviceToHost));

	const bool naiveValid =
		verifyAgainstCpuReference("Naive MatMul", h_naive, h_reference);
	const bool tiledValid =
		verifyAgainstCpuReference("Tiled MatMul", h_tiled, h_reference);

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	if (!naiveValid || !tiledValid)
		return EXIT_FAILURE;

	printf("Naive MatMul: %.4f ms/kernel\n", naiveMs);
	printf("Tiled MatMul: %.4f ms/kernel\n", tiledMs);

	if (tiledMs <= 0.0f)
	{
		fprintf(stderr, "Tiled kernel time is too small to calculate speedup.\n");
		return EXIT_FAILURE;
	}

	const float speedup = naiveMs / tiledMs;
	printf("Speedup: %.2fx\n", speedup);
	if (speedup > 1.0f)
		printf("Tiled MatMul is faster.\n");
	else if (speedup < 1.0f)
		printf("Naive MatMul is faster.\n");
	else
		printf("No meaningful timing difference for this workload.\n");

	return EXIT_SUCCESS;
}
