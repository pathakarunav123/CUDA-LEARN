// Naive vs shared-memory tiled matrix multiply.
//
// What this program does, in order:
//   1. Verifies BOTH kernels against a double-precision CPU reference
//      (odd shapes too, so the bounds-check paths actually run).
//   2. Warms up both kernels.
//   3. Runs several benchmark rounds on the SAME data and dimensions, and
//      ALTERNATES which kernel goes first in each round (order-bias check).
//   4. Prints every round, then median/min/max and the measured speedup.
//
// Build:  nvcc -O3 -std=c++17 -arch=native matmul_benchmark_v2.cu -o matmul_benchmark_v2
//         (-arch=native needs CUDA 11.6+; otherwise e.g. -arch=sm_86)
// Run:    ./matmul_benchmark_v2
// Check:  compute-sanitizer ./matmul_benchmark_v2   (optional, catches races/OOB)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
} while (0)

#define TILE 16

// ------------------------------------------------------------------ kernels
// Unchanged from the original code, so results stay comparable.

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

// ------------------------------------------------------------------ helpers

// Frees the GPU memory automatically, even on an early return.
class DeviceBuffer
{
public:
	explicit DeviceBuffer(size_t count) : bytes_(count * sizeof(float))
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes_));
	}
	~DeviceBuffer() { cudaFree(ptr_); }
	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	float* get() const { return ptr_; }
	size_t bytes() const { return bytes_; }

private:
	float* ptr_ = nullptr;
	size_t bytes_;
};

void launchKernel(
	bool useTiled,
	const float* A, const float* B, float* C,
	int M, int K, int N)
{
	const dim3 threads(TILE, TILE);
	const dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
	if (useTiled)
		tiledMatMul<<<blocks, threads>>>(A, B, C, M, K, N);
	else
		matMul<<<blocks, threads>>>(A, B, C, M, K, N);
}

// Seeded random values in [-1, 1]. Same seed -> same data every run, and
// random data (unlike i % 5 + 1) cannot hide a wrong index by repeating.
std::vector<float> makeRandomMatrix(size_t count, unsigned seed)
{
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> m(count);
	for (float& x : m)
		x = dist(gen);
	return m;
}

// Double-precision CPU reference (i-k-j order so B is read row by row).
std::vector<double> cpuReference(
	const std::vector<float>& A,
	const std::vector<float>& B,
	int M, int K, int N)
{
	std::vector<double> C(size_t(M) * N, 0.0);
	for (int i = 0; i < M; ++i)
		for (int k = 0; k < K; ++k)
		{
			const double a = A[size_t(i) * K + k];
			for (int j = 0; j < N; ++j)
				C[size_t(i) * N + j] += a * static_cast<double>(B[size_t(k) * N + j]);
		}
	return C;
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

		// !isfinite also catches the NaN we pre-fill into C (kernel wrote nothing).
		if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance)
		{
			fprintf(stderr,
					"  %s FAILED at C[%zu]: GPU %.6f, CPU %.6f (tolerance %.6f)\n",
					kernelName, i, actual, expected, tolerance);
			return false;
		}
	}

	printf("  %s matches the CPU reference.\n", kernelName);
	return true;
}

// Runs each kernel once on a fresh NaN-filled C and checks it against the CPU.
bool verifyShape(int M, int K, int N)
{
	printf("Verify %d x %d x %d\n", M, K, N);

	const std::vector<float> hA = makeRandomMatrix(size_t(M) * K, 1);
	const std::vector<float> hB = makeRandomMatrix(size_t(K) * N, 2);
	const std::vector<double> reference = cpuReference(hA, hB, M, K, N);
	std::vector<float> output(size_t(M) * N);

	DeviceBuffer dA(hA.size()), dB(hB.size()), dC(output.size());
	CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

	bool ok = true;
	for (int tiled = 0; tiled < 2; ++tiled)
	{
		// 0xFF bytes = NaN. A kernel that skips an element cannot pass by
		// inheriting the other kernel's old result.
		CUDA_CHECK(cudaMemset(dC.get(), 0xFF, dC.bytes()));
		launchKernel(tiled != 0, dA.get(), dB.get(), dC.get(), M, K, N);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(output.data(), dC.get(), dC.bytes(), cudaMemcpyDeviceToHost));

		ok = verifyAgainstCpuReference(tiled ? "Tiled MatMul" : "Naive MatMul",
									   output, reference) && ok;
	}
	return ok;
}

// Average time (ms) of one kernel launch over `launches` back-to-back launches.
float timeBatch(
	bool useTiled,
	const float* dA, const float* dB, float* dC,
	int M, int K, int N,
	int launches)
{
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < launches; ++i)
		launchKernel(useTiled, dA, dB, dC, M, K, N);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float totalMs = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return totalMs / launches;
}

double median(std::vector<double> v)
{
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double minOf(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
double maxOf(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

// ------------------------------------------------------------------ main

int main()
{
	constexpr int M = 512;
	constexpr int K = 512;
	constexpr int N = 512;

	constexpr int kWarmupLaunches = 20;
	constexpr int kLaunchesPerRound = 50;
	constexpr int kRounds = 10;  // keep even: half the rounds per order

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

	// ---- 1. Correctness first. No timing is worth reporting if a kernel is wrong.
	printf("== Step 1: verification against CPU reference ==\n");
	bool ok = verifyShape(33, 47, 65);      // not multiples of TILE -> bounds checks run
	ok = verifyShape(257, 300, 129) && ok;  // odd rectangular shape
	ok = verifyShape(M, K, N) && ok;        // the benchmark shape itself
	if (!ok)
	{
		fprintf(stderr, "\nVerification failed. Timing skipped.\n");
		return EXIT_FAILURE;
	}

	// ---- 2. Same dimensions and same input data for both kernels.
	const std::vector<float> hA = makeRandomMatrix(size_t(M) * K, 1);
	const std::vector<float> hB = makeRandomMatrix(size_t(K) * N, 2);
	DeviceBuffer dA(hA.size()), dB(hB.size()), dC(size_t(M) * N);
	CUDA_CHECK(cudaMemcpy(dA.get(), hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dB.get(), hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

	// Warm up BOTH kernels (context, module load, clocks, caches).
	for (int i = 0; i < kWarmupLaunches; ++i)
	{
		launchKernel(false, dA.get(), dB.get(), dC.get(), M, K, N);
		launchKernel(true, dA.get(), dB.get(), dC.get(), M, K, N);
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// ---- 3. Benchmark rounds with alternating order.
	printf("\n== Step 2: benchmark %dx%dx%d, %d rounds x %d launches ==\n",
		   M, K, N, kRounds, kLaunchesPerRound);
	printf("%-6s %-14s %12s %12s %10s\n",
		   "round", "order", "naive ms", "tiled ms", "speedup");

	std::vector<double> naiveMs, tiledMs, speedups;
	std::vector<double> speedupNaiveFirst, speedupTiledFirst;

	for (int r = 0; r < kRounds; ++r)
	{
		const bool naiveFirst = (r % 2 == 0);
		float n = 0.0f, t = 0.0f;

		if (naiveFirst)
		{
			n = timeBatch(false, dA.get(), dB.get(), dC.get(), M, K, N, kLaunchesPerRound);
			t = timeBatch(true, dA.get(), dB.get(), dC.get(), M, K, N, kLaunchesPerRound);
		}
		else
		{
			t = timeBatch(true, dA.get(), dB.get(), dC.get(), M, K, N, kLaunchesPerRound);
			n = timeBatch(false, dA.get(), dB.get(), dC.get(), M, K, N, kLaunchesPerRound);
		}

		const double s = static_cast<double>(n) / t;
		naiveMs.push_back(n);
		tiledMs.push_back(t);
		speedups.push_back(s);
		(naiveFirst ? speedupNaiveFirst : speedupTiledFirst).push_back(s);

		printf("%-6d %-14s %12.4f %12.4f %9.2fx\n",
			   r + 1, naiveFirst ? "naive->tiled" : "tiled->naive", n, t, s);
	}

	// ---- 4. Report what was actually measured.
	const double flops = 2.0 * M * double(N) * K;
	const double medNaive = median(naiveMs);
	const double medTiled = median(tiledMs);

	printf("\n== Summary (measured on this run) ==\n");
	printf("Naive : median %.4f ms  (min %.4f, max %.4f)  %.1f GFLOP/s\n",
		   medNaive, minOf(naiveMs), maxOf(naiveMs), flops / (medNaive * 1e6));
	printf("Tiled : median %.4f ms  (min %.4f, max %.4f)  %.1f GFLOP/s\n",
		   medTiled, minOf(tiledMs), maxOf(tiledMs), flops / (medTiled * 1e6));
	printf("Speedup: median %.2fx  (min %.2fx, max %.2fx)\n",
		   median(speedups), minOf(speedups), maxOf(speedups));

	const double a = median(speedupNaiveFirst);
	const double b = median(speedupTiledFirst);
	const double orderGapPct = 100.0 * std::fabs(a - b) / (0.5 * (a + b));
	printf("Order check: naive-first median %.2fx, tiled-first median %.2fx (gap %.1f%%)\n",
		   a, b, orderGapPct);
	if (orderGapPct > 5.0)
		printf("  Gap above 5%%: run order affects the result, treat the speedup as approximate.\n");
	else
		printf("  Gap within 5%%: no clear order bias.\n");

	if (minOf(speedups) > 1.0)
		printf("Verdict: tiled was faster in all %d rounds.\n", kRounds);
	else if (maxOf(speedups) < 1.0)
		printf("Verdict: naive was faster in all %d rounds.\n", kRounds);
	else
		printf("Verdict: mixed result across rounds, not conclusive.\n");

	return EXIT_SUCCESS;
}
