// Transpose experiment: naive vs tiled 32x32 vs padded 32x33, all with 32x32 blocks.
//
//   Naive         baseline: coalesced reads, strided writes
//   Tiled 32x32   coalescing + shared-memory overhead (tile[32][32], bank conflicts on read)
//   Padded 32x33  same tiled approach, bank layout changed (tile[32][33])
//
// What this program does, in order:
//   1. Verifies ALL three kernels against an exact CPU transpose (odd shapes included).
//   2. Warms up all three kernels.
//   3. Runs benchmark rounds on the SAME matrix, rotating the run order through all
//      6 permutations so no variant is always first or last.
//   4. Prints every round, then median/min/max, bandwidth, speedups and an order check.
//
// Build:  nvcc -O3 -std=c++17 -arch=native transpose_benchmark.cu -o transpose_benchmark
//         (-arch=native needs CUDA 11.6+; otherwise e.g. -arch=sm_89)
// Run:    ./transpose_benchmark
// Check:  compute-sanitizer ./transpose_benchmark   (optional, catches races/OOB)

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

#define TILE 32  // 32 x 32 = 1024 threads per block, the per-block maximum

// ------------------------------------------------------------------ kernels

// Variant A: naive.
__global__ void naiveTranspose(
	const float* input, float* output,
	int height, int width)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < height && col < width)
		output[col * height + row] = input[row * width + col];
}

// Variant B: tiled, without padding.
__global__ void tiledTranspose(
	const float* input, float* output,
	int height, int width)
{
	__shared__ float tile[TILE][TILE];

	int col = blockIdx.x * TILE + threadIdx.x;
	int row = blockIdx.y * TILE + threadIdx.y;

	if (row < height && col < width)
		tile[threadIdx.y][threadIdx.x] =
			input[row * width + col];

	__syncthreads();

	int outCol = blockIdx.y * TILE + threadIdx.x;
	int outRow = blockIdx.x * TILE + threadIdx.y;

	if (outRow < width && outCol < height)
		output[outRow * height + outCol] =
			tile[threadIdx.x][threadIdx.y];
}

// Variant C: tiled, with padding. Logic identical to B; only the tile layout changes.
__global__ void paddedTranspose(
	const float* input, float* output,
	int height, int width)
{
	__shared__ float tile[TILE][TILE + 1];

	int col = blockIdx.x * TILE + threadIdx.x;
	int row = blockIdx.y * TILE + threadIdx.y;

	if (row < height && col < width)
		tile[threadIdx.y][threadIdx.x] =
			input[row * width + col];

	__syncthreads();

	int outCol = blockIdx.y * TILE + threadIdx.x;
	int outRow = blockIdx.x * TILE + threadIdx.y;

	if (outRow < width && outCol < height)
		output[outRow * height + outCol] =
			tile[threadIdx.x][threadIdx.y];
}

// ------------------------------------------------------------------ variants

using KernelFn = void (*)(const float*, float*, int, int);

struct Variant
{
	const char* name;
	const char* measures;
	char letter;  // short tag for the per-round order column
	KernelFn kernel;
};

const Variant kVariants[] = {
	{"Naive",        "baseline",                                'N', naiveTranspose},
	{"Tiled 32x32",  "coalescing + shared-memory overhead",     'T', tiledTranspose},
	{"Padded 32x33", "same tiled approach, bank layout changed", 'P', paddedTranspose},
};
constexpr int kNumVariants = 3;

// ------------------------------------------------------------------ helpers

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

// One launch configuration for all three kernels.
void launchVariant(KernelFn kernel, const float* in, float* out, int height, int width)
{
	const dim3 threads(TILE, TILE);
	const dim3 blocks(
		(width + TILE - 1) / TILE,
		(height + TILE - 1) / TILE);
	kernel<<<blocks, threads>>>(in, out, height, width);
}

std::vector<float> makeRandomMatrix(size_t count, unsigned seed)
{
	std::mt19937 gen(seed);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::vector<float> m(count);
	for (float& x : m)
		x = dist(gen);
	return m;
}

// A transpose only moves values, so the CPU reference is exact: out[c][r] = in[r][c].
std::vector<float> cpuTranspose(const std::vector<float>& in, int height, int width)
{
	std::vector<float> out(in.size());
	for (int r = 0; r < height; ++r)
		for (int c = 0; c < width; ++c)
			out[size_t(c) * height + r] = in[size_t(r) * width + c];
	return out;
}

// Runs every variant once on a NaN-filled output and compares exactly.
// NaN never equals anything, so an element a kernel failed to write also fails.
bool verifyShape(int height, int width)
{
	printf("Verify %d x %d\n", height, width);

	const std::vector<float> hIn = makeRandomMatrix(size_t(height) * width, 1);
	const std::vector<float> expected = cpuTranspose(hIn, height, width);
	std::vector<float> got(hIn.size());

	DeviceBuffer dIn(hIn.size()), dOut(hIn.size());
	CUDA_CHECK(cudaMemcpy(dIn.get(), hIn.data(), dIn.bytes(), cudaMemcpyHostToDevice));

	bool allOk = true;
	for (const Variant& v : kVariants)
	{
		CUDA_CHECK(cudaMemset(dOut.get(), 0xFF, dOut.bytes()));  // 0xFF bytes = NaN
		launchVariant(v.kernel, dIn.get(), dOut.get(), height, width);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(got.data(), dOut.get(), dOut.bytes(), cudaMemcpyDeviceToHost));

		bool ok = true;
		for (size_t i = 0; i < expected.size(); ++i)
		{
			if (!(got[i] == expected[i]))
			{
				fprintf(stderr, "  %-13s FAILED at out[%zu]: GPU %f, CPU %f\n",
						v.name, i, got[i], expected[i]);
				ok = false;
				break;
			}
		}
		if (ok)
			printf("  %-13s matches the CPU reference (exact)\n", v.name);
		allOk = allOk && ok;
	}
	return allOk;
}

// Average time (ms) of one launch over `launches` back-to-back launches.
float timeBatch(KernelFn kernel, const float* in, float* out, int height, int width, int launches)
{
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));
	for (int i = 0; i < launches; ++i)
		launchVariant(kernel, in, out, height, width);
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
	if (v.empty())
		return 0.0;
	std::sort(v.begin(), v.end());
	const size_t n = v.size();
	return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double minOf(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
double maxOf(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

// Per-round ratio a[r] / b[r]: how many times faster b is than a in each round.
std::vector<double> ratios(const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> r(a.size());
	for (size_t i = 0; i < a.size(); ++i)
		r[i] = a[i] / b[i];
	return r;
}

void printRatio(const char* label, const std::vector<double>& r)
{
	printf("%-28s median %.2fx  (min %.2fx, max %.2fx)  ", label, median(r), minOf(r), maxOf(r));
	if (minOf(r) > 1.0)
		printf("-> faster in all %zu rounds\n", r.size());
	else if (maxOf(r) < 1.0)
		printf("-> slower in all %zu rounds\n", r.size());
	else
		printf("-> mixed across rounds, not conclusive\n");
}

// ------------------------------------------------------------------ main

int main()
{
	constexpr int kHeight = 4096;
	constexpr int kWidth = 4096;
	constexpr int kWarmupLaunches = 10;
	constexpr int kLaunchesPerRound = 20;
	constexpr int kRounds = 12;  // multiple of 6: every run order is used equally often

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

	// ---- 1. Correctness first.
	printf("== Step 1: verification against CPU reference ==\n");
	bool ok = verifyShape(33, 47);       // not multiples of 32: edge blocks, idle threads
	ok = verifyShape(100, 257) && ok;    // wide
	ok = verifyShape(257, 100) && ok;    // tall
	ok = verifyShape(5, 70) && ok;       // one dimension smaller than a tile
	ok = verifyShape(1024, 1024) && ok;  // clean multiple of the tile
	if (!ok)
	{
		fprintf(stderr, "\nVerification failed. Timing skipped.\n");
		return EXIT_FAILURE;
	}

	// ---- 2. Same matrix for every variant, then warm all of them up.
	const size_t count = size_t(kHeight) * kWidth;
	const std::vector<float> hIn = makeRandomMatrix(count, 1);
	DeviceBuffer dIn(count), dOut(count);
	CUDA_CHECK(cudaMemcpy(dIn.get(), hIn.data(), dIn.bytes(), cudaMemcpyHostToDevice));

	for (const Variant& v : kVariants)
		for (int i = 0; i < kWarmupLaunches; ++i)
			launchVariant(v.kernel, dIn.get(), dOut.get(), kHeight, kWidth);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// ---- 3. Benchmark rounds, rotating through all 6 run orders.
	static const int perms[6][kNumVariants] = {
		{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

	printf("\n== Step 2: transpose %d x %d, %dx%d blocks, %d rounds x %d launches ==\n",
		   kHeight, kWidth, TILE, TILE, kRounds, kLaunchesPerRound);
	printf("%-6s %-8s %13s %13s %13s   (ms per launch)\n",
		   "round", "order", kVariants[0].name, kVariants[1].name, kVariants[2].name);

	std::vector<double> ms[kNumVariants];       // ms[variant][round]
	std::vector<int> position[kNumVariants];    // position[variant][round]: 0 = ran first
	for (int v = 0; v < kNumVariants; ++v)
	{
		ms[v].assign(kRounds, 0.0);
		position[v].assign(kRounds, 0);
	}

	for (int r = 0; r < kRounds; ++r)
	{
		const int* order = perms[r % 6];
		for (int p = 0; p < kNumVariants; ++p)
		{
			const int v = order[p];
			ms[v][r] = timeBatch(kVariants[v].kernel, dIn.get(), dOut.get(),
								 kHeight, kWidth, kLaunchesPerRound);
			position[v][r] = p;
		}
		printf("%-6d %c>%c>%c    %13.4f %13.4f %13.4f\n", r + 1,
			   kVariants[order[0]].letter, kVariants[order[1]].letter, kVariants[order[2]].letter,
			   ms[0][r], ms[1][r], ms[2][r]);
	}

	// ---- 4. Report what was actually measured.
	// Each launch reads the whole matrix once and writes it once.
	const double bytesMoved = 2.0 * double(dIn.bytes());

	printf("\n== Summary (measured on this run) ==\n");
	printf("%-14s %-42s %10s %22s %12s\n",
		   "Variant", "What it measures", "median ms", "(min .. max)", "median GB/s");
	for (int v = 0; v < kNumVariants; ++v)
	{
		std::vector<double> gbps(kRounds);
		for (int r = 0; r < kRounds; ++r)
			gbps[r] = bytesMoved / (ms[v][r] * 1e6);
		printf("%-14s %-42s %10.4f   (%.4f .. %.4f) %12.1f\n",
			   kVariants[v].name, kVariants[v].measures,
			   median(ms[v]), minOf(ms[v]), maxOf(ms[v]), median(gbps));
	}
	printf("(GB/s counts one read + one write of the %d x %d matrix per launch)\n\n",
		   kHeight, kWidth);

	printf("Speedups (per-round ratios, then median):\n");
	printRatio("Tiled  vs Naive:",  ratios(ms[0], ms[1]));
	printRatio("Padded vs Naive:",  ratios(ms[0], ms[2]));
	printRatio("Padded vs Tiled:",  ratios(ms[1], ms[2]));

	// Order check: for each variant, median time when it ran 1st, 2nd and 3rd.
	printf("\nOrder check (median ms when the variant ran 1st / 2nd / 3rd in a round):\n");
	for (int v = 0; v < kNumVariants; ++v)
	{
		double byPos[kNumVariants];
		for (int p = 0; p < kNumVariants; ++p)
		{
			std::vector<double> sel;
			for (int r = 0; r < kRounds; ++r)
				if (position[v][r] == p)
					sel.push_back(ms[v][r]);
			byPos[p] = median(sel);
		}
		const double hi = std::max({byPos[0], byPos[1], byPos[2]});
		const double lo = std::min({byPos[0], byPos[1], byPos[2]});
		const double gapPct = 100.0 * (hi - lo) / (0.5 * (hi + lo));
		printf("  %-14s %.4f / %.4f / %.4f   gap %.1f%%  %s\n",
			   kVariants[v].name, byPos[0], byPos[1], byPos[2], gapPct,
			   gapPct > 5.0 ? "<- above 5%: run order affects this variant"
							: "(within 5%: no clear order bias)");
	}

	return EXIT_SUCCESS;
}
