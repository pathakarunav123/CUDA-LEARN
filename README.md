# CUDA-LEARN

<p align="center">
  <strong>A hands-on collection of CUDA C++ examples for learning GPU programming.</strong>
  <br />
  Explore kernels, memory access, shared memory, tiling, and performance measurement through small, runnable programs.
</p>

---

## Overview

This repository is a learning workspace for NVIDIA CUDA. The examples build from basic kernel launches and element-wise operations to matrix algorithms and memory-focused experiments.

```mermaid
flowchart LR
    A[CUDA basics] --> B[Vector and matrix operations]
    B --> C[Global and shared memory]
    C --> D[Tiling and access patterns]
    D --> E[Correctness checks and benchmarks]
```

The code is intended for learning and experimentation. Individual examples may use different approaches to error handling, validation, or timing.

## Examples

| Topic | Files | What to explore |
|---|---|---|
| CUDA basics | [`hello.cu`](./hello.cu), [`test_nvcc_min.cu`](./test_nvcc_min.cu) | Launching a kernel and identifying GPU threads |
| Element-wise operations | [`vectorAdd.cu`](./vectorAdd.cu), [`matrixAdd.cu`](./matrixAdd.cu), [`copyMat.cu`](./copyMat.cu) | Indexing, bounds checks, and transferring data between host and device |
| Memory and access patterns | [`memory.cu`](./memory.cu), [`shared_memo.cu`](./shared_memo.cu), [`stride2.cu`](./stride2.cu), [`stride32.cu`](./stride32.cu), [`bandwidth.md`](./bandwidth.md) | Shared memory, strides, coalescing, and effective bandwidth |
| Matrix operations | [`matTranspose.cu`](./matTranspose.cu), [`matMul.cu`](./matMul.cu), [`Tiling.cu`](./Tiling.cu), [`naiveVsTile.cu`](./naiveVsTile.cu) | Transpose, matrix multiplication, and tiled computation |
| Transpose benchmark | [`NaiveVsTiled_trans.cu`](./NaiveVsTiled_trans.cu), [visual explanation](./NaiveVsTiled_trans_README.md) | Comparing naive, tiled, and padded transpose kernels with correctness checks and repeated timings |

## Prerequisites

- An NVIDIA GPU supported by the installed CUDA Toolkit
- NVIDIA CUDA Toolkit, including `nvcc`
- CMake 3.24 or newer
- A C++/CUDA-capable compiler supported by your CUDA Toolkit

Check that the compiler and CMake are available:

```powershell
nvcc --version
cmake --version
```

## Build with CMake

The CMake project configures the standard example targets and uses the GPU's native architecture by default.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

To select a specific CUDA architecture instead, pass its compute capability as `CMAKE_CUDA_ARCHITECTURES` when configuring. For example, replace `89` below with the architecture supported by your GPU and CUDA Toolkit:

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

The CMake targets include `hello`, `vectorAdd`, `matrixAdd`, `copyMat`, `matTranspose`, `matMul`, `Tiling`, `naiveVsTile`, and `shared_memo`.

### Run an example

On Windows PowerShell:

```powershell
.\vectorAdd.exe
```

On Linux:

```bash
./vectorAdd
```

The current CMake configuration writes executables into the repository root. Rebuild after editing an example to run the updated program.

## Build the transpose benchmark

`NaiveVsTiled_trans.cu` is a standalone benchmark and is not currently one of the CMake targets. Build it directly with `nvcc`:

```powershell
nvcc -O3 -std=c++17 -arch=native NaiveVsTiled_trans.cu -o NaiveVsTiled_trans.exe
.\NaiveVsTiled_trans.exe
```

The `-arch=native` option requires CUDA 11.6 or newer. With an older Toolkit, replace it with the appropriate architecture flag, such as `-arch=sm_XX`.

The benchmark verifies all three kernels against a CPU reference before timing, checks non-tile-sized matrices, rotates kernel execution order, and reports timing and bandwidth summaries. For its explanation, see [NaiveVsTiled_trans_README.md](./NaiveVsTiled_trans_README.md).

## Learning path

1. Start with [`hello.cu`](./hello.cu) to see a kernel launch and CUDA thread indices.
2. Try [`vectorAdd.cu`](./vectorAdd.cu) and [`matrixAdd.cu`](./matrixAdd.cu) to practice data movement and element indexing.
3. Explore [`shared_memo.cu`](./shared_memo.cu), [`stride2.cu`](./stride2.cu), and [`stride32.cu`](./stride32.cu) to understand memory behavior.
4. Compare [`matMul.cu`](./matMul.cu) with [`Tiling.cu`](./Tiling.cu) to study tiled matrix multiplication.
5. Run [`NaiveVsTiled_trans.cu`](./NaiveVsTiled_trans.cu) to explore transpose coalescing, shared-memory bank conflicts, and benchmarking.

## Contributing

Ideas, corrections, and additional CUDA learning examples are welcome. Keep examples focused, document any hardware or Toolkit requirements, and include correctness checks where practical.

## License

No license is currently specified. Unless a license is added, the repository's contents are not granted for reuse, redistribution, or modification.
