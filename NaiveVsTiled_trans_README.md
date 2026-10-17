# CUDA Matrix Transpose: Naive vs Tiled vs Padded

This note explains the GPU transpose experiment in `NaiveVsTiled_trans.cu` using a visual, easy-to-follow breakdown.

---

## 1. Program goal

The program compares three different CUDA transpose implementations:

- Naive transpose
- Tiled transpose (32x32 tile)
- Padded tiled transpose (32x33 tile)

It checks correctness against a CPU reference, warms up the kernels, benchmarks all variants, and prints timing summary and speedups.

---

## 2. What the matrix looks like

A matrix is stored in row-major order:

```text
Input matrix (height x width)

  c0 c1 c2 c3
r0 [ 1  2  3  4 ]
r1 [ 5  6  7  8 ]
r2 [ 9 10 11 12 ]

Memory layout in 1D:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

Transpose means:

```text
out[row][col] = in[col][row]
```

So the new matrix becomes:

```text
Transpose result

  r0 r1 r2
c0 [ 1  5  9 ]
c1 [ 2  6 10 ]
c2 [ 3  7 11 ]
c3 [ 4  8 12 ]
```

---

## 3. High-level flow of the program

```text
CPU side
  |
  |-- Generate random matrix
  |
  |-- Verify all kernels against a CPU transpose
  |      -> naiveTranspose
  |      -> tiledTranspose
  |      -> paddedTranspose
  |
  |-- Warm up all kernels
  |
  |-- Run 12 benchmark rounds
  |      -> rotate through all 6 execution orders
  |      -> record timing for each kernel
  |
  |-- Print median / min / max / bandwidth / speedups
  |
  '-- Show order-bias analysis
```

---

## 4. Kernel 1: Naive transpose

```text
Global memory
  input[row * width + col]
         |
         v
Each CUDA thread handles one element

  thread x = blockIdx.x * blockDim.x + threadIdx.x
  thread y = blockIdx.y * blockDim.y + threadIdx.y

  if (row < height && col < width)
      output[col * height + row] = input[row * width + col]
```

### Visual idea

Each thread does:

```text
read one matrix element
write it to its transposed location
```

Problem:
- reads are coalesced
- writes are strided
- performance suffers due to non-contiguous writes

```text
Naive write pattern
before:
[ a b c d ]
[ e f g h ]

after transpose:
[ a e i ... ]
[ b f j ... ]
```

This causes memory writes to be spread out in global memory.

---

## 5. Kernel 2: Tiled transpose (32x32)

```text
Global memory
     |
     v
  Load 32x32 block into shared memory

  __shared__ float tile[32][32]

  tile[threadIdx.y][threadIdx.x] = input[row * width + col]

  __syncthreads()

  outRow = blockIdx.x * TILE + threadIdx.y
  outCol = blockIdx.y * TILE + threadIdx.x

  output[outRow * height + outCol] = tile[threadIdx.x][threadIdx.y]
```

### Visual explanation

```text
Tile block (32x32)
┌─────────────────────────┐
│  thread grid           │
│  [x,y] positions       │
└─────────────────────────┘
         |
         | load into shared memory
         v
┌─────────────────────────┐
│ tile[32][32]            │
│ shared block data       │
└─────────────────────────┘
         |
         | transpose inside tile
         v
  write transposed values back to global memory
```

### Why use shared memory?

Shared memory lets threads cooperate on a small tile before writing to global memory, reducing inefficient access patterns.

---

## 6. Kernel 3: Padded tiled transpose (32x33)

```text
__shared__ float tile[32][33]
```

This is the same idea as the tiled version, except the tile has an extra column.

### Why padding helps

In shared memory, threads can conflict on memory banks. Padding changes the layout so that threads reading/writing nearby elements avoid bank conflict patterns.

```text
Without padding
  tile[32][32]

With padding
  tile[32][33]

This produces a slightly better memory layout for transpose access.
```

So the logic is the same, but the bank arrangement changes.

---

## 7. Thread block and grid structure

The program uses:

```text
threads = dim3(32, 32)
blocks = ceil(width / 32), ceil(height / 32)
```

### Example

```text
For a 4096 x 4096 matrix:
  blocks.x = 128
  blocks.y = 128
  total threads = 128 * 128 * 1024 = ~16 million threads
```

```text
Block index
  blockIdx.x -> x-position of block
  blockIdx.y -> y-position of block

Thread index
  threadIdx.x -> x inside block
  threadIdx.y -> y inside block
```

---

## 8. Correctness check

The program verifies each kernel using an exact CPU transpose.

```text
expected = cpuTranspose(input)
GPU output = kernel(input)

if (GPU output != expected)
   kernel failed
```

It tests sizes such as:

- 33 x 47
- 100 x 257
- 257 x 100
- 5 x 70
- 1024 x 1024

This ensures the kernels work even for uneven or non-tile-sized matrices.

---

## 9. Benchmarking strategy

The program does not run only once. It does more careful performance measurement:

```text
1. Warm-up each kernel 10 times
2. Run 12 rounds
3. Use 6 execution order permutations
4. Measure 20 launches per round
5. Record ms per launch
```

### Why rotate execution order?

To avoid a false result caused by startup effects or thermal changes.

Example orders:

```text
N -> T -> P
N -> P -> T
T -> N -> P
T -> P -> N
P -> N -> T
P -> T -> N
```

This prevents one kernel from always going first or last.

---

## 10. Timing summary and speedup logic

For each kernel, the program computes:

- median runtime
- minimum runtime
- maximum runtime
- bandwidth in GB/s
- per-round speedup ratios

### Formula

```text
Bandwidth ≈ (read bytes + write bytes) / time
```

Because each transpose reads the matrix once and writes it once:

```text
bytesMoved = 2 * matrix_size_in_bytes
```

Then it compares:

```text
Tiled vs Naive
Padded vs Naive
Padded vs Tiled
```

---

## 11. Visual performance comparison

```text
Runtime comparison
Naive      ────────────────
Tiled      ─────────────
Padded     ─────────────

Lower is better
```

Typical expectation:

```text
Naive  >  Tiled  >  Padded
```

But the actual ordering may vary with GPU architecture, memory subsystem, and workload size.

---

## 12. Final interpretation

This experiment teaches the following CUDA concepts:

- coalesced memory access matters
- shared memory can improve transpose performance
- tile-based algorithms help with locality
- padding can reduce shared-memory bank conflicts
- benchmarking must be fair and repeatable

In short:

```text
Naive transpose = simple and correct, but memory-inefficient
Tiled transpose = better due to shared memory reuse
Padded transpose = tuned tiled version with improved bank access
```

---

## 13. Short summary

```text
Naive      -> direct global-memory transpose
Tiled      -> shared-memory tile transpose
Padded     -> tiled transpose with bank conflict reduction
```

This file is a practical CUDA study of how memory access patterns affect performance on GPUs.
