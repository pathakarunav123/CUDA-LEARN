In CUDA programming, bandwidth is the rate at which data is transferred between the GPU's processors and its memory hierarchy. It measures how fast your GPU can read or write data, typically expressed in Gigabytes per second (GB/s).


The Two Types of BandwidthTo understand performance, you must compare two values:Theoretical Peak Bandwidth: The absolute maximum speed the GPU hardware is physically capable of achieving under perfect conditions.

Effective (Achieved) Bandwidth: The actual speed your specific program achieves when running a kernel. This is calculated using the formula from before: (Bytes Read + Bytes Written) / Time.

Why We Focus on Achieving High BandwidthWe care about achieving high bandwidth because most CUDA applications are memory-bound, not compute-bound.

The Memory Wall: Modern GPU arithmetic units (ALUs) are incredibly fast. They can process data much faster than global memory can supply it.

Starving the GPU: If your memory bandwidth is low, your powerful GPU processors sit idle doing nothing while waiting for data to arrive from the main graphics memory (VRAM).

Maximising Throughput: By optimizing your code to achieve an effective bandwidth close to the theoretical peak, you eliminate this bottleneck, ensuring the processors are fed data continuously.

How We Achieve High BandwidthWe achieve high bandwidth primarily through coalesced memory access.
Hardware Design: The GPU does not fetch data byte-by-byte. It fetches data in large blocks (usually 32, 64, or 128 bytes at a time).

The Strategy: When all 32 threads in a warp request data from a single continuous block, the GPU can satisfy all 32 threads with just one single memory transaction.

The Penalty of Misalignment: If your threads look for data scattered randomly across memory, the GPU must execute 32 separate memory transactions to get the same amount of data. This wastes time, clogs the memory bus, and causes your achieved bandwidth to plummet.