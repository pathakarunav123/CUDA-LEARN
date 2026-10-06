// A common strategy is to partition the data into subsets called tiles
//such that each tile fits into the shared memory. The term tile draws on the
//analogy that a large wall (i.e., the global memory data) can often be
//covered by tiles (i.e., subsets that each can fit into the shared memory).
//An important criterion is that the kernel computations on these tiles can
//be done independently of each other. Note that not all data structures
//can be partitioned into tiles given an arbitrary kernel function.

