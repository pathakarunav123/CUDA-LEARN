#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello()
{
    printf("Hello from GPU! thread = %d, block = %d\n", threadIdx.x, blockIdx.x);
}
//Vector Addition kernel

int main()
{
    hello<<<2, 8>>>();
   // hello<<<2,4>>>();
   cudaDeviceSynchronize();
   return 0;
}