#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void add(int *a, int *b, int *c, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        c[i] = a[i] + b[i];
    } 
    //illeagel memory 
   // c[i] = a[i] + b[i];
}
int main(){
     int N =100;
   int *d_a, *d_b, *d_c;
   int h_a[100]={1,2,3,4,5,6,7,8,9};
   int h_b[100]={6,5,6,7,8,9,10,11,12};
   int h_c[100];
   size_t size = N * sizeof(int);
   CUDA_CHECK(cudaMalloc((void**)&d_a, size));
   CUDA_CHECK(cudaMalloc((void**)&d_b, size));
   CUDA_CHECK(cudaMalloc((void**)&d_c, size));

   CUDA_CHECK(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

   int threadsPerBlock = 256;
   int blocks = (N+threadsPerBlock-1)/threadsPerBlock;
   add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
   CUDA_CHECK(cudaGetLastError());


  // cudaError_t err1 = cudaGetLastError();
    //printf("GET 1: %s\n", cudaGetErrorString(err1));

    //cudaError_t err2 = cudaGetLastError();
    //printf("GET 2: %s\n", cudaGetErrorString(err2));
   //cudaPeekAtLatError()
   //CUDA_CHECK(cudaPeekAtLastError());
  // cudaError_t err1 = cudaPeekAtLastError();


//printf("PEEK 1: %s\n", cudaGetErrorString(err1));

//cudaError_t err2 = cudaPeekAtLastError();
//printf("PEEK 2: %s\n", cudaGetErrorString(err2));

   CUDA_CHECK(cudaDeviceSynchronize());
   CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
   for(int i=0; i<N; i++){
    printf("%d + %d = %d\n", h_a[i],h_b[i],h_c[i]);
   }

   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

    return 0;
 

}