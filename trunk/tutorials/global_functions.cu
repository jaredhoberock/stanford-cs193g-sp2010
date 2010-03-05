// This example introduces CUDA's abstraction of data parallel computational
// "kernels", or __global__ functions.  A __global__ function acts like the
// main() function of a GPU program, and is allowed to manipulate device
// memory directly.

#include <stdlib.h>
#include <stdio.h>


// "kernels" or __global__ functions are the entry points to code that executes on the GPU
// The keyword __global__ indicates to the compiler that this function is a GPU entry point.
// __global__ functions must return void, and may only be called or "launched" from code that
// executes on the CPU.
__global__ void kernel(int *array)
{
  // compute the index of this particular thread
  // in the grid:

  // multiply the index of this thread's block (blockIdx.x)
  // by the number of threads per block (blockDim.x)
  // and add the index of this thread inside its block (threadIdx.x)

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // write out 7 to a single element of the array using standard
  // array indexing notation:
  array[index] = 7;
}


int main(void)
{
  // create arrays of 256 elements
  int num_elements = 256;

  // compute the size of the arrays in bytes
  int num_bytes = num_elements * sizeof(int);

  // pointers to host & device arrays
  int *device_array = 0;
  int *host_array = 0;

  // malloc a host array
  host_array = (int*)malloc(num_bytes);

  // cudaMalloc a device array
  cudaMalloc((void**)&device_array, num_bytes);

  // if either memory allocation failed, report an error message
  if(host_array == 0 || device_array == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // launch the global function by choosing the number of CUDA threads
  // to instantiate:

  // choose a number of threads per block
  // 128 threads (4 warps) tends to be a good number
  int block_size = 128;

  // divide the number of elements to process by the block size
  // to determine the number of blocks to launch
  int grid_size = num_elements / block_size;

  // To invoke the global function, use the triple chevron notation.
  // The first argument is the number of blocks (grid_size).
  // The second argument is the number of threads per block (block_size).
  // This is called "configuring" the launch.
  // After the triple chevrons, pass function arguments as normal.
  kernel<<<grid_size,block_size>>>(device_array);

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n");
 
  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}

