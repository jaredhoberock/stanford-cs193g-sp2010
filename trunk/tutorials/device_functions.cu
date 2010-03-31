// This example introduces __device__ functions, which are special functions
// which may be called from code executing on the device.

#include <stdlib.h>
#include <stdio.h>


// __device__ functions may only be called from __global__ functions or other
// __device__ functions.  Unlike __global__ functions, __device__ functions are
// not configured, and have no restriction on return type.
__device__ int get_constant(void)
{
  // just return 7
  return 7;
}

__device__ int get_block_index(void)
{
  // return the index of the current thread's block
  return blockIdx.x;
}

__device__ int get_thread_index(void)
{
  // return the index of the current thread within its block
  return threadIdx.x;
}

__device__ int get_global_index(void)
{
  // return the index of the current thread across the entire grid launch
  return blockIdx.x * blockDim.x + threadIdx.x;
}


// kernel1 returns the result of calling the __device__ function return_constant():
__global__ void kernel1(int *array)
{
  int index = get_global_index();
  array[index] = get_constant();
}


// kernel2 returns the result of calling the __device__ function return_block_index():
__global__ void kernel2(int *array)
{
  int index = get_global_index();
  array[index] = get_block_index();
}


// kernel3 returns the result of calling the __device__ function return_thread_index():
__global__ void kernel3(int *array)
{
  int index = get_global_index();
  array[index] = get_thread_index();
}


// kernel4 returns the result of calling the __device__ function return_thread_index():
__global__ void kernel4(int *array)
{
  int index = get_global_index();
  array[index] = get_global_index();
}


int main(void)
{
  int num_elements = 256;

  int num_bytes = num_elements * sizeof(int);

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

  // choose a launch configuration
  int block_size = 128;
  int grid_size = num_elements / block_size;

  // launch each kernel and print out the results

  kernel1<<<grid_size,block_size>>>(device_array);

  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel1 results:\n");
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");


  kernel2<<<grid_size,block_size>>>(device_array);

  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel2 results:\n");
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");


  kernel3<<<grid_size,block_size>>>(device_array);

  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel3 results:\n");
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");


  kernel4<<<grid_size,block_size>>>(device_array);

  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  printf("kernel4 results:\n");
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n\n");


  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}

