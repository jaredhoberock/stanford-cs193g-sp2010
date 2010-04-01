// This example demonstrates how to check for CUDA errors,
// which indicate programming bugs, or otherwise broken
// program behavior.

#include <stdlib.h>
#include <stdio.h>


// our old friend, the "fill-with-7" kernel
__global__ void kernel(int *array)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
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

  // choose a grid configuration
  int block_size = 128;
  int grid_size = num_elements / block_size;

  // accidentally pass the kernel a pointer to the host array
  // this will cause the kernel to crash when it attempts to
  // dereferences a host pointer
  kernel<<<grid_size,block_size>>>(host_array);

  // make the host wait until the kernel is finished executing before
  // checking for the last CUDA error. otherwise, we won't detect
  // an error if one occurred
  cudaThreadSynchronize();

  // this kind of "blocking" operation is usually only appropriate during
  // the debugging process because it forces the host to wait on the device
  // while it could be busy doing other things. once the code has been
  // debugged, frequent error checking code should be eliminated or disabled

  // ask CUDA for the last error to occur (if one exists)
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // something's gone wrong
    // print out the CUDA error as a string
    printf("CUDA Error: %s\n", cudaGetErrorString(error));

    // we can't recover from the error -- exit the program
    return 1;
  }

  // no error occurred, proceed as usual

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the first 10 elements
  for(int i=0; i < 10; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n");

  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}

