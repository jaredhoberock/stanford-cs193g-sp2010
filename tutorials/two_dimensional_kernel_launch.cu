// This example demonstrates how to launch two dimensional grids of CUDA threads.

#include <stdlib.h>
#include <stdio.h>


__global__ void kernel(int *array)
{
  // compute the two dimensional index of this particular
  // thread in the grid

  // do the usual computation separately in each dimension:
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // use the two 2D indices to compute a single linear index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  // use the two 2D block indices to compute a single linear block index
  int result = blockIdx.y * gridDim.x + blockIdx.x;

  // write out the result
  array[index] = result;
}


int main(void)
{
  int num_elements_x = 16;
  int num_elements_y = 16;

  int num_bytes = num_elements_x * num_elements_y * sizeof(int);

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

  // choose a two dimensional launch configuration
  // use the dim3 type when launches are not one dimensional

  // create 4x4 thread blocks
  dim3 block_size;
  block_size.x = 4;
  block_size.y = 4;

  // configure a two dimensional grid as well
  dim3 grid_size;
  grid_size.x = num_elements_x / block_size.x;
  grid_size.y = num_elements_y / block_size.y;

  // grid_size & block_size are passed as arguments to the
  // triple chevrons as usual
  kernel<<<grid_size,block_size>>>(device_array);

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int row = 0; row < num_elements_y; ++row)
  {
    for(int col = 0; col < num_elements_x; ++col)
    {
      printf("%2d ", host_array[row * num_elements_x + col]);
    }
    printf("\n");
  }
  printf("\n");

  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}

