// #include stdlib.h for malloc/free
#include <stdlib.h>

// #include stdio.h for printf
#include <stdio.h>

// nvcc automatically #includes headers needed for cudaMalloc, cudaFree, cudaMemcpy, & cudaMemset


int main(void)
{
  // create arrays of 16 elements
  int num_elements = 16;

  // compute the size of the arrays in bytes
  int num_bytes = num_elements * sizeof(int);
  
  // pointers to host & device arrays
  int *device_array = 0;
  int *host_array = 0;
  
  // malloc a host array
  host_array = (int*)malloc(num_bytes);

  // cudaMalloc a device array
  // we pass cudaMalloc a pointer to the device_array pointer
  cudaMalloc((void**)&device_array, num_bytes);
  
  // if either memory allocation failed, report an error message
  if(host_array == 0 || device_array == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }
  
  // zero out the device array with cudaMemset
  cudaMemset(device_array, 0, num_bytes);

  // copy the contents of the device array to the host array to inspect the result
  // use cudaMemcpyDeviceToHost to indicate the direction of the copy
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);
  
  // print out the result element by element
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n");
  
  // use free to free the host array
  free(host_array);

  // use cudaFree to free the device array
  cudaFree( device_array );
}

