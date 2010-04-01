/* This is machine problem 1, part 1, shift cypher
 *
 * The problem is to take in a string of unsigned ints and an int,
 * the shift amount, and add the number to each element of
 * the string, effectively "shifting" each element in the 
 * string.
 * SUBMISSION GUIDELINES:
 * You should copy the complete shift_cyper function from your solution
 * into a file called mp1-part1-solution-kernel.cu and submit that file.
 * The function needs to have exactly the same interface (including __global__)
 * as the empty shift_cypher function given below. 
 */


#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include "mp1-util.h"


// Repeating from the tutorial, just in case you haven't looked at it.

// "kernels" or __global__ functions are the entry points to code that executes on the GPU
// The keyword __global__ indicates to the compiler that this function is a GPU entry point.
// __global__ functions must return void, and may only be called or "launched" from code that
// executes on the CPU.

void host_shift_cypher(uint *input_array, uint *output_array, int shift_amount, int alphabet_max, int array_length)
{
  int i;
  for(i=0;i<array_length;i++)
  {
    int element = input_array[i];
    int shifted = element + shift_amount;
    if(shifted > alphabet_max || shifted < 0)
    {
      shifted = shifted % (alphabet_max + 1);
    }
    output_array[i] = (uint)shifted;
  }
}


// This kernel implements a per element shift
__global__ void shift_cypher(uint *input_array, uint *output_array, int shift_amount, int alphabet_max, int array_length)
{
  // your code here
}


int main(void)
{
  // initialize
  srand(time(NULL));
  
  // create arrays of 16M elements
  int num_elements = 1 << 24;

  
  int alphabet_max = 45647;
  
  // compute the size of the arrays in bytes
  int num_bytes = num_elements * sizeof(uint);

  // pointers to host & device arrays
  uint *host_input_array = 0;
  uint *host_output_array = 0;
  uint *host_output_checker_array = 0;
  uint *device_input_array = 0;
  uint *device_output_array = 0;
  
  event_pair timer;
  

  // malloc host arrays
  host_input_array = (uint*)malloc(num_bytes);
  host_output_array = (uint*)malloc(num_bytes);
  host_output_checker_array = (uint*)malloc(num_bytes);

  // cudaMalloc device arrays
  cudaMalloc((void**)&device_input_array, num_bytes);
  cudaMalloc((void**)&device_output_array, num_bytes);
  
  // if either memory allocation failed, report an error message
  if(host_input_array == 0 || host_output_array == 0 || host_output_checker_array == 0 || 
	device_input_array == 0 || device_output_array == 0)
  {
    printf("couldn't allocate memory\n");
    return 1;
  }


  // generate random input string
  int shift_amount = rand();
  
  for(int i=0;i< num_elements;i++)
  {
    host_input_array[i] = (uint)rand(); 
  }
  
  // do copies to and from gpu once to get rid of timing weirdness
  // on first time accesses due to driver
  cudaMemcpy(device_input_array, host_input_array, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);

  start_timer(&timer);
  // copy input to GPU
  cudaMemcpy(device_input_array, host_input_array, num_bytes, cudaMemcpyHostToDevice);
  check_launch("copy to gpu");
  stop_timer(&timer,"copy to gpu");
  
  // choose a number of threads per block
  // we use 512 threads here
  int block_size = 512;

  int grid_size = num_elements / block_size;

  start_timer(&timer);
  // launch kernel
  shift_cypher<<<grid_size,block_size>>>(device_input_array, device_output_array, shift_amount, alphabet_max, num_elements);
  check_launch("gpu shift cypher");
  stop_timer(&timer,"gpu shift cypher");

  start_timer(&timer);
  // download and inspect the result on the host:
  cudaMemcpy(host_output_array, device_output_array, num_bytes, cudaMemcpyDeviceToHost);
  check_launch("copy from gpu");
  stop_timer(&timer,"copy from gpu");
  
  start_timer(&timer);
  // generate reference output
  host_shift_cypher(host_input_array, host_output_checker_array, shift_amount, alphabet_max, num_elements);
  stop_timer(&timer,"host shift cypher");
  
  // check CUDA output versus reference output
  int error = 0;
  for(int i=0;i<num_elements;i++)
  {
    if(host_output_array[i] != host_output_checker_array[i]) 
    { 
      error = 1;
    }
  }
  
  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else
  {
    printf("Worked! CUDA and reference output match. \n");
  }
 
  // deallocate memory
  free(host_input_array);
  free(host_output_array);
  free(host_output_checker_array);
  cudaFree(device_input_array);
  cudaFree(device_output_array);
}

