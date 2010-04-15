// This example demonstrates a block-wise inclusive
// parallel prefix sum (scan) algorithm.

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>


// This kernel computes, per-block, a block-sized scan
// of the input.  It assumes that the block size evenly
// divides the input size
__global__ void inclusive_scan(const unsigned int *input,
                               unsigned int *result)
{
  extern __shared__ unsigned int sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  unsigned int sum = input[i];
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for(int offset = 1; offset < blockDim.x; offset <<= 1)
  {
    if(threadIdx.x >= offset)
    {
      sum += sdata[threadIdx.x - offset];
    }

    // wait until every thread has updated its partial sum
    __syncthreads();

    // write my partial sum
    sdata[threadIdx.x] = sum;

    // wait until every thread has written its partial sum
    __syncthreads();
  }

  // we're done! each thread writes out its result
  result[i] = sdata[threadIdx.x];
}
                               

int main(void)
{
  // use small input sizes for illustrative purposes
  const int num_blocks = 4;
  const int block_size = 16;
  const int num_elements = num_blocks * block_size;

  // generate random input in [0,5] on the host
  std::vector<unsigned int> h_input(num_elements);
  for(unsigned int i = 0; i < num_elements; ++i)
  {
    h_input[i] = rand() % 6;
  }

  // copy input to device memory
  unsigned int *d_input = 0;
  cudaMalloc((void**)&d_input, sizeof(unsigned int) * num_elements);
  cudaMemcpy(d_input, &h_input[0], sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice);

  // allocate space for the result
  unsigned int *d_result = 0;
  cudaMalloc((void**)&d_result, sizeof(unsigned int) * num_elements);

  inclusive_scan<<<num_blocks, block_size, block_size * sizeof(unsigned int)>>>(d_input, d_result);

  // copy result to host memory
  std::vector<unsigned int> h_result(num_elements);
  cudaMemcpy(&h_result[0], d_result, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost);

  // print out the results
  for(int b = 0; b < num_blocks; ++b)
  {
    std::cout << "Block " << b << std::endl << std::endl;

    std::cout << "Input: " << std::endl;
    for(int i = 0; i < block_size; ++i)
    {
      printf("%2d ", h_input[b * block_size + i]);
    }
    std::cout << std::endl;

    std::cout << "Result: " << std::endl;
    for(int i = 0; i < block_size; ++i)
    {
      printf("%2d ", h_result[b * block_size + i]);
    }
    std::cout << std::endl << std::endl << std::endl;
  }

  return 0;
}

