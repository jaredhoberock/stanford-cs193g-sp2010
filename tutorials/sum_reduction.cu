// This example demonstrates a parallel sum reduction
// using two kernel launches

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>


float random_float(void)
{
  return static_cast<float>(rand()) / RAND_MAX;
}


// this kernel computes, per-block, the sum
// of a block-sized portion of the input
// using a block-wide reduction
__global__ void block_sum(const float *input,
                          float *per_block_results,
                          const size_t n)
{
  extern __shared__ float sdata[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // load input into __shared__ memory
  float x = 0;
  if(i < n)
  {
    x = input[i];
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  // contiguous range pattern
  for(int offset = blockDim.x / 2;
      offset > 0;
      offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      // add a partial sum upstream to our own
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }

    // wait until all threads in the block have
    // updated their partial sums
    __syncthreads();
  }

  // thread 0 writes the final result
  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}


int main(void)
{
  // create array of 256k elements
  const int num_elements = 1<<18;

  // generate random input on the host
  std::vector<float> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = random_float();
  }

  const float host_result = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
  std::cerr << "Host sum: " << host_result << std::endl;

  // move input to device memory
  float *d_input = 0;
  cudaMalloc((void**)&d_input, sizeof(float) * num_elements);
  cudaMemcpy(d_input, &h_input[0], sizeof(float) * num_elements, cudaMemcpyHostToDevice);

  const size_t block_size = 512;
  const size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);

  // allocate space to hold one partial sum per block, plus one additional
  // slot to store the total sum
  float *d_partial_sums_and_total = 0;
  cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1));

  // launch one kernel to compute, per-block, a partial sum
  block_sum<<<num_blocks,block_size,block_size * sizeof(float)>>>(d_input, d_partial_sums_and_total, num_elements);

  // launch a single block to compute the sum of the partial sums
  block_sum<<<1,num_blocks,num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);

  // copy the result back to the host
  float device_result = 0;
  cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Device sum: " << device_result << std::endl;

  // deallocate device memory
  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);

  return 0;
}

