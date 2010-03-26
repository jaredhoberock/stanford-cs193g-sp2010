// This example demonstrates the use of shared per-block variables to
// implement an optimized adjacent difference algorithm.  In this example,
// a per-block __shared__ array acts as a "bandwidth multiplier" by eliminating
// redundant loads issued by neighboring threads.

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>

// compute the number of lines of code each implementation requires
const unsigned int simple_implementation_begin = __LINE__;

// a simple version of adjacent_difference which issues redundant loads from off-chip global memory
__global__ void adjacent_difference_simple(int *result, int *input)
{
  // compute this thread's global index
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i > 0)
  {
    // each thread loads two elements from global memory
    int x_i = input[i];
    int x_i_minus_one = input[i-1];

    // compute the difference using values stored in registers
    result[i] = x_i - x_i_minus_one;
  }
}
const unsigned int simple_implementation_size = __LINE__ - simple_implementation_begin;


const unsigned int optimized_implementation_begin = __LINE__;

// an optimized version of adjacent_difference which eliminates redundant loads
__global__ void adjacent_difference(int *result, int *input)
{
  // a __shared__ array with one element per thread
  // the size of the array is allocated dynamically upon kernel launch
  extern __shared__ int s_data[];

  // each thread reads one element to s_data
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  // since one array gets allocated per-block, we index the array using our
  // per-block thread index, threadIdx
  // the global array, input, is indexed as usual
  s_data[threadIdx.x] = input[i];

  // avoid race condition: ensure all loads to s_data complete before we try to read from it
  __syncthreads();

  if(threadIdx.x > 0)
  {
    // compute the difference directly from s_data
    // it is implemented in fast, on-chip memory
    result[i] = s_data[threadIdx.x] - s_data[threadIdx.x - 1];
  }
  else if(i > 0)
  {
    // handle thread block boundary
    // the first thread in a block needs data that was read by the
    // last thread of the previous block into its shared array
    // this thread can't access that array, so issue one redundant load per block
    result[i] = s_data[threadIdx.x] - input[i-1];
  }
}
const unsigned int optimized_implementation_size = __LINE__ - optimized_implementation_begin;


int main(void)
{
  // create a large workload so we can easily measure the
  // performance difference of both implementations
  const size_t block_size = 512;
  const size_t num_blocks = (1<<24) / block_size;
  const size_t n = num_blocks * block_size;

  // generate random input on the host
  std::vector<int> h_input(n);
  std::generate(h_input.begin(), h_input.end(), rand);

  // allocate storage for the device
  int *d_input = 0, *d_result = 0;
  cudaMalloc((void**)&d_input, sizeof(int) * n);
  cudaMalloc((void**)&d_result, sizeof(int) * n);

  // copy input to the device
  cudaMemcpy(d_input, &h_input[0], sizeof(int) * n, cudaMemcpyHostToDevice);

  // time the kernel launches using CUDA events
  cudaEvent_t launch_begin, launch_end;
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);

  // to get accurate timings, launch a single "warm-up" kernel

  // dynamically allocate the __shared__ array by passing its
  // size in bytes to the 3rd parameter of the triple chevrons
  adjacent_difference_simple<<<num_blocks,block_size,block_size*sizeof(int)>>>(d_result, d_input);

  const size_t num_launches = 100;

  // time many kernel launches and take the average time
  float average_simple_time = 0;
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    adjacent_difference_simple<<<num_blocks,block_size,block_size*sizeof(int)>>>(d_result, d_input);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_simple_time += time;
  }
  average_simple_time /= num_launches;

  // now time the optimized kernel

  // again, launch a single "warm-up" kernel
  adjacent_difference<<<num_blocks,block_size,block_size*sizeof(int)>>>(d_result, d_input);

  // time many kernel launches and take the average time
  float average_optimized_time = 0;
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    adjacent_difference<<<num_blocks,block_size,block_size*sizeof(int)>>>(d_result, d_input);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_optimized_time += time;
  }
  average_optimized_time /= num_launches;

  // report the effective throughput of each kernel in GB/s
  // the effective throughput is measured as size of input read + size of output written divided by time
  float simple_throughput = static_cast<float>(2 * n * sizeof(int)) / (average_simple_time / 1000.0f) / 1000000000.0f;
  float optimized_throughput = static_cast<float>(2 * n * sizeof(int)) / (average_optimized_time / 1000.0f) / 1000000000.0f;

  // compute throughput per line of code to measure how productive we were
  float simple_throughput_per_sloc = simple_throughput / simple_implementation_size;
  float optimized_throughput_per_sloc = optimized_throughput / optimized_implementation_size;

  std::cout << "Work load size: " << n << std::endl;
  std::cout << "simple implementation SLOCs: " << simple_implementation_size << std::endl;
  std::cout << "Optimized implementation SLOCs: " << optimized_implementation_size << std::endl << std::endl;

  std::cout << "Throughput of simple kernel: " << simple_throughput << " GB/s" << std::endl;
  std::cout << "Throughput of optimized kernel: " << optimized_throughput << " GB/s" << std::endl;
  std::cout << "Performance improvement: " << optimized_throughput / simple_throughput << "x" << std::endl;
  std::cout << std::endl;

  std::cout << "Throughput of simple kernel per line of code: " << simple_throughput_per_sloc << " GB/s/sloc" << std::endl;
  std::cout << "Throughput of optimized kernel per line of code: " << optimized_throughput_per_sloc << " GB/s/sloc" << std::endl;
  std::cout << "Performance improvement per line of code: " << optimized_throughput_per_sloc / simple_throughput_per_sloc << "x" << std::endl;

  // destroy the CUDA events
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_input);
  cudaFree(d_result);

  return 0;
}

