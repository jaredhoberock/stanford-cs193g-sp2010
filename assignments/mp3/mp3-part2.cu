// Parallel prefix sum [1], also called "scan", is a basic parallel primitive
// which serves as a building block for more sophisticated parallel algorithms.
// Interestingly, while scan is a fundamental parallel algorithm, it is totally
// unnecessary in serial processing.
//
// In part 2, we'll implement an "exclusive" variant of scan to build a 
// higher-level primitive operation, stream compaction. We'll build our 
// compaction algorithm by combining an exclusive scan with another pattern,
// scatter. This part will also give you some experience in developing good
// pratices for organizing the modules of more substantial parallel
// applications
//
// Your job is tricky this time -- scan is a multi-kernel implementation, and
// we have not provided a serial reference. Start small, and work your way up,
// validating the output of each individual kernel as you go. We've provided
// you with a general outline and some implementation hints in scan.cu. The
// compaction implementation is also your responsibility, but is much simpler
// by comparison. Scatter is so easy that we've given it to you. Search this
// file, compact.cu, and scan.cu for TODO to find where you need to start
// hacking.
//
// In part 3, we'll use your stream compaction code to implement a simple
// scheduling scheme in order to improve the throughput of our sparse
// Black-Scholes evaluations.

// [1] http://en.wikipedia.org/wiki/Prefix_sum

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <limits>
#include <algorithm>

#include "compact.h"
#include "mp3-util.h"

bool allocate_device_storage(unsigned int *&d_input, unsigned int *&d_compacted_result, size_t n)
{
  // TODO: your device memory allocations here
  // TODO: don't forget to check for CUDA errors!

  // TODO: return true upon successful memory allocation
  return false;
}

void deallocate_device_storage(unsigned int *d_input, unsigned int *d_compacted_result)
{
  // TODO: your device memory deallocations here
  // TODO: don't forget to check for CUDA errors!
}

int main(void)
{
  event_pair timer;

  // create arrays for 8M elements
  size_t num_elements = 1<<23;

// when debugging, it's often easiest to
// start with small N and work your way up:
// size_t num_elements = 32;
// size_t num_elements = 1<<6;
// size_t num_elements = 1<<9;
// size_t num_elements = 1<<12;
// size_t num_elements = 1<<13;
// size_t num_elements = 1<<14;
// size_t num_elements = 1<<15;
// size_t num_elements = 1<<16;

  // TODO: you can also use this simple data set to debug your scan
  // implementation against the notes in scan.cu
  //unsigned int temp[] = {0, 1, 2, 4, 10, 2, 3, 5, 0, 3};

  // allocate host storage
  std::vector<unsigned int> h_input(num_elements);
  std::vector<unsigned int> h_compacted_result(h_input.size());

  // generate random input
  srand(13);
  for(size_t i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = rand();
  }

  // some pointers to the arrays which will live in device memory
  unsigned int *d_input = 0;
  unsigned int *d_compacted_result = 0;

  // allocate device storage
  if(!allocate_device_storage(d_input, d_compacted_result, num_elements))
  {
    std::cerr << "Error allocating device memory!" << std::endl;
    exit(-1);
  }

  // fill the device arrays with 0
  cudaMemset(d_input, 0, sizeof(unsigned int) * num_elements);
  cudaMemset(d_compacted_result, 0, sizeof(unsigned int) * num_elements);
  check_cuda_error("After cudaMemset", __FILE__, __LINE__);

  // copy input to GPU
  start_timer(&timer);
  // TODO: your host to device copies here
  stop_timer(&timer, "host to device copy of input");

  // compact the array on the gpu, pushing even elements to the front of the list
  start_timer(&timer);
  size_t num_compacted_elements = compact_even_elements(d_input, num_elements, d_compacted_result);
  float gpu_time = stop_timer(&timer, "GPU compaction");

  // compact the array on the cpu, pushing even elements to the front of the list
  start_timer(&timer);
  size_t num_compacted_elements_reference = 0;
  for(size_t i = 0; i < h_input.size(); ++i)
  {
    unsigned int x = h_input[i];
    if(is_even(x))
    {
      h_compacted_result[num_compacted_elements_reference] = x;
      ++num_compacted_elements_reference;
    }
  }
  float cpu_time = stop_timer(&timer, "CPU compaction");

  // resize the host result to the compacted size
  h_compacted_result.resize(num_compacted_elements_reference);

  // validate gpu results
  if(num_compacted_elements != num_compacted_elements_reference)
  {
    std::cerr << "Error: the number of compacted elements doesn't match the reference!" << std::endl;
    std::cerr << "num_compacted_elements: " << num_compacted_elements << std::endl;
    std::cerr << "num_compacted_elements_reference: " << num_compacted_elements_reference << std::endl;
    exit(-1);
  }

  std::vector<unsigned int> h_validate_me(num_elements);
  cudaMemcpy(&h_validate_me[0], d_compacted_result, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost);
  if(!std::equal(h_compacted_result.begin(), h_compacted_result.end(), h_validate_me.begin()))
  {
    std::cerr << "Error: the compacted results don't match!" << std::endl;
    size_t num_errors = 0;
    for(size_t i = 0; i < h_compacted_result.size(); ++i)
    {
      if(h_compacted_result[i] != h_validate_me[i])
      {
        std::cerr << "h_compacted_result[" << i << "]: " << h_compacted_result[i] << " d_compacted_result[" << i << "]: " << h_validate_me[i] << std::endl;
        ++num_errors;
      }

      if(num_errors == 10) break;
    }
    exit(-1);
  }

  // output a report
  std::cout << std::endl;

  // calculate performance
  // throughput = (sizeof compaction input + sizeof compaction output) / time
  float gpu_throughput = static_cast<float>(sizeof(unsigned int) * num_elements + num_compacted_elements_reference) / (gpu_time / 1000.0f);
  float cpu_throughput = static_cast<float>(sizeof(unsigned int) * num_elements + num_compacted_elements_reference) / (cpu_time / 1000.0f);

  std::cout << "Throughput of GPU compaction: " << (gpu_throughput / 1e9) << " GB/sec" << std::endl;
  std::cout << "Throughput of CPU compaction: " << (cpu_throughput / 1e9) << " GB/sec" << std::endl << std::endl;

  std::cout << "GPU Speedup: " << gpu_throughput / cpu_throughput << "x" << std::endl;

  deallocate_device_storage(d_input, d_compacted_result);

  return 0;
}

