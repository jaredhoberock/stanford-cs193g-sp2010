// Your parallel prefix sum implementation goes here.
// We've given you some skeleton code which you can fill in.
// The skeleton is just a suggestion -- programming ninjas
// feel free to do your own thing, but the interface in
// scan.h must be preserved.

#include "scan.h"
#include "mp3-util.h"
#include <algorithm>
#include <iostream>

// assuming blockDim.x = 4,
// then given n = 10 and
// segment_size = 4 and
// d_data = [ 1 0 1 1 | 1 1 0 0 | 1 0 ]
// this kernel yields
// d_data = [ 1 1 2 3 | 1 2 2 2 | 1 1 ] and
// d_per_block_sums = [ 3 2 1 ]

// assuming blockDim.x = 4,
// then given n = 10 and
// segment_size = 8 and
// d_data  = [ 1 0 1 1 1 1 0 0 | 1 0 ]
// this kernel yields
// d_data  = [ 1 1 2 3 4 5 5 5 | 1 1 ] and
// d_per_block_sums = [ 5 1 ]

// if d_per_block_sums == 0, then the kernel shouldn't output
// the per-block sums

__global__ void inplace_inclusive_scan_kernel(unsigned int *d_data,
                                              const size_t n,
                                              const size_t segment_size,
                                              unsigned int *d_per_block_sums)
{
  // implementation hints:
  // dynamically allocate a __shared__ array, with one word per thread
  // do a block-wide prefix sum over the __shared__ array
  // output each thread's prefix sum result to d_output
  // thread 0 outputs each block's sum to d_per_block_sums

  // TODO: your kernel implementation here
}


// assuming blockDim.x = 4,
// then given n = 10 and
// segment_size = 4 and
// d_per_block_carry_in = [ 3 5 6 ] and
// d_result = [ 1 1 2 3 | 1 2 2 2 | 1 1 ]
// this kernel yields
// d_result = [ 0 1 1 2 | 3 4 5 5 | 5 6 ]

// assuming blockDim.x = 4,
// then given n = 10 and
// segment_size = 8 and
// d_per_block_carry_in = [ 5 6 ] and
// d_result = [ 1 1 2 3 4 5 5 5 | 1 1 ]
// this kernel yields
// d_result = [ 0 1 1 2 3 4 5 5 | 5 6 ]
__global__ void exclusive_scan_update(const unsigned int *d_per_block_carry_in,
                                      const size_t n,
                                      const size_t segment_size,
                                      unsigned int *d_result)
{
  // implementation hint:
  // to turn an inclusive scan result into an exclusive scan,
  // each thread needs the result of its previous neighbor.
  // consider using a __shared__ memory strategy similar to the
  // adjacent_difference example from lecture 4

  // TODO: your kernel implementation here
}



void inplace_exclusive_scan(unsigned int *d_data,
                            const size_t n)
{
  // one way to implement exclusive scan is with three kernel launches:
  // 1. partition the input into equally-sized segments, each which will be handled
  //    by a single thread block. the segment length may be larger than the block_size.
  //    (you can experimentally determine the best segment length, or just let segment_length = block_size to begin with)
  //    each thread block loads a subset of the segment at a time into a __shared__ array, doing a block-wide
  //    prefix sum across it.  
  //    block-wide inclusive scan across the shared array.  At the same time, output a per-block sum.
  //    The idea of step 1. is to make the size of the per-block array as small as possible in order to minimize
  //    malloc & bandwidth requirements.
  //    hint: choose a maximum number of blocks to launch (say, the number of SMs on the GPU), and then hold the
  //    block_size constant.  Now solve for segment_size.
  // 2. inclusive scan the array of per-block sums in place
  //    use the same kernel as before, but launch a single thread block
  //    you don't need to produce per-block sums this time 
  // 3. transform the local inclusive scan results plus the per-segment sums into a global exclusive scan result
  //    with a final kernel
  // This will be relatively* simple code resulting in a moderately fast scan.
  // *relative to state of the art implementations

  // TODO: your implementation here
  // hint: when debugging, call the check_cuda_error function
  // in mp3-util.h after every kernel launch

}

