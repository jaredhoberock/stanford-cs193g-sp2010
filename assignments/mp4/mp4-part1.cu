// In Part 1 of MP 4, we'll revisit our binning problem from previous MPs one last time
// by building it from parallel primitives using the Thrust library [1]. One way to
// approach the binning problem is by treating it as a sorting problem. In this
// approach, we first map each 3D point to the 1D index of the bin containing it.
// Next, we sort the points, treating their bin index as the sorting key. Finally, in
// order to count the number of points per bin, we perform a keyed reduction on this
// sorted array of bin IDs.
//
// To implement this MP, search this file for TODO and fill in the appropriate Thrust
// calls which implement each parallel primitive. Students familiar with C++ should
// find this MP trivial. The point is introduce you to the Thrust library in the hope
// that it proves useful in your class project. If you aren't familiar with C++
// templates, take a look at the Thrust Quick Start Guide [2] and try to copy from
// the idioms described therein. The header files which contain the Thrust calls you'll
// need have already been #included for you.
//
// [1] http://thrust.googlecode.com
// [2] http://code.google.com/p/thrust/wiki/QuickStartGuide

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/functional.h>
#include <cassert>
#include "mp4-util.h"
#include <iostream>

// TODO: functions in these headers will be useful for implementing a solution
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>


// hash a point in R^3 to its linear bin index, given the
// 3D grid resolution
__host__ __device__
unsigned int float3_to_linear_bin_index(const float3 &p,
                                        const unsigned int width,
                                        const unsigned int height,
                                        const unsigned int depth)
{
  // find the raster indices of p's bin
  const unsigned int x = p.x * width;
  const unsigned int y = p.y * height;
  const unsigned int z = p.z * depth;

  // return the bucket's linear index
  return (z * width * height) + (y * width) + x;
}


struct to_bin_index
  : thrust::unary_function<float3,unsigned int>
{
  to_bin_index(const unsigned int w,
               const unsigned int h,
               const unsigned int d)
    : width(w), height(h), depth(d)
  {}

  __host__ __device__
  unsigned int operator()(const float3 &x) const
  {
    unsigned int result = 0;
    // TODO: your computation here
    return result;
  }

  unsigned int width, height, depth;
};


int main(void)
{
  event_pair timer;

  const unsigned int width  = 256;
  const unsigned int height = 256;
  const unsigned int depth  = 256;

  const size_t num_bins = width * height * depth;

  const size_t n = 1<<22;

  // host storage
  thrust::host_vector<float3> h_points(n);

  // generate random points
  thrust::generate(h_points.begin(), h_points.end(), random_float3);

  // vectors initialize their elements to 0 when they are primitive types (like unsigned int)
  thrust::host_vector<unsigned int> h_num_points_per_bin(num_bins);

  // bin points on the host
  start_timer(&timer);
  for(thrust::host_vector<float3>::iterator x = h_points.begin();
      x != h_points.end();
      ++x)
  {
    // compute the bin index of the point
    const unsigned int idx = float3_to_linear_bin_index(*x, width, height, depth);

    // increment the count in this bin
    ++h_num_points_per_bin[idx];
  }

  float cpu_time = stop_timer(&timer, "CPU binning");

  // device storage for results
  thrust::device_vector<unsigned int> d_indices(n);
  thrust::device_vector<unsigned int> d_compacted_bin_indices(num_bins);
  thrust::device_vector<unsigned int> d_compacted_counts_per_bin(num_bins);
  thrust::device_vector<unsigned int> d_num_points_per_bin(num_bins);

  // copy points to the device
  thrust::device_vector<float3> d_points = h_points;

  start_timer(&timer);

  // TODO: your thrust calls here
  // hint: one way to bin points without atomics is to
  //       1. first map the points to their bin IDs
  //       2. sort the points by their IDs
  //       3. count the number of points per non-empty bin with a keyed reduction
  //       4. scatter the counts from the non-empty bins to the final d_num_points_per_bin array
  float gpu_time = stop_timer(&timer, "GPU binning");

  std::cout << n << " points" << std::endl;
  std::cout << "CPU time: " << cpu_time << " msecs" << std::endl;
  std::cout << "GPU time: " << gpu_time << " msecs" << std::endl;
  std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;

  assert(h_num_points_per_bin == d_num_points_per_bin);

  return 0;
}

