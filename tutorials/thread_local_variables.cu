// This example demonstrates the use of thread-local variables to
// implement a k-Nearest Neighbor algorithm. Each CUDA thread will
// read a unique reference point, p, into a per-thread local register
// variable while maintaining a priority queue, or heap, of the ten
// nearest points to p in a per-thread local array variable.

#include <util/stl_heap.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

// a function which computes the Euclidean distance between two 2D points
__device__ float distance(const float2 x, const float2 y)
{
  float dx = x.x - y.x;
  float dy = x.y - y.y;

  return sqrtf(dx*dx + dy*dy);
}


// a function object which compares two 2D points distance to a reference,
// and returns true if the left-hand-side point is closer than the right-hand-side
// \see http://en.wikipedia.org/wiki/Function_object
struct compare_distance_to_p
{
  __device__
  compare_distance_to_p(const float2 _p)
    : p(_p){}

  __device__
  float operator()(const float2 x, const float2 y) const
  {
    return distance(x,p) < distance(y,p);
  }

  float2 p;
};


// this kernel computes, for every point p in point set P, the ten nearest points q in Q
__global__ void ten_nn(float2 *result,
                       const float2 *ps,
                       const float2 *qs,
                       const size_t num_qs)
{
  // p goes in a register
  float2 p = ps[threadIdx.x];

  // per-thread heap goes in off-chip memory
  float2 heap[10];

  // read through num_qs_per_p points, maintaining
  // the nearest 10 qs to p in the heap

  // first fill up the local array
  // we assume num_qs >= 10
  int i = 0;
  for(; i < 10; ++i)
  {
    heap[i] = qs[i];
  }

  // create a function object which will measure the distance
  // between points in the heap and p
  // this serves as our heap priority criterion: points nearer to
  // p have higher priority than points farther from p
  compare_distance_to_p priority_criterion(p);

  // heapify the array
  make_heap(heap, heap + 10, priority_criterion);

  // maintain the heap as we exhaust the input
  for(; i < num_qs; ++i)
  {
    // get the current farthest q from p
    float2 farthest_q = heap[0];

    // read the next q
    float2 q = qs[i];

    // is q higher priority than farthest_q?
    if(priority_criterion(q,farthest_q))
    {
      // q is closer, so pop the top of the heap
      pop_heap(heap, heap + 10, priority_criterion);

      // push q on the heap
      heap[9] = q;
      push_heap(heap, heap + 10, priority_criterion);
    }
  }

  // the ten points closest to p are now stored in heap
  // write out to result
  float2 *result_begin = result + 10 * threadIdx.x;
  for(i = 0; i < 10; ++i)
  {
    // XXX these writes to global memory don't coalesce
    result_begin[i] = heap[i];
  }
}

// this function returns a random point in the unit square
float2 random_float2(void)
{
  return make_float2(static_cast<float>(rand()) / RAND_MAX,
                     static_cast<float>(rand()) / RAND_MAX);
}


int main(void)
{
  const size_t num_ps = 512;
  const size_t num_qs = 100 * num_ps;

  // allocate host storage for our P and Q point sets
  std::vector<float2> h_ps(num_ps);
  std::vector<float2> h_qs(num_qs);

  // generate random point sets
  for(int i = 0; i < h_ps.size(); ++i)
    h_ps[i] = random_float2();

  for(int i = 0; i < h_qs.size(); ++i)
    h_qs[i] = random_float2();

  // copy the point sets to the device
  float2 *d_ps = 0, *d_qs = 0;
  cudaMalloc((void**)&d_ps, sizeof(float2) * num_ps);
  cudaMalloc((void**)&d_qs, sizeof(float2) * num_qs);

  cudaMemcpy(d_ps, &h_ps[0], sizeof(float2) * num_ps, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qs, &h_qs[0], sizeof(float2) * num_qs, cudaMemcpyHostToDevice);

  // allocate storage on the device for 10 nearest neighbors for every point in P
  float2 *d_result = 0;
  cudaMalloc((void**)&d_result, sizeof(float2) * 10 * num_ps);

  // just launch one block to keep the example simple
  ten_nn<<<1,num_ps>>>(d_result, d_ps, d_qs, num_qs);

  // copy the result back to the host
  std::vector<float2> h_result(10 * num_ps);
  cudaMemcpy(&h_result[0], d_result, sizeof(float2) * 10 * num_ps, cudaMemcpyDeviceToHost);

  // write out the 10 nearest neighbors in Q for the first 10 points in P
  for(int i = 0; i < 10; ++i)
  {
    std::cout << "points closest to (" << h_ps[i].x << " " << h_ps[i].y << "): ";

    for(int j = 0; j < 10; ++j)
    {
      std::cout << "(" << h_result[10*i + j].x << " " << h_result[10*i + j].y << ") ";
    }
    std::cout << std::endl << std::endl;
  }

  // deallocate device storage
  cudaFree(d_ps);
  cudaFree(d_qs);
  cudaFree(d_result);

  return 0;
}

