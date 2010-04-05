// This example demonstrates the use of shared per-block arrays
// implement an optimized dense matrix multiplication algorithm.
// Like the shared_variables.cu example, a per-block __shared__
// array acts as a "bandwidth multiplier" by eliminating redundant
// loads issued by neighboring threads.

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>

#define TILE_WIDTH 16

// compute the number of lines of code each implementation requires
const unsigned int simple_implementation_begin = __LINE__;

// a simple version of matrix_multiply which issues redundant loads from off-chip global memory
__global__ void matrix_multiply_simple(float *a, float *b, float *ab, size_t width)
{
  // calculate the row & column index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  float result = 0;

  // do dot product between row of a and column of b
  for(int k = 0; k < width; ++k)
  {
    result += a[row*width+k] * b[k*width+col];
  }

  // write out this thread's result
  ab[row*width+col] = result;
}
const unsigned int simple_implementation_size = __LINE__ - simple_implementation_begin;


const unsigned int optimized_implementation_begin = __LINE__;

// an optimized version of matrix_multiplication which eliminates redundant loads
__global__ void matrix_multiply(float *a, float *b, float *ab, size_t width)
{
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x,  by = blockIdx.y;

  // allocate 2D tiles in __shared__ memory
  __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

  // calculate the row & column index of the element
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  float result = 0;

  // loop over the tiles of the input in phases
  for(int p = 0; p < width/TILE_WIDTH; ++p)
  {
    // collaboratively load tiles into __shared__
    s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
    s_b[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];

    // wait until all data is loaded before allowing
    // any thread in this block to continue
    __syncthreads();

    // do dot product between row of s_a and column of s_b
    for(int k = 0; k < TILE_WIDTH; ++k)
    {
      result += s_a[ty][k] * s_b[k][tx];
    }

    // wait until all threads are finished with the data
    // before allowing any thread in this block to continue
    __syncthreads();
  }

  // write out this thread's result
  ab[row*width+col] = result;
}
const unsigned int optimized_implementation_size = __LINE__ - optimized_implementation_begin;


int main(void)
{
  // create a large workload so we can easily measure the
  // performance difference of both implementations

  // note that n measures the width of the matrix, not the number of total elements
  const size_t n = 1<<10;
  const dim3 block_size(TILE_WIDTH,TILE_WIDTH);
  const dim3 num_blocks(n / block_size.x, n / block_size.y);

  // generate random input on the host
  std::vector<float> h_a(n*n), h_b(n*n), h_c(n*n);
  for(int i = 0; i < n*n; ++i)
  {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // allocate storage for the device
  float *d_a = 0, *d_b = 0, *d_c = 0;
  cudaMalloc((void**)&d_a, sizeof(float) * n * n);
  cudaMalloc((void**)&d_b, sizeof(float) * n * n);
  cudaMalloc((void**)&d_c, sizeof(float) * n * n);

  // copy input to the device
  cudaMemcpy(d_a, &h_a[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);

  // time the kernel launches using CUDA events
  cudaEvent_t launch_begin, launch_end;
  cudaEventCreate(&launch_begin);
  cudaEventCreate(&launch_end);

  // to get accurate timings, launch a single "warm-up" kernel
  matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);

  // time many kernel launches and take the average time
  const size_t num_launches = 100;
  float average_simple_time = 0;
  std::cout << "Timing simple implementation...";
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply_simple<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_simple_time += time;
  }
  average_simple_time /= num_launches;
  std::cout << " done." << std::endl;

  // now time the optimized kernel

  // again, launch a single "warm-up" kernel
  matrix_multiply<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);

  // time many kernel launches and take the average time
  float average_optimized_time = 0;
  std::cout << "Timing optimized implementation...";
  for(int i = 0; i < num_launches; ++i)
  {
    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);
    matrix_multiply<<<num_blocks,block_size>>>(d_a, d_b, d_c, n);
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    // measure the time spent in the kernel
    float time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    average_optimized_time += time;
  }
  average_optimized_time /= num_launches;
  std::cout << " done." << std::endl;

  // report the effective throughput of each kernel in GFLOPS
  // the effective throughput is measured as the number of floating point operations performed per second:
  // (one mul + one add) * N^3
  float simple_throughput = static_cast<float>(2 * n * n * n) / (average_simple_time / 1000.0f) / 1000000000.0f;
  float optimized_throughput = static_cast<float>(2 * n * n * n) / (average_optimized_time / 1000.0f) / 1000000000.0f;

  // compute throughput per line of code to measure how productive we were
  float simple_throughput_per_sloc = simple_throughput / simple_implementation_size;
  float optimized_throughput_per_sloc = optimized_throughput / optimized_implementation_size;

  std::cout << "Matrix size: " << n << "x" << n << std::endl;
  std::cout << "Tile size: " << TILE_WIDTH << "x" << TILE_WIDTH << std::endl;
  std::cout << "Simple implementation SLOCs: " << simple_implementation_size << std::endl;
  std::cout << "Optimized implementation SLOCs: " << optimized_implementation_size << std::endl << std::endl;

  std::cout << "Throughput of simple kernel: " << simple_throughput << " GFLOPS" << std::endl;
  std::cout << "Throughput of optimized kernel: " << optimized_throughput << " GFLOPS" << std::endl;
  std::cout << "Performance improvement: " << optimized_throughput / simple_throughput << "x" << std::endl;
  std::cout << std::endl;

  std::cout << "Throughput of simple kernel per line of code: " << simple_throughput_per_sloc << " FLOPS/sloc" << std::endl;
  std::cout << "Throughput of optimized kernel per line of code: " << optimized_throughput_per_sloc << " FLOPS/sloc" << std::endl;
  std::cout << "Performance improvement per line of code: " << optimized_throughput_per_sloc / simple_throughput_per_sloc << "x" << std::endl;

  // destroy the CUDA events
  cudaEventDestroy(launch_begin);
  cudaEventDestroy(launch_end);

  // deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

