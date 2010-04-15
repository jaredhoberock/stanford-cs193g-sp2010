#pragma once

#include <stdio.h>

#ifdef MP3_USE_DOUBLE_PRECISION
typedef double real;
#else
typedef float real;
#endif

inline void check_cuda_error(const char *message, const char *filename, const int lineno)
{
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error after %s at %s:%d: %s\n", message, filename, lineno, cudaGetErrorString(error));
    exit(-1);
  }
}


inline real random_real(real low, real high)
{
  real t = (real)rand() / (real)RAND_MAX;
  return (1.0 - t) * low + t * high;
}

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};


inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}


inline float stop_timer(event_pair * p, char *name)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  printf("%s took %.1f ms\n",name, elapsed_time);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}


template<typename RealType>
inline RealType relative_error(const RealType a, const RealType b)
{
  return fabs((a-b)/b);
}


template<typename RealType>
inline bool almost_equal_relative(const RealType a, const RealType b, const RealType max_relative_error)
{
  if(a == b) return true;

  if(relative_error(a,b) <= max_relative_error)
    return true;

  return false;
}


inline bool fuzzy_validate(const real *x, const real *reference, const size_t n, const bool verbose = false)
{
  size_t num_mismatches = 0;

  real tolerance = 0;
  if(sizeof(real) == 4)
  {
    tolerance = 1e-3;
  }
  else
  {
    tolerance = 1e-6;
  }

  size_t max_num_mismatches_to_tolerate = 10.0 * tolerance * static_cast<real>(n);

  for(size_t i = 0; i < n; ++i)
  {
    if(!almost_equal_relative(x[i],reference[i], tolerance)) 
    {
      ++num_mismatches;

      if(verbose && num_mismatches < 10)
      {
        printf("validate(): x[%d]: %0.30f, reference[%d]: %0.30f, relative error: %.9f\n", (int)i, x[i], (int)i, reference[i], relative_error(x[i],reference[i]));
      }
    }
  }

  return num_mismatches <= max_num_mismatches_to_tolerate;
}


inline __host__ __device__
bool is_even(const unsigned int x)
{
  return !(x & 1);
}


