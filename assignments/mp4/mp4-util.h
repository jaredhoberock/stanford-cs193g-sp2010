#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifdef MP4_USE_DOUBLE_PRECISION
typedef double real;
#else
typedef float real;
#endif

#include <thrust/random.h>


float random_float(void)
{
  static thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01;

  return u01(rng);
}

float3 random_float3(void)
{
  return make_float3(random_float(), random_float(), random_float());
}

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


template<typename Option>
inline bool fuzzy_validate(const thrust::device_vector<Option> &x,
                           const thrust::host_vector<Option> &reference,
                           const bool verbose = false)
{
  if(x.size() != reference.size()) return false;

  thrust::host_vector<Option> temp = x;

  size_t num_call_mismatches = 0;
  size_t num_put_mismatches  = 0;

  real tolerance = 0;
  if(sizeof(real) == 4)
  {
    tolerance = 1e-3;
  }
  else
  {
    tolerance = 1e-6;
  }

  size_t max_num_mismatches_to_tolerate = 10.0 * tolerance * static_cast<real>(x.size());

  for(size_t i = 0; i < temp.size(); ++i)
  {
    Option tmp = temp[i];
    Option ref = reference[i];

    if(!almost_equal_relative(tmp.call,ref.call, tolerance)) 
    {
      ++num_call_mismatches;

      if(verbose && num_call_mismatches < 10)
      {
        printf("fuzzy_validate(): call mismatch: x[%d]: %0.30f, reference[%d]: %0.30f, relative error: %.9f\n", (int)i, tmp.call, (int)i, ref.call, relative_error(tmp.call,ref.call));
      }
    }

    if(!almost_equal_relative(tmp.put,ref.put, tolerance)) 
    {
      ++num_put_mismatches;

      if(verbose && num_put_mismatches < 10)
      {
        printf("fuzzy_validate(): put mismatch: x[%d]: %0.30f, reference[%d]: %0.30f, relative error: %.9f\n", (int)i, tmp.put, (int)i, ref.put, relative_error(tmp.put,ref.put));
      }
    }
  }

  return (num_call_mismatches <= max_num_mismatches_to_tolerate) && (num_put_mismatches <= max_num_mismatches_to_tolerate);
}


