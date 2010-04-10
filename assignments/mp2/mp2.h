#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctime>

#include <cuda.h>

// pointers to host arrays
extern float3 *h_particles;
extern int *h_bins;
extern int *h_knn;
extern int *h_bin_counters;
extern int *h_bins_checker;
extern int *h_knn_checker;
extern float3 *h_particles_checker;
extern int *h_bin_counters_checker;
extern int *h_particles_binids_checker;

inline __device__ __host__ bool operator !=(float3 a, float3 b)
{
    return (a.x!=b.x) || (a.y!=b.y) ||(a.y!=b.y);
}
inline __device__ __host__ float3 operator +(float3 a, float3 b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
inline __device__ __host__ float3 operator -(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
inline __device__ __host__ float3 operator *(float3 a, float3 b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
inline __device__ __host__ float3 operator *(float3 a, float b)
{
    return make_float3(a.x*b, a.y*b, a.z*b);
}
inline __device__ __host__ float3 operator *(float a, float3 b)
{
    return make_float3(b.x*a, b.y*a, b.z*a);
}

__host__ __device__ 
float dist2(float3 a, float3 b);

template
<typename T>
__host__ __device__
void init_list(T *base_ptr, unsigned int size, T val);

__host__ __device__
void insert_list(float *dist_list, int *id_list, int size, float dist, int id);

void host_binning(float3 *particles, int *bins, int *bin_counters, int *overflow_flag, int3 gridding, int bin_size, int array_length);

template <int num_neighbors> void host_binned_knn(float3 *particles, int *bins, int *knn, int3 binning_dim, int bin_size);

bool cross_check_results(int num_particles, int num_bins, int bin_size, int num_neighbors,
                         int *h_bin_counters, int *h_bin_counters_checker,
                         int *h_particles_binids_checker, int *h_bins_checker,
                         int *h_bins, int *h_knn, int *h_knn_checker);

