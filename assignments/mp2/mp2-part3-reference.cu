#include "mp2.h"

// TODO enable this to print debugging information
//const bool print_debug = true;
const bool print_debug = false;
  
unsigned int host_bin_index(float3 particle, int3 gridding) 
{
  unsigned int x_index = (unsigned int)(particle.x * (1 << gridding.x));
  unsigned int y_index = (unsigned int)(particle.y * (1 << gridding.y));
  unsigned int z_index = (unsigned int)(particle.z * (1 << gridding.z));
  unsigned int index = 0;
  index |= z_index;
  index <<= gridding.y;
  index |= y_index;
  index <<= gridding.x;
  index |=  x_index;
  
  return index;
}

void host_binning(float3 *particles, int *bins, int *bin_counters, int *overflow_flag, int3 gridding, int bin_size, int array_length)
{
  for(int i=0;i<array_length;i++)
  {
    unsigned int bin = host_bin_index(particles[i],gridding);
    if(bin_counters[bin] < bin_size)
    {
      unsigned int offset = bin_counters[bin];
      bin_counters[bin]++;
      bins[bin*bin_size + offset] = i;
    }
    else {
      *overflow_flag = true;
    }
    
  }
}

bool cross_check_results(int num_particles, int num_bins, int bin_size, int num_neighbors,
                         int *h_bin_counters, int *h_bin_counters_checker,
                         int *h_particles_binids_checker, int *h_bins_checker,
                         int *h_bins, int *h_knn, int *h_knn_checker)
{
  int error = 0;
  
  for(int i=0;i<num_bins;i++)
  {
  if(h_bin_counters[i] != h_bin_counters_checker[i])
  {
    if(print_debug) printf("mismatch! bin %d: cuda:%d host:%d particles \n",i,h_bin_counters[i],h_bin_counters_checker[i]);
    error = 1;
  }
  for(int j=0; j<bin_size;j++)
  {
    // record which these particles went into bin i in the reference version
    if(h_bins_checker[i*bin_size+j] != -1)
    {
      h_particles_binids_checker[h_bins_checker[i*bin_size+j]] = i;
    }
  }
  for(int j=0; j<bin_size;j++)
  {
    if(h_bins[i*bin_size+j] != -1)
    {
      if(h_particles_binids_checker[h_bins[i*bin_size+j]] != i)
      {
        error = 1;
      }
    }
  }
  }
  for(int i=0;i<num_particles;i++)
  {
    for(int j=0;j<num_neighbors;j++)
    {
      if(h_knn[i*num_neighbors + j] != h_knn_checker[i*num_neighbors + j])
      {   
        error = 1;
      }
    }
  }
  
  if(error)
  {
  printf("Output of CUDA version and normal version didn't match! \n");
  }
  else {
  printf("Worked! CUDA and reference output match. \n");
  }
  return error;
}
