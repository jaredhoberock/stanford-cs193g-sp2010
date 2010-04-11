/* This is machine problem 2, binning
 * The problem is that you have particles in a 3D domain
 * which is quantized into blocks or bins. You want to figure
 * out which block each particle belongs to.
 * Use the atomic functions that you learned about in lecture 3
 * to implement the same functionality as the reference version on the cpu.
 *
 * FOR EXTRA CREDIT: 
 * Write a version of your binning kernel that uses atomics hierarchically, 
 * accumulating updates first into shared memory and then merging the results 
 * from shared memory into the global memory. 
 * As a hint, think about binning particles first into a coarse grid in a first kernel,
 * and then binning the particles from each coarse bin into the 
 * final bins in a second kernel.
 */



/*
 * SUBMISSION INSTRUCTIONS
 * =========================
 * 
 * You can submit your entire working directory for this assignment 
 * from any of the cluster machines by using our submit script. We want to be able
 * to just run "make" to compile your code.
 * The submit script bundles the entire current directory into
 * a submission. Thus, you use it by CDing to a the directory for your assignment,
 * and running:
 * 
 *   > cd *some directory*
 *   > /usr/class/cs193g/bin/submit mp2
 * 
 * This will submit the current directory as your assignment. You can submit
 * as many times as you want, and we will use your last submission.
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include <cuda.h>

#include "mp2-util.h"

// TODO enable this to print debugging information
//const bool print_debug = true;
const bool print_debug = false;

event_pair timer;

// the particle coordinates are already normalized (in the domain [0,1] )
// gridding provides the base 2 log of how finely the domain is subdivided
// in each direction. So gridding.x == 6 means that the x-axis is subdivided
// into 64 parts. (i.e. 2^(gridding.x) = number of bins on x axis)
// Overall there cannot be more than 4B bins, so we can just concatenate the bin
// indices into a single uint.

unsigned int bin_index(float3 particle, int3 gridding) 
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
    unsigned int bin = bin_index(particles[i],gridding);
    if(bin_counters[bin] < bin_size)
    {
      unsigned int offset = bin_counters[bin];
      // let's not do the whole precrement / postcrement thing...
      bin_counters[bin]++;
      bins[bin*bin_size + offset] = i;
    }
    else {
      *overflow_flag = true;
    }

  }
}

bool cross_check_results(int * h_bins, int * h_bins_checker, int * h_bin_counters, int * h_bin_counters_checker, int * h_particles_binids_checker, int num_particles, int num_bins, int bin_size)
{
  int error = 0;

  for(int i=0;i<num_bins;i++)
  {
    if(h_bin_counters[i] != h_bin_counters_checker[i])
    {

      if(print_debug) fprintf(stderr,"mismatch! bin %d: cuda:%d host:%d particles \n",i,h_bin_counters[i],h_bin_counters_checker[i]);
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
      if(h_bins_checker[i*bin_size+j] != -1)
      {
        if(h_particles_binids_checker[h_bins[i*bin_size+j]] != i)
        {
          error = 1;
        }
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

template
<typename T>
__global__ void initialize(T *array,T value, unsigned int array_length)
{
  int gid = blockIdx.x * blockDim.x  + threadIdx.x;
  
  if(gid < array_length)
  {
    array[gid] = value;
  }
}

void device_binning(float3 * h_particles, int * h_bins, int * h_bin_counters, int3 gridding, int num_particles, int num_bins, int bin_size)
{
  // TODO: your implementation here

	// How do I call a templated kernel? It's actually easy...
	// int* array;
	// int value = 0;
	// int array_length = 0;
	// initialize<<<griddim,blockdim>>>(array, value, array_length);
	// The compiler will figure out the types of your arguments and codegen a implementation for each type you use.

}

int main(void)
{  
  // create arrays of 8M elements
  int num_particles = 8*1024*1024;
  int log_bpd = 6;
  int bins_per_dim = 1 << log_bpd;
  unsigned int num_bins = bins_per_dim*bins_per_dim*bins_per_dim;
  // extra space to account for load imbalance to prevent frequent aborts due to bin overflow 
  int bin_size = num_particles/num_bins * 3;
  int3 gridding = make_int3(log_bpd,log_bpd,log_bpd);
  
  float3 *h_particles = 0;
  int *h_bins = 0;
  int *h_bin_counters = 0;
  int *h_bins_checker = 0;
  float3 *h_particles_checker = 0;
  int *h_bin_counters_checker = 0;
  int *h_particles_binids_checker = 0;
  int h_overflow_flag_checker = 0;

  // malloc host array
  h_particles = (float3*)malloc(num_particles * sizeof(float3));
  h_bins = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_bin_counters = (int*)malloc(num_bins * sizeof(int));
  h_particles_checker = (float3*)malloc(num_particles * sizeof(float3));
  h_bins_checker = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_particles_binids_checker = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_bin_counters_checker = (int*)malloc(num_bins * sizeof(int));

  // if either memory allocation failed, report an error message
  if(h_particles == 0 ||  
      h_bins == 0 || h_bin_counters == 0 ||  
      h_bins_checker == 0 || h_bin_counters_checker == 0 ||
      h_particles_binids_checker == 0)
  {
    printf("couldn't allocate memory\n");
    exit(1);
  }
  
  // generate random input
  // initialize
  srand(13);

  for(int i=0;i< num_particles;i++)
  {
    h_particles[i] = h_particles_checker[i] = make_float3((float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX,(float)rand()/(float)RAND_MAX);
  }
  for(int i=0;i<num_bins;i++)
  {
    h_bin_counters[i] = h_bin_counters_checker[i] = 0;
  }
  for(int i=0;i<num_bins*bin_size;i++)
  {
    h_bins[i] = h_bins_checker[i] = h_particles_binids_checker[i] = -1;
  }

  device_binning(h_particles, h_bins, h_bin_counters, gridding, num_particles, num_bins, bin_size);
  
  // generate reference output
  start_timer(&timer);
  host_binning(h_particles_checker, h_bins_checker, h_bin_counters_checker, &h_overflow_flag_checker, gridding, bin_size, num_particles);
  stop_timer(&timer,"cpu binning");
  
  if(h_overflow_flag_checker)
  {
    printf("one of the bins overflowed!\n");
    exit(1);
  }

  // check CUDA output versus reference output
  cross_check_results(h_bins, h_bins_checker, h_bin_counters, h_bin_counters_checker, h_particles_binids_checker, num_particles, num_bins, bin_size);

  // deallocate memory
  free(h_particles);
  free(h_bins);
  free(h_bin_counters);
  free(h_particles_checker);
  free(h_bins_checker);
  free(h_particles_binids_checker);
  free(h_bin_counters_checker);
 
  return 0;
}

