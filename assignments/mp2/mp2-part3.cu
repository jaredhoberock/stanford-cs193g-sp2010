/* This is machine problem 2, part 3, k nearest neighbor with binning
 *
 * In part 3 we'll combine the things you did in parts 1 and 2.
 * The basic problem is the same as in part 2 (find the k nearest
 * neighbors of each particle), but you're going to use a algorithmically
 * smarter approach. Instead of comparing each particle against every 
 * other particle, you will first bin the particles into bins of a given
 * size, and then only compare each particle against the particles 
 * which are in its own bin and all the 26 neighboring bins.
 * Use what you learned in part 1 for the binning step, and what
 * you learned in part 2 for finding the k nearest neighbors.
 * Your code should run with both small bins (8-16 particles/bin) and 
 * large bins (more than 512 particles per bin) so you shouldn't have
 * any dependency between your threadblock size and the bin size. 
 * We'll be testing the speed of your code with different bin sizes (from 16 to 1024), 
 * so you should make sure your code runs efficiently at all sizes.
 *
 */
 
/*
 * SUBMISSION INSTRUCTIONS
 * =========================
 * 
 * You can submit the assignment from any of the cluster machines by using
 * our submit script. Th submit script bundles the entire current directory into
 * a submission. Thus, you use it by CDing to a the directory for your assignment,
 * and running:
 * 
 *   > cd *some directory*
 *   > /usr/class/cs193g/bin/submit mp2
 * 
 * This will submit the current directory as your assignment. You can submit
 * as many times as you want, and we will use your last submission.
 */

#include "mp2.h"
#include "mp2-util.h"

event_pair timer;

// the particle coordinates are already normalized (in the domain [0,1[ )
// gridding provides the base 2 log of how finely the domain is subdivided
// in each direction. So gridding.x == 6 means that the x-axis is subdivided
// into 64 parts.
// Overall there cannot be more than 4B bins, so we can just concatenate the bin
// indices into a single uint.

__host__ __device__ 
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

__host__ __device__ 
float dist2(float3 a, float3 b)
{
  float3 d = a - b;
  float d2 = d.x*d.x + d.y*d.y + d.z*d.z;
  return d2;
}

template
<typename T>
__host__ __device__
void init_list(T *base_ptr, unsigned int size, T val)
{
  for(int i=0;i<size;i++)
  {
    base_ptr[i] = val;
  }
}

__host__ __device__
void insert_list(float *dist_list, int *id_list, int size, float dist, int id)
{
 int k;
 for (k=0; k < size; k++) {
   if (dist < dist_list[k]) {
     // we should insert it in here, so push back and make it happen
     for (int j = size - 1; j > k ; j--) {
       dist_list[j] = dist_list[j-1];
       id_list[j] = id_list[j-1];
     }
     dist_list[k] = dist;
     id_list[k] = id;
     break;
   }
 }
}

void allocate_host_memory(int num_particles, int num_bins, int bin_size, int num_neighbors,
                          float3 *&h_particles, int *&h_bins, int *&h_knn, int *&h_bin_counters,
                          float3 *&h_particles_checker, int *&h_bins_checker, int *&h_knn_checker,
                          int *&h_particles_binids_checker, int *&h_bin_counters_checker)
{
  // malloc host array
  h_particles = (float3*)malloc(num_particles * sizeof(float3));
  h_bins = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_knn = (int*)malloc(num_particles * num_neighbors * sizeof(int));
  h_bin_counters = (int*)malloc(num_bins * sizeof(int));
  h_particles_checker = (float3*)malloc(num_particles * sizeof(float3));
  h_bins_checker = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_knn_checker = (int*)malloc(num_particles * num_neighbors * sizeof(int));
  h_particles_binids_checker = (int*)malloc(num_bins * bin_size * sizeof(int));
  h_bin_counters_checker = (int*)malloc(num_bins * sizeof(int));
  
  // if any memory allocation failed, report an error message
  if(h_particles == 0 || 
   h_bins == 0 || h_knn == 0 || h_bin_counters == 0 ||  
   h_bins_checker == 0 || h_knn_checker == 0 || h_bin_counters_checker == 0 ||
   h_particles_binids_checker == 0)
  {
    printf("couldn't allocate host memory\n");
    exit(1);
  }
}


void allocate_device_memory(int num_particles, int num_bins, int bin_size, int num_neighbors,
                            float3 *&d_particles, int *&d_bins,
                            int *&d_knn, int *&d_bin_counters, int *&d_overflow_flag)
{
  // TODO: your device memory allocations here
  // TODO: don't forget to check for errors
}


void deallocate_host_memory(float3 *h_particles, int *h_bins, int *h_knn, int *h_bin_counters,
                            float3 *h_particles_checker, int *h_bins_checker, int *h_knn_checker,
                            int *h_particles_binids_checker, int *h_bin_counters_checker)
{
  // deallocate memory
  free(h_particles);
  free(h_bins);
  free(h_knn);
  free(h_bin_counters);
  free(h_particles_checker);
  free(h_bins_checker);
  free(h_knn_checker);
  free(h_particles_binids_checker);
  free(h_bin_counters_checker);
}


void deallocate_device_memory(float3 *d_particles, int *d_bins,
                              int *d_knn, int *d_bin_counters, int *d_overflow_flag)
{
  // TODO: your device memory deallocations here
  // TODO: don't forget to check for errors
}


template
<int num_neighbors>
void host_knn_particle(float3 *particles, int *bins, int *part_knn, int id, int bin_id, int bx, int by, int bz, int3 binning_dim, int bin_size)
{
  // for each particle
  // loop over all the neighbor bins in x,y and z,
  // as well as the bin it is in itself
  
  float neigh_dist[num_neighbors];
  int neigh_ids[num_neighbors];
    
  init_list(&neigh_dist[0],num_neighbors,2.0f);
  init_list(&neigh_ids[0],num_neighbors,-1);
  
  float3 pos = particles[id];

  for(int x_offset=-1;x_offset<2;x_offset++)
  {
    int nx = bx + x_offset;
    if(nx > -1 && nx < binning_dim.x)
    {
      for(int y_offset=-1;y_offset<2;y_offset++)
      {
        int ny = by + y_offset;
        if(ny > -1 && ny < binning_dim.y)
        {
          for(int z_offset=-1;z_offset<2;z_offset++)
          {
            int nz = bz + z_offset;
            if(nz > -1 && nz < binning_dim.z)
            {
              int neigh_bin_id = nx + binning_dim.x*(ny + binning_dim.y*nz);
              
              // loop over all the particles in those bins
              for(int bin_offset=0;bin_offset<bin_size;bin_offset++)
              {
                int neigh_particle_id = bins[neigh_bin_id*bin_size + bin_offset];
                // skip empty bin entries and don't interact with yourself
                if(neigh_particle_id != -1 && neigh_particle_id != id)
                {
                  float rsq = dist2(pos,particles[neigh_particle_id]);
                  insert_list(&neigh_dist[0], &neigh_ids[0], num_neighbors, rsq, neigh_particle_id);
                }
              }
            }
          }
        }
      }
    }
  }
  for(int j=0;j<num_neighbors;j++)
  { 
    part_knn[j] = neigh_ids[j];
  }
}

template
<int num_neighbors>
void host_binned_knn(float3 *particles, int *bins, int *knn, int3 binning_dim, int bin_size)
{
  // loop over all bins
  for(int bx=0;bx<binning_dim.x;bx++)
  {
    for(int by=0;by<binning_dim.y;by++)
    {
      for(int bz=0;bz<binning_dim.z;bz++)
      {
        // now that we have a bin, loop over all
        // particles in the bin
        // skipping those bin entries which are empty
        int bin_id = bx + binning_dim.x*(by + binning_dim.y*bz);
        for(int j=0;j<bin_size;j++)
        {
          int id = bins[bin_id*bin_size + j];
          if(id != -1)
          {
            host_knn_particle<num_neighbors>(particles, bins, &knn[id*num_neighbors],id, bin_id, bx, by, bz, binning_dim, bin_size);
          }
        }
      }
    }
  }
}

int main(void)
{  
  // create arrays of 512K elements
  int num_particles = 512*1024;
  int log_bpd = 4;
  int bins_per_dim = 1 << log_bpd;
  const int num_neighbors = 5;
  unsigned int num_bins = bins_per_dim*bins_per_dim*bins_per_dim;
  // some extra space to account for load imbalance
  int bin_size = num_particles/num_bins * 2;
  int3 gridding = make_int3(log_bpd,log_bpd,log_bpd);
  int3 binning_dim = make_int3(bins_per_dim,bins_per_dim,bins_per_dim);

  // pointers to host arrays
  int h_overflow_flag = 0;
  float3 *h_particles = 0;
  int *h_bins = 0;
  int *h_knn = 0;
  int *h_bin_counters = 0;
  float3 *h_particles_checker = 0;
  int *h_bins_checker = 0;
  int *h_knn_checker = 0;
  int *h_particles_binids_checker = 0;
  int *h_bin_counters_checker = 0;
  
  allocate_host_memory(num_particles, num_bins, bin_size, num_neighbors,
                       h_particles, h_bins, h_knn, h_bin_counters, h_particles_checker,
                       h_bins_checker, h_knn_checker, h_particles_binids_checker, h_bin_counters_checker);

  // pointers to device memory
  float3 *d_particles = 0;
  int *d_bins = 0;
  int *d_knn = 0;
  int *d_bin_counters = 0;
  int *d_overflow_flag = 0;

  allocate_device_memory(num_particles, num_bins, bin_size, num_neighbors,
                         d_particles, d_bins,
                         d_knn, d_bin_counters, d_overflow_flag);

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
  for(int i=0;i<num_particles*num_neighbors;i++)
  {
    h_knn[i] = h_knn_checker[i] = -1;
  }
  
  // copy input to GPU
  start_timer(&timer);
  // TODO: your copy of input from host to device here
  stop_timer(&timer,"copy to gpu");
  
  start_timer(&timer);
  // TODO: your particle binning kernel launch here
  check_cuda_error("binning");
  stop_timer(&timer,"binning");
  
  // TODO: your check for overflow here
  check_cuda_error("flag copy");
  if(h_overflow_flag)
  {
    printf("one of the bins overflowed!\n");
    exit(1);
  }
    
  start_timer(&timer);  
  // TODO: your binned k-nearest neighbor kernel launch here
  check_cuda_error("binned knn");
  stop_timer(&timer,"binned knn");
  
  // download and inspect the result on the host
  start_timer(&timer);
  // TODO: your copy of results from device to host here
  check_cuda_error("copy from gpu");
  stop_timer(&timer,"copy back from gpu");

  // generate reference output
  start_timer(&timer);
  host_binning(h_particles_checker, h_bins_checker, h_bin_counters_checker, &h_overflow_flag, gridding, bin_size, num_particles);
  stop_timer(&timer,"cpu binning");
  start_timer(&timer);
  host_binned_knn<num_neighbors>(h_particles_checker, h_bins_checker, h_knn_checker, binning_dim, bin_size);
  stop_timer(&timer,"cpu binning");
  
  // check CUDA output versus reference output
  cross_check_results(num_particles, num_bins, bin_size, num_neighbors,
                      h_bin_counters, h_bin_counters_checker, h_particles_binids_checker,
                      h_bins_checker, h_bins, h_knn, h_knn_checker);
  
  deallocate_host_memory(h_particles, h_bins, h_knn, h_bin_counters, h_particles_checker,
                         h_bins_checker, h_knn_checker, h_particles_binids_checker, h_bin_counters_checker);

  deallocate_device_memory(d_particles, d_bins,
                           d_knn, d_bin_counters, d_overflow_flag);
  
  return 0;
}

