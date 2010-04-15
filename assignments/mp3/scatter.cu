#include "scatter.h"
#include "mp3-util.h"

// scatter is simple: for an input list d_input and indices d_scatter_indices,
// copy the even-valued elements of d_input to their new locations in
// d_result given by their indices in d_scatter_indices
// for example, given
// d_input           = [ 0 1 2 4 10 2 3 5 0 3 ] and
// d_scatter_indices = [ 0 1 1 2  3 4 5 5 5 6 ] and
// n = 10,
// this kernel yields
// d_result          = [ 0 2 4 10 2 0 <undefined...> ]
__global__ void scatter_even_elements_kernel(const unsigned int *d_input,
                                             const unsigned int *d_scatter_indices,
                                             const size_t n,
                                             unsigned int *d_result)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int grid_size = gridDim.x * blockDim.x;

  for(; i < n; i += grid_size)
  {
    const unsigned int x = d_input[i];

    if(is_even(x))
    {
      d_result[d_scatter_indices[i]] = x;
    }
  }
}

void scatter_even_elements(const unsigned int *d_input,
                           const unsigned int *d_scatter_indices,
                           const size_t n,
                           unsigned int *d_result)
{
  size_t block_size = 512;
  size_t num_blocks = (n / block_size) + (n % block_size) ? 1 : 0;
  scatter_even_elements_kernel<<<num_blocks,block_size>>>(d_input, d_scatter_indices, n, d_result);
}


__global__ void scatter_options_kernel(const real min_call_threshold,
                                       const real min_put_threshold,
                                       const real *d_call,
                                       const real *d_put,
                                       const real *d_stock_price_input,
                                       const real *d_option_strike_input,
                                       const real *d_option_years_input,
                                       const unsigned int *d_scatter_indices,
                                       const size_t n,
                                       real *d_stock_price_result,
                                       real *d_option_strike_result,
                                       real *d_option_years_result)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int grid_size = gridDim.x * blockDim.x;

  for(; i < n; i += grid_size)
  {
    const real call = d_call[i];
    const real put  = d_put[i];

    if((call >= min_call_threshold) && (put >= min_put_threshold))
    {
      d_stock_price_result[d_scatter_indices[i]] = d_stock_price_input[i];
      d_option_strike_result[d_scatter_indices[i]] = d_option_strike_input[i];
      d_option_years_result[d_scatter_indices[i]] = d_option_years_input[i];
    }
  }
}


void scatter_options(const real min_call_threshold,
                     const real min_put_threshold,
                     const real *d_call,
                     const real *d_put,
                     const real *d_stock_price_input,
                     const real *d_option_strike_input,
                     const real *d_option_years_input,
                     const unsigned int *d_scatter_indices,
                     const size_t n,
                     real *d_stock_price_result,
                     real *d_option_strike_result,
                     real *d_option_years_result)
{
  size_t block_size = 512;
  size_t num_blocks = (n / block_size) + (n % block_size) ? 1 : 0;
  scatter_options_kernel<<<num_blocks,block_size>>>(min_call_threshold,
                                                    min_put_threshold,
                                                    d_call,
                                                    d_put,
                                                    d_stock_price_input,
                                                    d_option_strike_input,
                                                    d_option_years_input,
                                                    d_scatter_indices,
                                                    n,
                                                    d_stock_price_result,
                                                    d_option_strike_result,
                                                    d_option_years_result);
  //check_cuda_error("scatter_options: after scatter_options_kernel", __FILE__, __LINE__);
}

