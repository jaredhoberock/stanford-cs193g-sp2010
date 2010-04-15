#include "compact.h"
#include "scan.h"
#include "scatter.h"
#include "mp3-util.h"


// compact_even_elements copies the even-valued elements of d_input
// to a compacted output in d_result, and returns the number of 
// elements in the compacted output
// for example, given
// d_input           = [ 0 1 2 4 10 2 3 5 0 3 ] and
// n = 10,
// this function yields
// d_result          = [ 0 2 4 10 2 0 <undefined...> ]
// and returns the value 6
size_t compact_even_elements(const unsigned int *d_input,
                             const size_t n,
                             unsigned int *d_result)
{
  // implement compaction by first creating a temporary "bit vector", with a 1
  // corresponding to each even element, and a 0 corresponding to each odd
  // element of d_input. next, do an in-place exclusive scan on this temporary buffer
  // then, scatter elements from d_input to d_result using the result of
  // the scan as a scatter map
  // hint: use the scatter_even_elements function in scatter.h

  // for example, given
  // d_input = [ 0 1 2 4 10 2 3 5 0 3 ]
  // your temporary bit vector should look like:
  // d_temp  = [ 1 0 1 1  1 1 0 0 1 0 ]

  // TODO: your implementation here

  size_t num_compacted_elements = 0;

  return num_compacted_elements;
}



// compact_options copies the input options whose call and put
// results from the first round meet or exceed the given call & put
// thresholds to a compacted output in three result arrays.
size_t compact_options(const real min_call_threshold,
                       const real min_put_threshold,
                       const real *d_call,
                       const real *d_put,
                       const real *d_stock_price_input,
                       const real *d_option_strike_input,
                       const real *d_option_years_input,
                       const size_t n,
                       real *d_stock_price_result,
                       real *d_option_strike_result,
                       real *d_option_years_result)
{
  // implement compaction by first creating a temporary "bit vector", with a 1
  // corresponding to each option which meets or exceeds the min_call_threshold
  // and min_put_threshold, and a 0, otherwise.
  // next, do an in-place exclusive scan on this temporary buffer
  // then, scatter elements from d_input to d_result using the result of
  // the scan as a scatter map
  // hint: you can mostly copy & paste from your implementation of compact_even_elements
  // hint: use the scatter_options function in scatter.h

  // TODO: your implementation here
  size_t num_compacted_elements = 0;

  return num_compacted_elements;
}

