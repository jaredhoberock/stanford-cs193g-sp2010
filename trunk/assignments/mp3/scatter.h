#pragma once

#include "mp3-util.h"

void scatter_even_elements(const unsigned int *d_input,
                           const unsigned int *d_scatter_indices,
                           const size_t n,
                           unsigned int *d_result);

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
                     real *d_option_years_result);

