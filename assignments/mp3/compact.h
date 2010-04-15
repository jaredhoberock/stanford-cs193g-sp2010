#pragma once

#include "mp3-util.h"

size_t compact_even_elements(const unsigned int *d_input,
                             const size_t n,
                             unsigned int *d_result);

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
                       real *d_option_years_result);

