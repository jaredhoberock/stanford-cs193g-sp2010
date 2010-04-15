#pragma once

#include "mp3-util.h"


void black_scholes_host(const real *d_stock_price,
                        const real *d_option_strike,
                        const real *d_option_years,
                        real *d_call_result,
                        real *d_put_result,
                        const real riskless_rate,
                        const real volatility,
                        const size_t n);


size_t filtered_black_scholes_host(const real *d_stock_price,
                                   const real *d_option_strike,
                                   const real *d_option_years,
                                   const real *d_previous_round_call_result,
                                   const real *d_previous_round_put_result,
                                   real *d_call_result,
                                   real *d_put_result,
                                   const real min_call_result,
                                   const real min_put_result,
                                   const real riskless_rate,
                                   const real volatility,
                                   const size_t n);


size_t compacted_black_scholes_host(const real *d_stock_price,
                                    const real *d_option_strike,
                                    const real *d_option_years,
                                    const real *d_previous_round_call_result,
                                    const real *d_previous_round_put_result,
                                    real *d_call_result,
                                    real *d_put_result,
                                    const real min_call_result,
                                    const real min_put_result,
                                    const real riskless_rate,
                                    const real volatility,
                                    const size_t n);

__global__
void black_scholes_kernel(const real *d_stock_price,
                          const real *d_option_strike,
                          const real *d_option_years,
                          real *d_call_result,
                          real *d_put_result,
                          const real riskless_rate,
                          const real volatility,
                          const size_t n);


__global__
void naively_filtered_black_scholes_kernel(const real *d_stock_price,
                                           const real *d_option_strike,
                                           const real *d_option_years,
                                           const real *d_previous_round_call_result,
                                           const real *d_previous_round_put_result,
                                           real *d_call_result,
                                           real *d_put_result,
                                           const real min_call_result,
                                           const real min_put_result,
                                           const real riskless_rate,
                                           const real volatility,
                                           const size_t n);
