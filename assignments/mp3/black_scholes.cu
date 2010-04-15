#include "black_scholes.h"
#include <math.h>


// Polynomial approximation of cumulative normal distribution function
__host__ __device__ 
inline real cumulative_normal_distribution(real d)
{
  const real       A1 = 0.31938153;
  const real       A2 = -0.356563782;
  const real       A3 = 1.781477937;
  const real       A4 = -1.821255978;
  const real       A5 = 1.330274429;
  const real RSQRT2PI = 0.39894228040143267793994605993438;
  
  real K = 1.0 / (1.0 + 0.2316419 * fabs(d));
  
  real cnd = RSQRT2PI * exp(-0.5 * d * d) * 
              (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
  
  if(d > 0)
  {
    cnd = 1.0 - cnd;
  }
  
  return cnd;
}


__host__ __device__
void black_scholes(const real stock_price,
                   const real option_strike,
                   const real option_years,
                   const real riskless_rate,
                   const real volatility,
                   real &call_result,
                   real &put_result)
{
   real sqrt_option_years = sqrt(option_years);
   real d1 = (log(stock_price / option_strike) + (riskless_rate + 0.5 * volatility * volatility) * option_years) / (volatility * sqrt_option_years);
   real d2 = d1 - volatility * sqrt_option_years;

   real CNDD1 = cumulative_normal_distribution(d1);
   real CNDD2 = cumulative_normal_distribution(d2);

   // calculate call and put simultaneously
   real expRT = exp(-riskless_rate * option_years);

   call_result = stock_price * CNDD1 - option_strike * expRT * CNDD2;
   put_result  = option_strike * expRT * (1.0 - CNDD2) - stock_price * (1.0 - CNDD1);
}


void black_scholes_host(const real *d_stock_price,
                        const real *d_option_strike,
                        const real *d_option_years,
                        real *d_call_result,
                        real *d_put_result,
                        const real riskless_rate,
                        const real volatility,
                        const size_t n)
{
  for(size_t i = 0; i < n; ++i)
  {
    // compute the call & put
    black_scholes(d_stock_price[i], d_option_strike[i], d_option_years[i],
                  riskless_rate, volatility,
                  d_call_result[i], d_put_result[i]);
  }
}


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
                                   const size_t n)
{
  size_t result = 0;
  for(size_t i = 0; i < n; ++i)
  {
    // load the previous round's call & put results
    const real previous_call_result = d_previous_round_call_result[i];
    const real previous_put_result  = d_previous_round_put_result[i];

    // check the previous results against the minimums we're interested in
    if(previous_call_result >= min_call_result &&
       previous_put_result  >= min_put_result)
    {
      // compute the call & put
      black_scholes(d_stock_price[i], d_option_strike[i], d_option_years[i],
                    riskless_rate, volatility,
                    d_call_result[i], d_put_result[i]);
      ++result;
    }
  }

  // return the number of filtered options
  return result;
}


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
                                    const size_t n)
{
  size_t result = 0;
  for(size_t i = 0; i < n; ++i)
  {
    // load the previous round's call & put results
    const real previous_call_result = d_previous_round_call_result[i];
    const real previous_put_result  = d_previous_round_put_result[i];

    // check the previous results against the minimums we're interested in
    if(previous_call_result >= min_call_result &&
       previous_put_result  >= min_put_result)
    {
      // compute the call & put
      black_scholes(d_stock_price[i], d_option_strike[i], d_option_years[i],
                    riskless_rate, volatility,
                    d_call_result[result], d_put_result[result]);
      ++result;
    }
  }

  // return the number of filtered options
  return result;
}


__global__
void black_scholes_kernel(const real *d_stock_price,
                          const real *d_option_strike,
                          const real *d_option_years,
                          real *d_call_result,
                          real *d_put_result,
                          const real riskless_rate,
                          const real volatility,
                          const size_t n)
{
  // TODO: your kernel implementation here
}

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
                                           const size_t n)
{
  // TODO: your kernel implementation here
}

