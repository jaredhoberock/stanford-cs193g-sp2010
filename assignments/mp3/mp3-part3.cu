// In Part 3, we'll use the stream compaction implementation we built
// in Part 2 to create a simple schedule for the second round of
// Black-Scholes jobs from Part 1. Since subsequent rounds of our options
// pricing are very sparse, a naive scheduling of this job will
// cause the majority of CUDA threads to idle while ther active warp
// neighbors work on stocks from the first round which meet our criteria. A better
// scheduling should compact all of the data which meets or exceeds the
// threshold for subsequent rounds to the front of the input, and launch only as many
// threads as there is work to do. This will ensure that all of the CUDA threads
// launched in subsequent rounds will be active (modulo the few inactive threads
// at the end of the last block). This compaction scheme eliminates the warp
// divergence penalty and increases the throughput of the second round of Black-Scholes.

// Note that your stream compaction implementation may come with significant overhead.
// How many sparse rounds of Black-Scholes does it take before the cost becomes worth
// it? It may take quite a few when computing in single precision. For double precision,
// it requires just a handful. You can control whether the "real" type is "float"
// (single precision) or "double" (double precision) by editing the Makefile.

// If you've made it this far, Part 3 should be trivial. The only thing you need
// to do is launch kernels and, as always, allocate & deallocate device storage in
// mp3-part3.cu and implement the body of compact_options in compact.cu. This is mostly
// a copy and paste job from compact_even_elements, but this time, there is more data
// to move around. Search for the lines marked TODO: and get hacking!

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <limits>
#include <numeric>

#include "black_scholes.h"
#include "compact.h"
#include "mp3-util.h"


bool allocate_device_storage(real *&d_first_round_call_result, real *&d_first_round_put_result,
                             real *&d_subsequent_round_call_result, real *&d_subsequent_round_put_result,
                             real *&d_stock_price, real *&d_option_strike,
                             real *&d_option_years,
                             real *&d_compacted_stock_price,
                             real *&d_compacted_option_strike,
                             real *&d_compacted_option_years,
                             const size_t n)
{
  // TODO: your device memory allocations here
  // TODO: don't forget to check for CUDA errors!

  // TODO: return true to indicate successful memory allocation
  return false;
}


void deallocate_device_storage(real *d_first_round_call_result, real *d_first_round_put_result,
                               real *d_subsequent_round_call_result, real *d_subsequent_round_put_result,
                               real *d_stock_price, real *d_option_strike,
                               real *d_option_years,
                               real *d_compacted_stock_price,
                               real *d_compacted_option_strike,
                               real *d_compacted_option_years)
{
  // TODO: your device memory deallocations here
  // TODO: don't forget to check for CUDA errors!
}


int main(void)
{
  event_pair timer;

  const size_t num_subsequent_rounds = 5;
  float compaction_time = 0;
  std::vector<float> gpu_time(1 + num_subsequent_rounds);
  std::vector<float> cpu_time(1 + num_subsequent_rounds);

  // create arrays for 4M options
  size_t num_options = 1<<22;

  // allocate host storage
  std::vector<real> h_first_round_call_result(num_options,0);
  std::vector<real> h_first_round_put_result(num_options, 0);
  std::vector<real> h_subsequent_round_call_result(num_options,0);
  std::vector<real> h_subsequent_round_put_result(num_options, 0);
  std::vector<real> h_stock_price(num_options);
  std::vector<real> h_option_strike(num_options);
  std::vector<real> h_option_years(num_options);

  // generate options set
  srand(5347);
  for(int i = 0; i < num_options; ++i)
  {
    h_stock_price[i]   = random_real(5.0,  30.0);
    h_option_strike[i] = random_real(1.0, 100.0);
    h_option_years[i]  = random_real(0.25, 10.0);
  }

  // some pointers to the data set which will live in device memory
  real *d_first_round_call_result      = 0;
  real *d_first_round_put_result       = 0;
  real *d_subsequent_round_call_result = 0;
  real *d_subsequent_round_put_result  = 0;
  real *d_stock_price                  = 0;
  real *d_option_strike                = 0;
  real *d_option_years                 = 0;
  real *d_compacted_stock_price        = 0;
  real *d_compacted_option_strike      = 0;
  real *d_compacted_option_years       = 0;

  // allocate device storage
  if(!allocate_device_storage(d_first_round_call_result, d_first_round_put_result,
                              d_subsequent_round_call_result, d_subsequent_round_put_result,
                              d_stock_price, d_option_strike, d_option_years,
                              d_compacted_stock_price,
                              d_compacted_option_strike,
                              d_compacted_option_years,
                              num_options))
  {
    std::cerr << "Error allocating device memory!" << std::endl;
    exit(-1);
  }

  // fill the result arrays with 0
  cudaMemset(d_first_round_call_result,      0, sizeof(real) * num_options);
  cudaMemset(d_first_round_put_result,       0, sizeof(real) * num_options);
  cudaMemset(d_subsequent_round_call_result, 0, sizeof(real) * num_options);
  cudaMemset(d_subsequent_round_put_result,  0, sizeof(real) * num_options);

  // copy input to GPU
  start_timer(&timer);
  // TODO: your host to device copies here
  stop_timer(&timer, "host to device copy of input"); 


  // BEGIN ROUND 0

  // we will use the two following parameters
  // to first round of the Black-Scholes algorithm
  const real first_round_riskless_rate = 0.02;
  const real first_round_volatility    = 0.30;

  // do one round of Black-Scholes using our parameters
  start_timer(&timer);
  // TODO: your black_scholes_kernel launch here
  gpu_time[0] = stop_timer(&timer, "GPU Black-Scholes round 0");
  check_cuda_error("GPU Black-Scholes round 0", __FILE__, __LINE__);

  // do round 0 of Black-Scholes on the host
  start_timer(&timer);
  black_scholes_host(&h_stock_price[0],
                     &h_option_strike[0],
                     &h_option_years[0],
                     &h_first_round_call_result[0],
                     &h_first_round_put_result[0],
                     first_round_riskless_rate,
                     first_round_volatility,
                     num_options);
  cpu_time[0] = stop_timer(&timer, "CPU Black-Scholes round 0");

  // validate gpu results from round 0
  std::vector<real> h_validate_me(num_options);
  cudaMemcpy(&h_validate_me[0], d_first_round_call_result, sizeof(real) * num_options, cudaMemcpyDeviceToHost);
  // pass true as a final optional argument to fuzzy_validate for verbose output
  if(!fuzzy_validate(&h_validate_me[0], &h_first_round_call_result[0], num_options))
  {
    std::cerr << "Error: round 0 of call results don't match!" << std::endl;
    exit(-1);
  }

  cudaMemcpy(&h_validate_me[0],  d_first_round_put_result,  sizeof(real) * num_options, cudaMemcpyDeviceToHost);
  if(!fuzzy_validate(&h_validate_me[0], &h_first_round_put_result[0], num_options))
  {
    std::cerr << "Error: round 0 of put results don't match!" << std::endl;
    exit(-1);
  }


  // BEGIN COMPACTION


  // in subsequent rounds, select the stocks whose call & put prices from the first round
  // meet or exceed these thresholds
  const real min_call_threshold = 2.0;
  const real min_put_threshold  = 4.0;

  // compact the options, copying those that meet our call & put thresholds
  // to the arrays for round 2
  start_timer(&timer);
  size_t num_compacted_options = 0;
  // TODO: your call to compact_options here
  compaction_time = stop_timer(&timer, "GPU Compaction");


  // BEGIN SUBSEQUENT ROUNDS

  size_t num_compacted_options_reference = 0;

  for(int round = 1; round < num_subsequent_rounds + 1; ++round)
  {
    // change the parameters of the model in each subsequent round
    const real riskless_rate = random_real(0.03, 0.04);
    const real volatility    = random_real(0.50, 0.60);

    // do round of Black-Scholes using new parameters on the device
    start_timer(&timer);
    // TODO: your black_scholes_kernel launch here
    char message[256];
    sprintf(message, "GPU Black-Scholes round %d", round);
    gpu_time[round] = stop_timer(&timer, message);
    check_cuda_error(message, __FILE__, __LINE__);


    // do a round of Black-Scholes on the host using new parameters
    // filter the set of options to compute given the results of the last round,
    // but compact the output
    start_timer(&timer);
    num_compacted_options_reference =
      compacted_black_scholes_host(&h_stock_price[0],
                                   &h_option_strike[0],
                                   &h_option_years[0],
                                   &h_first_round_call_result[0],
                                   &h_first_round_put_result[0],
                                   &h_subsequent_round_call_result[0],
                                   &h_subsequent_round_put_result[0],
                                   min_call_threshold,
                                   min_put_threshold,
                                   riskless_rate,
                                   volatility,
                                   num_options);
    sprintf(message, "CPU Black-Scholes round %d", round);
    cpu_time[round] = stop_timer(&timer, message);

    if(num_compacted_options_reference != num_compacted_options)
    {
      std::cerr << "Error: round " << round << " num_compacted_options (" << num_compacted_options << ") doesn't match num_compacted_options_reference (" << num_compacted_options_reference << ")" << std::endl;
      exit(-1);
    }

    // validate gpu results from this round
    cudaMemcpy(&h_validate_me[0], d_subsequent_round_call_result, sizeof(real) * num_compacted_options_reference, cudaMemcpyDeviceToHost);
    if(!fuzzy_validate(&h_validate_me[0], &h_subsequent_round_call_result[0], num_compacted_options_reference))
    {
      std::cerr << "Error: round " << round << " of call results don't match!" << std::endl;
      exit(-1);
    }

    cudaMemcpy(&h_validate_me[0],  d_subsequent_round_put_result,  sizeof(real) * num_compacted_options_reference, cudaMemcpyDeviceToHost);
    if(!fuzzy_validate(&h_validate_me[0], &h_subsequent_round_put_result[0], num_compacted_options_reference))
    {
      std::cerr << "Error: round " << round << " of put results don't match!" << std::endl;
      exit(-1);
    }

  } // end for subsequent round

  deallocate_device_storage(d_first_round_call_result, d_first_round_put_result,
                            d_subsequent_round_call_result, d_subsequent_round_put_result,
                            d_stock_price, d_option_strike,
                            d_option_years,
                            d_compacted_stock_price,
                            d_compacted_option_strike,
                            d_compacted_option_years);

  // output a report
  std::cout << std::endl;

  real first_round_gpu_throughput = static_cast<real>(num_options) / (gpu_time[0] / 1000.0f);
  real first_round_cpu_throughput = static_cast<real>(num_options) / (cpu_time[0] / 1000.0f);

  std::cout << "Round 0: " << num_options << " options" << std::endl;
  std::cout << "Throughput of GPU Black-Scholes Round 0: " << (first_round_gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Throughput of CPU Black-Scholes Round 0: " << (first_round_cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Speedup of Round 0: " << first_round_gpu_throughput / first_round_cpu_throughput << "x" << std::endl << std::endl;

  for(int i = 1; i < gpu_time.size(); ++i)
  {
    real gpu_throughput = static_cast<real>(num_compacted_options_reference) / (gpu_time[i] / 1000.0f);
    real cpu_throughput = static_cast<real>(num_compacted_options_reference) / (cpu_time[i] / 1000.0f);

    std::cout << "Round " << i << ": " << num_compacted_options_reference << " options" << std::endl;
    std::cout << "Throughput of GPU Black-Scholes Round " << i << ": " << (gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
    std::cout << "Throughput of CPU Black-Scholes Round " << i << ": " << (cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
    std::cout << "Speedup of Round " << i << ": " << gpu_throughput / cpu_throughput << "x" << std::endl << std::endl;
  }

  // report overall performance
  real total_gpu_time = compaction_time + std::accumulate(gpu_time.begin(), gpu_time.end(), 0.0);
  real total_cpu_time = std::accumulate(cpu_time.begin(), cpu_time.end(), 0.0);
  real gpu_throughput = static_cast<real>(num_options + num_subsequent_rounds*num_compacted_options_reference) / ((total_gpu_time) / 1000.0f);
  real cpu_throughput = static_cast<real>(num_options + num_subsequent_rounds*num_compacted_options_reference) / ((total_cpu_time) / 1000.0f);

  std::cout << "Overall GPU throughput: " << (gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Overall CPU throughput: " << (cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl << std::endl;

  std::cout << "Overall speedup: " << gpu_throughput / cpu_throughput << "x" << std::endl;

  return 0;
}

