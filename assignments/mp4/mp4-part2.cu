// In Part 2 of MP 4, we'll revisit our stream compaction problem from MP 3 by implementing
// it with Thrust. Our strategy will be straightforward -- replace your ad hoc kernels
// and compaction calls from MP 3 with analogous calls to Thrust. You should only need
// two Thrust functions to complete this part.

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include "mp4-util.h"
#include "black_scholes.h"

// TODO: hint: functions and types in these headers will be useful for implementing a solution
#include <thrust/transform.h>
#include <thrust/remove.h>


// this struct encapsulates a stock's three parameters
struct stock
{
  real price, option_strike, option_years;
};


// this struct encapsulates the results of the Black-Scholes algorithm for a particular option
struct option
{
  real call, put;
};


struct black_scholes_functor
{
  black_scholes_functor(const real rr, const real v)
    : riskless_rate(rr), volatility(v)
  {}

  __host__ __device__
  option operator()(const stock &s) const
  {
    option result;

    // TODO: your call to black_scholes() here

    return result;
  }

  const real riskless_rate, volatility;
};


struct option_fails_threshold
  : thrust::unary_function<option,bool>
{
  option_fails_threshold(const real c, const real p)
    : min_call_threshold(c), min_put_threshold(p)
  {}

  __host__ __device__
  bool operator()(const option &o) const
  {
    bool result;

    // TODO: your computation here

    return result;
  }

  real min_call_threshold, min_put_threshold;
};


int main(void)
{
  event_pair timer;

  const size_t num_subsequent_rounds = 5;
  float compaction_time = 0;
  thrust::host_vector<float> gpu_time(1 + num_subsequent_rounds);
  thrust::host_vector<float> cpu_time(1 + num_subsequent_rounds);

  size_t num_options = 1<<22;

  // allocate host storage
  thrust::host_vector<stock>  h_stocks(num_options);
  thrust::host_vector<option> h_first_round_result(num_options);
  thrust::host_vector<option> h_subsequent_round_result(num_options);

  // generate options set
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<real> dist;
  for(int i = 0; i < num_options; ++i)
  {
    h_stocks[i].price         = dist(rng, thrust::make_pair(5.0, 30.0));
    h_stocks[i].option_strike = dist(rng, thrust::make_pair(1.0, 100.0));
    h_stocks[i].option_years  = dist(rng, thrust::make_pair(0.25, 10.0));
  }

  // allocate device storage
  thrust::device_vector<stock>  d_stocks = h_stocks;
  thrust::device_vector<option> d_first_round_result(num_options);
  thrust::device_vector<option> d_subsequent_round_result(num_options);


  // BEGIN ROUND 0

  // we will use the two following parameters
  // to first round of the Black-Scholes algorithm
  const real first_round_riskless_rate = 0.02;
  const real first_round_volatility    = 0.30;

  // do one round of Black-Scholes using our parameters on the device
  start_timer(&timer);
  // TODO: your thrust call here
  gpu_time[0] = stop_timer(&timer, "GPU Black-Scholes round 0");

  // do one round of Black-Scholes using our parameters on the host
  start_timer(&timer);
  // TODO: your thrust call here
  cpu_time[0] = stop_timer(&timer, "CPU Black-Scholes round 0");

  // validate gpu results from round 0
  // pass true as a final optional argument to fuzzy_validate for verbose output
  if(!fuzzy_validate(d_first_round_result, h_first_round_result))
  {
    std::cerr << "Error: round 0 of call results don't match!" << std::endl;
    exit(-1);
  }

  // BEGIN COMPACTION

  // in subsequent rounds, select the stocks whose call & put prices from the first round
  // meet or exceed these thresholds
  const real min_call_threshold = 2.0;
  const real min_put_threshold  = 4.0;

  option_fails_threshold pred(min_call_threshold, min_put_threshold);
  start_timer(&timer);
  size_t num_compacted_options = 0

  // TODO: your thrust call here
  compaction_time = stop_timer(&timer, "GPU Compaction");

  // resize the stocks vectors to the number of compacted results
  d_stocks.resize(num_compacted_options);
  d_subsequent_round_result.resize(num_compacted_options);

  // BEGIN SUBSEQUENT ROUNDS
  size_t num_compacted_options_reference = 0;
  for(int round = 1; round < num_subsequent_rounds + 1; ++round)
  {
    // change the parameters of the model in each subsequent round
    const real riskless_rate = dist(rng, thrust::make_pair(0.03, 0.04));
    const real volatility    = dist(rng, thrust::make_pair(0.50, 0.60));

    // do a round of Black-Scholes using new parameters on the device
    start_timer(&timer);
    // TODO: your thrust call here

    char message[256];
    sprintf(message, "GPU Black-Scholes round %d", round);
    gpu_time[round] = stop_timer(&timer, message);


    // do a round of Black-Scholes using new parameters on the host
    // filter the set of options to compute given the results of the last round,
    // but compact the output
    start_timer(&timer);
    sprintf(message, "CPU Black-Scholes round %d", round);
    num_compacted_options_reference = 0;
    for(int i = 0; i < num_options; ++i)
    {
      if((h_first_round_result[i].call >= min_call_threshold) &&
          (h_first_round_result[i].put  >= min_put_threshold))
      {
        option o;
        black_scholes(h_stocks[i].price,
                      h_stocks[i].option_strike,
                      h_stocks[i].option_years,
                      riskless_rate, volatility,
                      o.call, o.put);

        h_subsequent_round_result[num_compacted_options_reference] = o;
        ++num_compacted_options_reference;
      }
    }
    cpu_time[round] = stop_timer(&timer, message);

    h_subsequent_round_result.resize(num_compacted_options_reference);

    if(num_compacted_options_reference != num_compacted_options)
    {
      std::cerr << "Error: round " << round << " num_compacted_options (" << num_compacted_options << ") doesn't match num_compacted_options_reference (" << num_compacted_options_reference << ")" << std::endl;
      exit(-1);
    }

    // validate gpu results from this round
    // pass true as a final optional argument to fuzzy_validate for verbose output
    if(!fuzzy_validate(d_subsequent_round_result, h_subsequent_round_result))
    {
      std::cerr << "Error: round " << round << " of call results don't match!" << std::endl;
      exit(-1);
    }
  }

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

