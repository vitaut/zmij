// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"

#include <math.h>    // isnan
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // exit
#include <string.h>  // memcpy, strlen

#include <algorithm>  // std::sort
#include <charconv>   // std::from_chars
#include <chrono>     // std::chrono::steady_clock::now

#include "fmt/base.h"

constexpr int num_trials = 15;
constexpr int max_digits = std::numeric_limits<double>::max_digits10;
constexpr int num_iterations_per_digit = 1;
constexpr int num_doubles_per_digit = 100'000;

std::vector<method> methods;

// Random number generator from dtoa-benchmark.
class rng {
 public:
  explicit rng(unsigned seed = 0) : seed_(seed) {}

  auto operator()() -> unsigned {
    seed_ = 214013 * seed_ + 2531011;
    return seed_;
  }

 private:
  unsigned seed_;
};

auto get_random_digit_data(int digit) -> const double* {
  static const std::vector<double> random_digit_data = []() {
    std::vector<double> data;
    data.reserve(num_doubles_per_digit * max_digits);
    rng r;
    for (int digit = 1; digit <= max_digits; digit++) {
      for (size_t i = 0; i < num_doubles_per_digit; i++) {
        double d = 0;
        uint64_t bits = 0;
        do {
          bits = uint64_t(r()) << 32;
          bits |= r();  // Must be a separate statement to prevent reordering.
          memcpy(&d, &bits, sizeof(d));
        } while (isnan(d) || isinf(d));

        // Limit the number of digits.
        char buffer[64] = {};
        fmt::format_to_n(buffer, sizeof(buffer), "{:.{}}", d, digit);
        d = 0;
        std::from_chars(buffer, buffer + strlen(buffer), d);
        data.push_back(d);
      }
    }
    return data;
  }();
  return random_digit_data.data() + (digit - 1) * num_doubles_per_digit;
}

using duration = std::chrono::steady_clock::duration;

auto to_ns(duration d) -> double {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
}

struct digit_result {
  double median_ns = std::numeric_limits<double>::min();
  double mad_ns = std::numeric_limits<double>::min();
};

struct benchmark_result {
  double aggregated_ns = 0;
  double min_ns = std::numeric_limits<double>::max();
  double max_ns = std::numeric_limits<double>::min();
  digit_result per_digit[max_digits + 1];
  bool noisy = false;
};

// Modeled after https://github.com/fmtlib/dtoa-benchmark but using medians and
// retries to be more robust to noise and avoid the downward bias of minima.
auto bench_random_digit(auto (*dtoa)(double, char*) -> char*, const std::string& name)
    -> benchmark_result {
  char buffer[256] = {};
  constexpr int num_retries = 15;
  benchmark_result results[num_retries];
  for (int retry = 0; retry < num_retries; ++retry) {
    benchmark_result result;
    for (int digit = 1; digit <= max_digits; ++digit) {
      const double* data = get_random_digit_data(digit);

      duration durations[num_trials] = {};
      for (int trial = 0; trial < num_trials; ++trial) {
        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < num_iterations_per_digit; ++iter) {
          for (int i = 0; i < num_doubles_per_digit; ++i) dtoa(data[i], buffer);
        }
        auto finish = std::chrono::steady_clock::now();
        durations[trial] = finish - start;
      }

      // Compute the median, which estimates typical performance and avoids the
      // systematic downward bias of using the minimum under one-sided timing
      // noise (as in dtoa-benchmark).
      static_assert(num_trials % 2 == 1);
      std::sort(durations, durations + num_trials);
      duration median_duration = durations[num_trials / 2];

      // Compute absolute deviations from the median.
      duration deviations[num_trials];
      for (int i = 0; i < num_trials; ++i) {
        auto d = durations[i] - median_duration;
        deviations[i] = d < duration::zero() ? -d : d;
      }

      // Compute median of deviations (MAD).
      std::sort(deviations, deviations + num_trials);
      duration mad_duration = deviations[num_trials / 2];

      double median_ns = to_ns(median_duration);
      median_ns /= num_iterations_per_digit * num_doubles_per_digit;

      double mad_ns = to_ns(mad_duration);
      mad_ns /= num_iterations_per_digit * num_doubles_per_digit;
      if (mad_ns / median_ns > 0.01) result.noisy = true;

      result.per_digit[digit].median_ns = median_ns;
      result.per_digit[digit].mad_ns = mad_ns;
      if (median_ns < result.min_ns) result.min_ns = median_ns;
      if (median_ns > result.max_ns) result.max_ns = median_ns;
    }

    for (int i = 1; i <= max_digits; ++i)
      result.aggregated_ns += result.per_digit[i].median_ns;
    result.aggregated_ns /= max_digits;

    results[retry] = result;
  }
  std::sort(results, results + num_retries,
            [](const benchmark_result& lhs, const benchmark_result& rhs) {
              return lhs.aggregated_ns < rhs.aggregated_ns;
            });
  return results[num_retries / 2];
}

auto main() -> int {
  std::sort(
      methods.begin(), methods.end(),
      [](const method& lhs, const method& rhs) { return lhs.name < rhs.name; });

  fmt::print("Mean of per-digit medians:\n");
  for (const method& m : methods) {
    benchmark_result result = bench_random_digit(m.dtoa, m.name);
    fmt::print("{:9}: {:5.2f}ns ({:5.2f}ns - {:5.2f}ns) {}\n", m.name,
               result.aggregated_ns, result.min_ns, result.max_ns,
               result.noisy ? "noisy" : "");
  }
}
