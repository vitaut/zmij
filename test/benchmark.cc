// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"

#include <stdint.h>  // uint64_t
#include <stdio.h>   // snprintf

#include <algorithm>  // std::sort
#include <charconv>   // std::from_chars
#include <chrono>     // std::chrono::steady_clock::now

#include "fmt/base.h"

constexpr int num_trials = 3;
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
        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.*g", digit, d);
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

struct digit_result {
  double duration_ns = std::numeric_limits<double>::min();
};

struct benchmark_result {
  double min_ns = std::numeric_limits<double>::max();
  double max_ns = std::numeric_limits<double>::min();
  digit_result per_digit[max_digits + 1];
};

// Modeled after https://github.com/fmtlib/dtoa-benchmark.
auto bench_random_digit(void (*dtoa)(double, char*), const std::string& name)
    -> benchmark_result {
  char buffer[256] = {};
  benchmark_result result;
  for (int digit = 1; digit <= max_digits; ++digit) {
    const double* data = get_random_digit_data(digit);

    duration run_duration = duration::max();
    for (int trial = 0; trial < num_trials; ++trial) {
      auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < num_iterations_per_digit; ++iter) {
        for (int i = 0; i < num_doubles_per_digit; ++i) dtoa(data[i], buffer);
      }
      auto finish = std::chrono::steady_clock::now();

      // Pick the smallest of trial runs.
      auto d = finish - start;
      if (d < run_duration) run_duration = d;
    }

    double ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(run_duration)
            .count();
    ns /= num_iterations_per_digit * num_doubles_per_digit;

    result.per_digit[digit].duration_ns = ns;
    if (ns < result.min_ns) result.min_ns = ns;
    if (ns > result.max_ns) result.max_ns = ns;
  }
  return result;
}

auto main() -> int {
  std::sort(
      methods.begin(), methods.end(),
      [](const method& lhs, const method& rhs) { return lhs.name < rhs.name; });

  for (const method& m : methods) {
    benchmark_result result = bench_random_digit(m.dtoa, m.name);
    fmt::print("{:9}: {:5.2f}..{:.2f}ns\n", m.name, result.min_ns,
               result.max_ns);
  }
}
