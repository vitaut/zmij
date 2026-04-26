// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"

#include <math.h>    // isnan
#include <stdint.h>  // uint64_t
#include <stdlib.h>  // strtod
#include <string.h>  // memcpy

#include <algorithm>  // std::sort
#include <limits>
#include <string>

#include <benchmark/benchmark.h>

#include "fmt/base.h"

constexpr int max_digits = std::numeric_limits<double>::max_digits10;
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
        d = strtod(buffer, nullptr);
        data.push_back(d);
      }
    }
    return data;
  }();
  return random_digit_data.data() + (digit - 1) * num_doubles_per_digit;
}

static void run_dtoa(benchmark::State& state,
                     auto (*dtoa)(double, char*) -> char*, int digit) {
  const double* data = get_random_digit_data(digit);
  char buffer[256];
  for (auto _ : state) {
    for (int i = 0; i < num_doubles_per_digit; ++i) {
      char* end = dtoa(data[i], buffer);
      benchmark::DoNotOptimize(end);
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations() *
                          int64_t{num_doubles_per_digit});
}

auto main(int argc, char** argv) -> int {
  std::sort(
      methods.begin(), methods.end(),
      [](const method& lhs, const method& rhs) { return lhs.name < rhs.name; });
  for (const method& m : methods) {
    for (int digit = 1; digit <= max_digits; ++digit) {
      auto name = m.name + "/d" + std::to_string(digit);
      benchmark::RegisterBenchmark(name.c_str(), run_dtoa, m.dtoa, digit);
    }
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
