// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"

#include <benchmark/benchmark.h>
#include <stdint.h>  // uint32_t, uint64_t
#include <stdlib.h>  // strtod, strtof
#include <string.h>  // memcpy

#include <algorithm>  // std::sort, std::shuffle
#include <cmath>      // std::abs, std::isnan, std::isinf
#include <fstream>
#include <limits>
#include <random>  // std::mt19937
#include <string>
#include <string_view>
#include <type_traits>

#include "fmt/format.h"

constexpr int num_per_digit = 100'000;

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

template <typename T>
auto get_random_digit_data(int digit) -> const T* {
  constexpr int max_digits = std::numeric_limits<T>::max_digits10;
  static const std::vector<T> random_digit_data = []() {
    std::vector<T> data;
    data.reserve(num_per_digit * max_digits);
    rng r;
    for (int d = 1; d <= max_digits; ++d) {
      for (size_t i = 0; i < num_per_digit; ++i) {
        T val = 0;
        do {
          if constexpr (sizeof(T) == sizeof(uint32_t)) {
            uint32_t bits = r();
            memcpy(&val, &bits, sizeof(val));
          } else {
            uint64_t bits = uint64_t(r()) << 32;
            bits |= r();  // Separate statement to prevent reordering.
            memcpy(&val, &bits, sizeof(val));
          }
        } while (std::isnan(val) || std::isinf(val));

        // Limit the number of digits.
        char buffer[64] = {};
        fmt::format_to_n(buffer, sizeof(buffer), "{:.{}}", val, d);
        if constexpr (std::is_same_v<T, double>)
          val = strtod(buffer, nullptr);
        else
          val = strtof(buffer, nullptr);
        data.push_back(val);
      }
    }
    return data;
  }();
  return random_digit_data.data() + (digit - 1) * num_per_digit;
}

template <typename T>
auto get_mixed_pool() -> const std::vector<T>& {
  constexpr int max_digits = std::numeric_limits<T>::max_digits10;
  static const std::vector<T> pool = [] {
    std::vector<T> v;
    v.reserve(num_per_digit * max_digits);
    for (int d = 1; d <= max_digits; ++d) {
      const T* p = get_random_digit_data<T>(d);
      v.insert(v.end(), p, p + num_per_digit);
    }
    std::shuffle(v.begin(), v.end(), std::mt19937(0));
    return v;
  }();
  return pool;
}

template <typename T>
static void run_to_chars(benchmark::State& state,
                         auto (*to_chars)(T, char*)->char*, int digit) {
  const T* data = get_random_digit_data<T>(digit);
  char buffer[256];
  for (auto _ : state) {
    for (int i = 0; i < num_per_digit; ++i) {
      char* end = to_chars(data[i], buffer);
      benchmark::DoNotOptimize(end);
      benchmark::ClobberMemory();
    }
  }
  state.counters["Throughput"] = benchmark::Counter(
      static_cast<double>(num_per_digit),
      benchmark::Counter::kIsIterationInvariantRate);
  const char* time_label =
      std::is_same_v<T, double> ? "Time/double" : "Time/float";
  state.counters[time_label] = benchmark::Counter(
      static_cast<double>(num_per_digit),
      benchmark::Counter::kIsIterationInvariantRate |
          benchmark::Counter::kInvert);
}

template <typename T>
static void run_to_chars_mixed(benchmark::State& state,
                               auto (*to_chars)(T, char*)->char*) {
  const auto& pool = get_mixed_pool<T>();
  char buffer[256];
  for (auto _ : state) {
    for (T x : pool) {
      char* end = to_chars(x, buffer);
      benchmark::DoNotOptimize(end);
      benchmark::ClobberMemory();
    }
  }
  state.counters["Throughput"] = benchmark::Counter(
      static_cast<double>(pool.size()),
      benchmark::Counter::kIsIterationInvariantRate);
  const char* time_label =
      std::is_same_v<T, double> ? "Time/double" : "Time/float";
  state.counters[time_label] = benchmark::Counter(
      static_cast<double>(pool.size()),
      benchmark::Counter::kIsIterationInvariantRate |
          benchmark::Counter::kInvert);
}

// Doubles extracted from the canonical canada.json corpus (GeoJSON polygon of
// Canada). canada.h is a bare initializer list, one number per line.
static const double canada_numbers[] = {
#include "canada.h"
};

static void run_to_chars_canada(benchmark::State& state,
                                auto (*to_chars)(double, char*)->char*) {
  constexpr size_t canada_numbers_count =
      sizeof(canada_numbers) / sizeof(canada_numbers[0]);
  char buffer[256];
  for (auto _ : state) {
    for (size_t i = 0; i < canada_numbers_count; ++i) {
      char* end = to_chars(canada_numbers[i], buffer);
      benchmark::DoNotOptimize(end);
      benchmark::ClobberMemory();
    }
  }
  state.counters["Throughput"] = benchmark::Counter(
      static_cast<double>(canada_numbers_count),
      benchmark::Counter::kIsIterationInvariantRate);
  state.counters["Time/double"] = benchmark::Counter(
      static_cast<double>(canada_numbers_count),
      benchmark::Counter::kIsIterationInvariantRate |
          benchmark::Counter::kInvert);
}

// Formats a counter value with 2 fractional digits, applying SI auto-scaling
// so the mantissa always sits in [1, 1000) (or in [0.01, 1) for tiny values).
static auto format_counter(double n) -> std::string {
  static const char* const big[] = {"k", "M", "G", "T", "P", "E", "Z", "Y"};
  static const char* const small[] = {"m", "u", "n", "p", "f", "a", "z", "y"};
  double v = n;
  const char* prefix = "";
  double a = std::abs(v);
  if (a > 999) {
    for (int i = 0; i < 8 && std::abs(v) > 999; ++i) {
      v /= 1000;
      prefix = big[i];
    }
  } else if (a > 0 && a < 0.01) {
    for (int i = 0; i < 8 && std::abs(v) < 1.0; ++i) {
      v *= 1000;
      prefix = small[i];
    }
  }
  return fmt::format("{:.2f}{}", v, prefix);
}

// Console reporter that formats counter cells with 2 fractional digits while
// reusing google benchmark's tabular column layout for everything else.
class pretty_reporter : public benchmark::ConsoleReporter {
 public:
  pretty_reporter() : benchmark::ConsoleReporter(OO_Tabular) {}

 protected:
  void PrintRunData(const Run& report) override {
    Run copy = report;
    std::string cells;
    for (const auto& kv : copy.counters) {
      const auto& c = kv.second;
      const char* unit = "";
      if (c.flags & benchmark::Counter::kIsRate)
        unit = (c.flags & benchmark::Counter::kInvert) ? "s" : "/s";
      auto cell = format_counter(c.value) + unit;
      std::size_t w = std::max<std::size_t>(10, kv.first.length());
      if (!cells.empty()) cells += ' ';
      cells += fmt::format("{:>{}}", cell, w);
    }
    if (!copy.report_label.empty()) {
      if (!cells.empty()) cells += ' ';
      cells += copy.report_label;
    }
    copy.counters.clear();
    copy.report_label = std::move(cells);
    benchmark::ConsoleReporter::PrintRunData(copy);
  }
};

template <typename T>
static void register_all(bool per_digit) {
  auto& v = methods<T>;
  std::sort(v.begin(), v.end(), [](const method<T>& a, const method<T>& b) {
    return a.name < b.name;
  });
  constexpr int max_digits = std::numeric_limits<T>::max_digits10;
  for (const auto& m : v) {
    if (per_digit) {
      for (int d = 1; d <= max_digits; ++d) {
        auto name = m.name + "/d" + std::to_string(d);
        benchmark::RegisterBenchmark(name.c_str(), run_to_chars<T>, m.to_chars,
                                     d);
      }
    }
    benchmark::RegisterBenchmark(m.name.c_str(), run_to_chars_mixed<T>,
                                 m.to_chars);
    if constexpr (std::is_same_v<T, double>) {
      auto canada_name = m.name + "/canada";
      benchmark::RegisterBenchmark(canada_name.c_str(), run_to_chars_canada,
                                   m.to_chars);
    }
  }
}

auto main(int argc, char** argv) -> int {
  bool per_digit = false;
  std::string json_out;
  int out = 1;
  for (int i = 1; i < argc; ++i) {
    auto arg = std::string_view(argv[i]);
    if (arg == "--per-digit") {
      per_digit = true;
    } else if (arg.substr(0, 11) == "--json-out=") {
      json_out = std::string(arg.substr(11));
    } else {
      argv[out++] = argv[i];
    }
  }
  argc = out;

  register_all<double>(per_digit);
  register_all<float>(per_digit);

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  if (!json_out.empty()) {
    std::ofstream json_file(json_out);
    benchmark::JSONReporter json_reporter;
    json_reporter.SetOutputStream(&json_file);
    benchmark::RunSpecifiedBenchmarks(&json_reporter);
  } else {
    pretty_reporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
  }
  benchmark::Shutdown();
}
