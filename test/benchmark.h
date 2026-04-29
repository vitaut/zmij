// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <string>
#include <vector>

template <typename T>
struct method {
  std::string name;
  auto (*to_chars)(T, char*) -> char*;
};

template <typename T>
inline std::vector<method<T>> methods;

template <typename T>
inline auto register_method_(const std::string& name,
                             auto (*fn)(T, char*) -> char*) -> int {
  methods<T>.push_back({name, fn});
  return 0;
}

#define REGISTER_DTOA(f) \
  static int register_dtoa_##f = register_method_<double>(#f, dtoa_##f)

#define REGISTER_FTOA(f) \
  static int register_ftoa_##f = register_method_<float>(#f, ftoa_##f)

#endif  // BENCHMARK_H_
