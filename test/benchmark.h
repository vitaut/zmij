// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <string>
#include <vector>

struct method {
  std::string name;
  void (*dtoa)(double, char*);
};

extern std::vector<method> methods;

#define REGISTER_METHOD(f)                     \
  static int register_##f = []() {             \
    methods.push_back(method{#f, dtoa##_##f}); \
    return 0;                                  \
  }()

#endif  // BENCHMARK_H_
