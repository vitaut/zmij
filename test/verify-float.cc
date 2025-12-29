// A program to verify correctness of https://github.com/vitaut/zmij/
// on all floats.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <stdint.h>  // uint32_t
#include <stdio.h>   // printf
#include <string.h>  // memcpy

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "dragonbox/dragonbox_to_chars.h"
#include "zmij.h"

auto main() -> int {
  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::atomic<unsigned long long> num_processed_floats(0);
  std::atomic<unsigned long long> num_errors(0);
  printf("Using %u threads\n", num_threads);

  auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < num_threads; ++i) {
    constexpr unsigned long long num_floats = 1ULL << 32;
    uint32_t begin = static_cast<uint32_t>(num_floats * i / num_threads);
    uint32_t end = static_cast<uint32_t>(num_floats * (i + 1) / num_threads);
    uint32_t n = end - begin;
    threads[i] = std::thread([i, begin, n, &num_processed_floats, &num_errors] {
      char actual[zmij::float_buffer_size] = {};
      char expected[32] = {};
      constexpr double percent = 100.0 / num_floats;
      uint64_t last_processed_count = 0;
      auto last_update_time = std::chrono::steady_clock::now();
      bool has_errors = false;
      for (uint32_t j = 0; j < n; ++j) {
        uint64_t num_doubles = j - last_processed_count + 1;
        if (num_doubles >= (1 << 21) || j == n - 1) {
          num_processed_floats += num_doubles;
          last_processed_count += num_doubles;
          auto now = std::chrono::steady_clock::now();
          if (i == 0 && now - last_update_time >= std::chrono::seconds(1)) {
            last_update_time = now;
            printf("Progress: %5.2f%%\n", num_processed_floats * percent);
          }
        }

        uint32_t bits = begin + j;

        float value = 0;
        memcpy(&value, &bits, sizeof(float));

        zmij::write(actual, sizeof(actual), value);
        *jkj::dragonbox::to_chars(value, expected) = '\0';

        if (strcmp(actual, expected) == 0) continue;
        if (strcmp(actual, "0") == 0 && strcmp(expected, "0e0") == 0) continue;
        if (strcmp(actual, "-0") == 0 && strcmp(expected, "-0e0") == 0)
          continue;

        ++num_errors;
        if (!has_errors) {
          printf("Output mismatch: %s != %s\n", actual, expected);
          has_errors = true;
        }
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  auto finish = std::chrono::steady_clock::now();

  using seconds = std::chrono::duration<double>;
  printf("Tested %llu values in %.2f seconds\n", num_processed_floats.load(),
         std::chrono::duration_cast<seconds>(finish - start).count());
  return num_errors != 0 ? 1 : 0;
}
