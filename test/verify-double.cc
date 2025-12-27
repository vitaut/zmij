// A verifier for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <stdint.h>  // uint32_t
#include <stdio.h>   // printf
#include <string.h>  // memcpy

#include <atomic>
#include <thread>
#include <vector>

#include "../zmij.cc"
#include "dragonbox/dragonbox.h"

namespace {

const uint64_t pow10[] = {
    1,
    10,
    100,
    1000,
    10000,
    100000,
    1000000,
    10000000,
    100000000,
    1000000000,
    10000000000,
    100000000000,
    1000000000000,
    10000000000000,
    100000000000000,
    1000000000000000,
    10000000000000000,
};

constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
constexpr uint64_t implicit_bit = uint64_t(1) << num_sig_bits;

inline auto verify(uint64_t bits, int bin_exp) -> bool {
  uint64_t bin_sig = bits & (implicit_bit - 1);
  fp actual = to_decimal(bin_sig, bin_exp, true, false);

  double value;
  memcpy(&value, &bits, sizeof(double));
  auto expected = jkj::dragonbox::to_decimal(value);

  uint32_t abbccddee = uint32_t(actual.sig / 100'000'000);
  uint32_t ffgghhii = uint32_t(actual.sig % 100'000'000);
  int num_zeros = 0;
  if (ffgghhii == 0) {
    num_zeros = 16 - count_trailing_nonzeros(to_bcd8(abbccddee % 100'000'000));
  } else {
    num_zeros = 8 - count_trailing_nonzeros(to_bcd8(ffgghhii));
  }
  if (num_zeros != 0) {
    expected.significand *= pow10[num_zeros];
    expected.exponent -= num_zeros;
  }

  if (actual.sig == expected.significand && actual.exp == expected.exponent)
    return true;

  using ullong = unsigned long long;
  printf("Output mismatch for %.17g: %llu * 10**%d != %llu * 10**%d\n", value,
         ullong(actual.sig), actual.exp, ullong(expected.significand),
         expected.exponent);
  return false;
}

}  // namespace

auto main() -> int {
  // Verify correctness for doubles with a given binary exponent.
  constexpr int bin_exp_biased = 1;
  constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
  static constexpr uint64_t num_significands = uint64_t(1) << 32;  // test a subset
  uint64_t bits = uint64_t(bin_exp_biased) << num_sig_bits;

  constexpr int num_exp_bits = 64 - num_sig_bits - 1;
  constexpr int exp_mask = (1 << num_exp_bits) - 1;
  constexpr int exp_bias = (1 << (num_exp_bits - 1)) - 1;
  int bin_exp = bin_exp_biased;

  if (((bin_exp + 1) & exp_mask) <= 1) {
    printf("Unsupported exponent\n");
  }
  bits ^= implicit_bit;
  bin_exp -= num_sig_bits + exp_bias;

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::atomic<unsigned long long> num_processed_doubles(0);
  std::atomic<unsigned long long> num_errors(0);
  printf("Using %u threads\n", num_threads);

  auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < num_threads; ++i) {
    uint64_t begin = bits | (num_significands * i / num_threads);
    uint64_t end = bits | (num_significands * (i + 1) / num_threads);

    // Skip irregular because those are tested elsewhere.
    if (begin == 0) ++begin;
    uint64_t n = end - begin;
    threads[i] = std::thread([i, begin, n, bin_exp, &num_processed_doubles,
                              &num_errors] {
      printf("Thread %d processing 0x%013llx - 0x%013llx\n", i, begin,
             begin + n - 1);

      constexpr double percent = 100.0 / num_significands;
      uint64_t last_processed_count = 0;
      auto last_update_time = std::chrono::steady_clock::now();
      for (uint64_t j = 0; j < n; ++j) {
        uint64_t num_doubles = j - last_processed_count + 1;
        if (num_doubles >= (1 << 21) || j == n - 1) {
          num_processed_doubles += num_doubles;
          last_processed_count += num_doubles;
          if (i == 0) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_update_time >= std::chrono::seconds(1)) {
              last_update_time = now;
              printf("Progress: %7.4f%%\n", num_processed_doubles * percent);
            }
          }
        }
        if (!verify(begin + j, bin_exp)) ++num_errors;
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  auto finish = std::chrono::steady_clock::now();

  using seconds = std::chrono::duration<double>;
  printf("%llu errors in %llu values in %.2f seconds\n", num_errors.load(),
         num_processed_doubles.load(),
         std::chrono::duration_cast<seconds>(finish - start).count());
  return num_errors != 0 ? 1 : 0;
}
