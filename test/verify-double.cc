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

void verify(uint64_t bits) {
  double value = 0;
  memcpy(&value, &bits, sizeof(double));

  constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
  constexpr uint64_t implicit_bit = uint64_t(1) << num_sig_bits;
  uint64_t bin_sig = bits & (implicit_bit - 1);  // binary significand
  bool regular = bin_sig != 0;

  constexpr int num_exp_bits = 64 - num_sig_bits - 1;
  constexpr int exp_mask = (1 << num_exp_bits) - 1;
  constexpr int exp_bias = (1 << (num_exp_bits - 1)) - 1;
  int bin_exp = int(bits >> num_sig_bits) & exp_mask;  // binary exponent

  bool subnormal = false;
  if (((bin_exp + 1) & exp_mask) <= 1) [[unlikely]] {
    if (bin_exp != 0) return;
    if (bin_sig == 0) return;
    // Handle subnormals.
    bin_sig |= implicit_bit;
    bin_exp = 1;
    subnormal = true;
  }
  bin_sig ^= implicit_bit;
  bin_exp -= num_sig_bits + exp_bias;

  fp actual = to_decimal(bin_sig, bin_exp, regular, subnormal);
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
    return;

  using ullong = unsigned long long;
  printf("Output mismatch for %.17g: %llu * 10**%d != %llu * 10**%d\n", value,
         ullong(actual.sig), actual.exp, ullong(expected.significand),
         expected.exponent);
}

}  // namespace

auto main() -> int {
  // Verify correctness for doubles with a given binary exponent.
  constexpr int bin_exp_biased = 1;
  constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
  constexpr uint64_t bits = uint64_t(bin_exp_biased) << num_sig_bits;
  constexpr uint64_t num_significands = uint64_t(1) << 32;  // test a subset

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::atomic<unsigned long long> num_processed_doubles(0);
  printf("Using %u threads\n", num_threads);

  auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < num_threads; ++i) {
    uint64_t begin = bits | (num_significands * i / num_threads);
    uint64_t end = bits | (num_significands * (i + 1) / num_threads);
    uint64_t n = end - begin;
    threads[i] = std::thread([i, begin, n, &num_processed_doubles] {
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
        verify(begin + j);
      }
    });
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  auto finish = std::chrono::steady_clock::now();

  using seconds = std::chrono::duration<double>;
  printf("Tested %llu values in %.2f seconds\n", num_processed_doubles.load(),
         std::chrono::duration_cast<seconds>(finish - start).count());
}
