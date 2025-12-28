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

// clang-format off
const uint64_t pow10[] = {
                  1,
                 1'0,
                1'0'0,
               1'0'0'0,
              1'0'0'0'0,
             1'0'0'0'0'0,
            1'0'0'0'0'0'0,
           1'0'0'0'0'0'0'0,
          1'0'0'0'0'0'0'0'0,
         1'0'0'0'0'0'0'0'0'0,
        1'0'0'0'0'0'0'0'0'0'0,
       1'0'0'0'0'0'0'0'0'0'0'0,
      1'0'0'0'0'0'0'0'0'0'0'0'0,
     1'0'0'0'0'0'0'0'0'0'0'0'0'0,
    1'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
   1'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
  1'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
};
// clang-format on

constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
constexpr uint64_t implicit_bit = uint64_t(1) << num_sig_bits;
constexpr int num_exp_bits = 64 - num_sig_bits - 1;
constexpr int exp_bias = (1 << (num_exp_bits - 1)) - 1;

constexpr auto debias(int bin_exp_biased) -> int {
  return bin_exp_biased - (num_sig_bits + exp_bias);
}

inline auto verify(uint64_t bits, uint64_t bin_sig, int bin_exp) -> bool {
  fp actual = to_decimal(bin_sig, bin_exp, true, false);

  double value;
  memcpy(&value, &bits, sizeof(double));
  auto expected = jkj::dragonbox::to_decimal(value);

  uint32_t abbccddee = uint32_t(actual.sig / 100'000'000);
  uint32_t ffgghhii = uint32_t(actual.sig % 100'000'000);
  int num_zeros = 0;
  if (ffgghhii == 0)
    num_zeros = 16 - count_trailing_nonzeros(to_bcd8(abbccddee % 100'000'000));
  else
    num_zeros = 8 - count_trailing_nonzeros(to_bcd8(ffgghhii));
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

auto is_pow10_exact_for_bin_exp(int bin_exp) -> bool {
  int dec_exp = compute_dec_exp(bin_exp, true);
  int pow10_index = -dec_exp - dec_exp_min;
  constexpr int exact_begin = 292, exact_end = 347;
  static_assert(pow10_significands[exact_begin].hi == 0x8000000000000000);
  static_assert(pow10_significands[exact_end].hi == 0xd0cf4b50cfe20765);
  return pow10_index >= exact_begin && pow10_index <= exact_end;
}

}  // namespace

auto main() -> int {
  // Verify correctness for doubles with a given binary exponent.
  constexpr int bin_exp_biased = 1;
  constexpr int num_sig_bits = std::numeric_limits<double>::digits - 1;
  static constexpr uint64_t num_significands = uint64_t(1)
                                               << 33;  // test a subset

  constexpr int exp_mask = (1 << num_exp_bits) - 1;
  if (bin_exp_biased == 0 || bin_exp_biased == exp_mask)
    printf("Unsupported exponent\n");

  constexpr uint64_t bits = (uint64_t(bin_exp_biased) << num_sig_bits);
  constexpr int bin_exp = debias(bin_exp_biased);
  constexpr int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exp_shift = compute_exp_shift(bin_exp, dec_exp);

  if (is_pow10_exact_for_bin_exp(bin_exp)) {
    printf("Power of 10 is exact for bin_exp=%d dec_exp=%d\n", bin_exp,
           dec_exp);
    return 0;
  }
  constexpr int pow10_index = -dec_exp - dec_exp_min;
  constexpr uint64_t pow10_lo = pow10_significands[pow10_index].lo;

  int num_inexact_exponents = 0;
  for (int exp = 0; exp < exp_mask; ++exp) {
    if (!is_pow10_exact_for_bin_exp(debias(exp))) ++num_inexact_exponents;
  }
  printf("Need to verify %d exponents\n", num_inexact_exponents);

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  std::atomic<unsigned long long> num_processed_doubles(0);
  std::atomic<unsigned long long> num_special_cases(0);
  std::atomic<unsigned long long> num_errors(0);
  printf("Using %u threads\n", num_threads);

  auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < num_threads; ++i) {
    uint64_t begin = (num_significands * i / num_threads);
    uint64_t end = (num_significands * (i + 1) / num_threads);

    // Skip irregular because those are tested elsewhere.
    if (begin == 0) ++begin;
    begin |= bits;
    end |= bits;
    uint64_t n = end - begin;
    threads[i] = std::thread([i, begin, n, bin_exp, &num_processed_doubles,
                              &num_special_cases, &num_errors] {
      printf("Thread %d processing 0x%013llx - 0x%013llx\n", i, begin,
             begin + n - 1);

      constexpr double percent = 100.0 / num_significands;
      uint64_t last_processed_count = 0;
      auto last_update_time = std::chrono::steady_clock::now();
      unsigned long long num_special_cases_local = 0;
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

        // The real power of 10 is in the range [pow10, pow10 + 1) ignoring
        // the exponent, where pow10 = (pow10_hi << 64) | pow10_lo.

        // Check for possible carry due to pow10 approximation error.
        // This checks all cases where integral and fractional can be off in
        // to_decimal. The rest is taken care of by the conservative boundary
        // checks on the fast path.
        uint64_t bin_sig = ((begin + j) & (implicit_bit - 1)) | implicit_bit;
        uint64_t bin_sig_shifted = bin_sig << exp_shift;
        uint64_t scaled_sig_lo = pow10_lo * bin_sig_shifted;
        bool carry = scaled_sig_lo + bin_sig_shifted < scaled_sig_lo;
        if (!carry) continue;
        ++num_special_cases_local;

        if (!verify(begin + j, bin_sig, bin_exp)) ++num_errors;
      }
      num_special_cases += num_special_cases_local;
    });
  }
  for (int i = 0; i < num_threads; ++i) threads[i].join();
  auto finish = std::chrono::steady_clock::now();

  using seconds = std::chrono::duration<double>;
  printf("%llu errors and %llu special cases in %llu values in %.2f seconds\n",
         num_errors.load(), num_special_cases.load(),
         num_processed_doubles.load(),
         std::chrono::duration_cast<seconds>(finish - start).count());
  return num_errors != 0 ? 1 : 0;
}
