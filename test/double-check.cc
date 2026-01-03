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
#include "modular-search.h"

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

using traits = float_traits<double>;

constexpr auto debias(int bin_exp_biased) -> int {
  return bin_exp_biased - (traits::num_sig_bits + traits::exp_bias);
}

inline auto verify(uint64_t bits, uint64_t bin_sig, int bin_exp,
                   bool& has_errors) -> bool {
  zmij::dec_fp actual =
      to_decimal(bin_sig, bin_exp, compute_dec_exp(bin_exp, true), true, false);

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

  if (has_errors) return false;
  has_errors = true;
  using ullong = unsigned long long;
  printf(
      "Output mismatch for %.17g (%llu * 2**%d): %llu * 10**%d != %llu * "
      "10**%d\n",
      value, bin_sig, bin_exp, ullong(actual.sig), actual.exp,
      ullong(expected.significand), expected.exponent);
  return false;
}

auto is_pow10_exact_for_bin_exp(int bin_exp) -> bool {
  int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exact_begin = -0, exact_end = 55;
  static_assert(pow10_significands[exact_begin].hi == 0x8000000000000000);
  static_assert(pow10_significands[exact_end].hi == 0xd0cf4b50cfe20765);
  return -dec_exp >= exact_begin && -dec_exp <= exact_end;
}

}  // namespace

auto main() -> int {
  // Verify correctness for doubles with a given binary exponent and
  // the first num_significands significands.
  constexpr int raw_exp = 1;
  constexpr uint64_t num_significands = uint64_t(1) << 36;

  constexpr int bin_exp = debias(raw_exp);
  if (raw_exp == 0 || raw_exp == traits::exp_mask) {
    fprintf(stderr, "Unsupported exponent\n");
    return 1;
  }
  int num_inexact_exponents = 0;
  for (int exp = 0; exp < traits::exp_mask; ++exp) {
    if (!is_pow10_exact_for_bin_exp(debias(exp))) ++num_inexact_exponents;
  }
  printf("Verifying binary exponent %d (0x%03x); %d total\n",
         bin_exp, raw_exp, num_inexact_exponents);

  constexpr int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exp_shift = compute_exp_shift(bin_exp, dec_exp);
  printf("dec_exp=%d exp_shift=%d\n", dec_exp, exp_shift);
  if (is_pow10_exact_for_bin_exp(bin_exp)) {
    printf("Power of 10 is exact for bin_exp=%d\n", bin_exp);
    return 0;
  }

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  printf("Using %u threads\n", num_threads);
  
  std::atomic<unsigned long long> num_processed_doubles(0);
  std::atomic<unsigned long long> num_special_cases(0);
  std::atomic<unsigned long long> num_errors(0);

  auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < num_threads; ++i) {
    uint64_t bin_sig_first = (num_significands * i / num_threads);
    uint64_t bin_sig_last = (num_significands * (i + 1) / num_threads) - 1;

    // Skip irregular because those are tested elsewhere.
    if (bin_sig_first == 0) ++bin_sig_first;
    bin_sig_first |= traits::implicit_bit;
    bin_sig_last |= traits::implicit_bit;

    auto fun = [i, bin_sig_first, bin_sig_last, &num_processed_doubles,
                &num_special_cases, &num_errors] {
      printf("Thread %d processing 0x%016llx - 0x%016llx\n", i, bin_sig_first,
             bin_sig_last);

      auto last_update_time = std::chrono::steady_clock::now();
      bool has_errors = false;

      constexpr uint64_t pow10_lo = pow10_significands[-dec_exp].lo;
      constexpr uint64_t exp_bits =
        uint64_t(raw_exp) << traits::num_sig_bits ^ traits::implicit_bit;

      // With great power of 10 comes great responsibility to check the
      // approximation error. The exact power of 10 significand is in the range
      // [pow10, pow10 + 1), where pow10 = (pow10_hi << 64) | pow10_lo.

      // Check for possible carry due to pow10 approximation error.
      // This checks all cases where integral and fractional can be off in
      // to_decimal. The rest is taken care of by the conservative boundary
      // checks on the fast path.
      num_special_cases += find_carried_away_doubles<pow10_lo, exp_shift>(
          bin_sig_first, bin_sig_last,
          [&](uint64_t index) {
            uint64_t bin_sig = bin_sig_first + index;
            uint64_t bits = exp_bits ^ bin_sig;
            if (!verify(bits, bin_sig, bin_exp, has_errors)) ++num_errors;
          },
          [&](uint64_t num_doubles) {
            num_processed_doubles += num_doubles;
            if (i != 0) return;
            auto now = std::chrono::steady_clock::now();
            if (now - last_update_time >= std::chrono::seconds(1)) {
              last_update_time = now;
              printf("Progress: %7.4f%%\n",
                     num_processed_doubles * 100.0 / num_significands);
            }
          });
    };
    threads[i] = std::thread(fun);
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
