// Tests for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "modular-search.h"

#include <gtest/gtest.h>

TEST(modular_search_test, find_min_n) {
  // (12345 * 0) % 100000 is in [0, 1000].
  EXPECT_EQ((find_min_n<12345, 100000>(0, 1000)), 0);

  // (1 * 500) % 100000 = 500, which is in [400, 600].
  EXPECT_EQ((find_min_n<500, 1000>(400, 500)), 1);

  // Multiple steps without modulus wrap: 10, 20, 30, 40, 50 (hit).
  EXPECT_EQ((find_min_n<10, 100>(45, 55)), 5);

  // Case requiring the Euclidean reduction (a > upper).
  // Steps:
  // n=1: 300 (no)
  // n=2: 600 (no)
  // n=3: 900 (no)
  // n=4: 1200 % 1000 = 200 (hit)
  EXPECT_EQ((find_min_n<300, 1000>(100, 200)), 4);

  // Not found because step and mod are even while the target is odd.
  EXPECT_EQ((find_min_n<2, 100>(5, 5)), not_found);
}

TEST(modular_search_test, find_min_n_significand_search) {
  constexpr uint64_t bin_sig_begin = (uint64_t(1) << 52) | 1;
  constexpr uint64_t bin_sig_end = (uint64_t(1) << 53) - 1;
  constexpr uint64_t pow10_lo = 0x6c07a2c26a8346d1;
  constexpr int exp_shift = 3;

  uint64_t start = (bin_sig_begin << exp_shift) * pow10_lo;
  constexpr uint64_t step = (1 << exp_shift) * pow10_lo;
  uint64_t threshold = ~uint64_t() - (bin_sig_end << exp_shift) + 1;

  uint128_t lower = threshold - start;
  uint128_t upper = ~uint64_t() - start;

  uint64_t n = find_min_n<step, uint128_t(1) << 64>(lower, upper);

  // Verify the hit: (start + n * step) should be >= threshold.
  EXPECT_GE(start + n * step, threshold);

  // Verify it is the smallest n: (x0 + (n - 1) * step) should be < threshold.
  EXPECT_GT(n, 0);
  EXPECT_LT(start + (n - 1) * step, threshold);
}

TEST(modular_search_test, find_min_n_overflow) {
  uint64_t n = find_min_n<0x6000000000000001ULL, (uint128_t(1) << 64)>(
      0xFFFFFFFFFFFFFF00ULL, 0xFFFFFFFFFFFFFFFFULL);
  EXPECT_EQ(n, 0x1fffffffffffff05);
}

TEST(modular_search_test, find_carried_away_doubles) {
  constexpr uint64_t implicit_bit = uint64_t(1) << 52;
  uint64_t bin_sig_begin = implicit_bit | uint64_t(1);
  uint64_t bin_sig_end = implicit_bit | uint64_t(1) << 20;

  constexpr uint64_t pow10_lo = 0x6c07a2c26a8346d1;
  constexpr int exp_shift = 3;

  uint64_t scaled_sig_lo = pow10_lo * (bin_sig_begin << exp_shift);
  constexpr uint64_t step = (1 << exp_shift) * pow10_lo;

  std::vector<uint64_t> expected;
  for (uint64_t bin_sig = bin_sig_begin; bin_sig < bin_sig_end;
       ++bin_sig, scaled_sig_lo += step) {
    uint64_t bin_sig_shifted = bin_sig << exp_shift;
    bool carry = scaled_sig_lo + bin_sig_shifted < scaled_sig_lo;
    if (carry) expected.push_back(bin_sig);
  }

  std::vector<uint64_t> actual;
  find_carried_away_doubles<pow10_lo, exp_shift>(
      bin_sig_begin, bin_sig_end,
      [&](uint64_t i) { actual.push_back(bin_sig_begin + i); });

  EXPECT_EQ(expected.size(), 2048);
  EXPECT_EQ(expected, actual);
}

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
