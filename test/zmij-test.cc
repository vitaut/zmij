// Tests for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

// Include zmij.cc instead of linking with the library to test multiple
// configurations without building multiple versions of the library and to test
// internal functions.
#include "../zmij.cc"

#include <gtest/gtest.h>

#include <limits>  // std::numeric_limits
#include <string>  // std::string

#include "dragonbox/dragonbox_to_chars.h"

auto dtoa(double value) -> std::string {
  char buffer[zmij::double_buffer_size] = {};
  auto n = zmij::write(buffer, sizeof(buffer), value);
  return {buffer, n};
}

auto ftoa(float value) -> std::string {
  char buffer[zmij::float_buffer_size] = {};
  auto n = zmij::write(buffer, sizeof(buffer), value);
  return {buffer, n};
}

TEST(zmij_test, utilities) {
  EXPECT_EQ(clz(1), 63);
  EXPECT_EQ(clz(~0ull), 0);

  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000000ull), 0);
  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000001ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000009ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x00090000'09000000ull), 7);
  EXPECT_EQ(count_trailing_nonzeros(0x01000000'00000000ull), 8);
  EXPECT_EQ(count_trailing_nonzeros(0x09000000'00000000ull), 8);
}

TEST(zmij_test, umul_upper_inexact_to_odd) {
  auto pow10 = pow10_significands[-292];
  EXPECT_EQ(umul_upper_inexact_to_odd(pow10.hi, pow10.lo,
                                      uint64_t(0x1234567890abcdef << 1)),
            0x24554a3ce60a45f5);
  EXPECT_EQ(umul_upper_inexact_to_odd(pow10.hi, pow10.lo,
                                      uint64_t(0x1234567890abce16 << 1)),
            0x24554a3ce60a4643);
}

TEST(dtoa_test, normal) {
  EXPECT_EQ(dtoa(6.62607015e-34), "6.62607015e-34");

  // Exact half-ulp tie when rounding to nearest integer.
  EXPECT_EQ(dtoa(5.444310685350916e+14), "5.444310685350916e+14");
}

TEST(dtoa_test, subnormal) {
  EXPECT_EQ(dtoa(std::numeric_limits<double>::denorm_min()), "5e-324");
  EXPECT_EQ(dtoa(1e-323), "1e-323");
  EXPECT_EQ(dtoa(1.2e-322), "1.2e-322");
  EXPECT_EQ(dtoa(1.5e-323), "1.5e-323");
  EXPECT_EQ(dtoa(1.24e-322), "1.24e-322");
  EXPECT_EQ(dtoa(1.234e-320), "1.234e-320");
}

TEST(dtoa_test, all_irregular) {
  for (uint64_t exp = 1; exp < 0x3ff; ++exp) {
    uint64_t bits = exp << 52;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(dtoa(value), expected);
  }
}

TEST(dtoa_test, all_exponents) {
  for (uint64_t exp = 0; exp <= 0x3ff; ++exp) {
    uint64_t bits = (exp << 52) | 1;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(dtoa(value), expected);
  }
}

TEST(dtoa_test, small_int) { EXPECT_EQ(dtoa(1), "1e+00"); }

TEST(dtoa_test, zero) {
  EXPECT_EQ(dtoa(0), "0");
  EXPECT_EQ(dtoa(-0.0), "-0");
}

TEST(dtoa_test, inf) {
  EXPECT_EQ(dtoa(std::numeric_limits<double>::infinity()), "inf");
}

TEST(dtoa_test, nan) {
  EXPECT_EQ(dtoa(-std::numeric_limits<double>::quiet_NaN()), "-nan");
}

TEST(dtoa_test, shorter) {
  // A possibly shorter underestimate is picked (u' in Schubfach).
  EXPECT_EQ(dtoa(-4.932096661796888e-226), "-4.932096661796888e-226");

  // A possibly shorter overestimate is picked (w' in Schubfach).
  EXPECT_EQ(dtoa(3.439070283483335e+35), "3.439070283483335e+35");
}

TEST(dtoa_test, single_candidate) {
  // Only an underestimate is in the rounding region (u in Schubfach).
  EXPECT_EQ(dtoa(6.606854224493745e-17), "6.606854224493745e-17");

  // Only an overestimate is in the rounding region (w in Schubfach).
  EXPECT_EQ(dtoa(6.079537928711555e+61), "6.079537928711555e+61");
}

TEST(dtoa_test, null_terminated) {
  char buffer[zmij::double_buffer_size] = {};
  zmij::write(buffer, sizeof(buffer), 9.061488e+15);
  EXPECT_STREQ(buffer, "9.061488e+15");
  zmij::write(buffer, sizeof(buffer), std::numeric_limits<double>::quiet_NaN());
  EXPECT_STREQ(buffer, "nan");
}

TEST(dtoa_test, no_buffer) {
  double value = 6.62607015e-34;
  auto n = zmij::write(nullptr, 0, value);
  std::string result(n, '\0');
  zmij::write(result.data(), n, value);
  EXPECT_EQ(result, "6.62607015e-34");
}

TEST(dtoa_test, to_decimal) {
  zmij::dec_fp dec = zmij::to_decimal(6.62607015e-34);
  EXPECT_EQ(dec.sig, 66260701500000000);
  EXPECT_EQ(dec.exp, -50);
  dec = zmij::to_decimal(-6.62607015e-34);
  EXPECT_EQ(dec.sig, -66260701500000000);
  EXPECT_EQ(dec.exp, -50);
}

TEST(dtoa_test, no_overrun) {
  char buffer[zmij::double_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  zmij::write(buffer, zmij::double_buffer_size, -1.2345678901234567e+123);
  EXPECT_EQ(buffer, std::string("-1.2345678901234567e+123"));
  EXPECT_EQ(buffer[zmij::double_buffer_size], '?');
  EXPECT_EQ(buffer[zmij::double_buffer_size - 1], '\0');
}

TEST(ftoa_test, normal) {
  EXPECT_EQ(ftoa(6.62607e-34f), "6.62607e-34");
  EXPECT_EQ(ftoa(1.342178e+08f), "1.342178e+08");
  EXPECT_EQ(ftoa(1.3421781e+08f), "1.3421781e+08");
}

TEST(ftoa_test, subnormal) {
  EXPECT_EQ(ftoa(std::numeric_limits<float>::denorm_min()), "1e-45");
}

TEST(ftoa_test, no_buffer) {
  float value = 6.62607e-34;
  auto n = zmij::write(nullptr, 0, value);
  std::string result(n, '\0');
  zmij::write(result.data(), n, value);
  EXPECT_EQ(result, "6.62607e-34");
}

TEST(ftoa_test, no_overrun) {
  char buffer[zmij::float_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  zmij::write(buffer, zmij::float_buffer_size, -1.00000005e+15f);
  EXPECT_EQ(buffer, std::string("-1.00000005e+15"));
  EXPECT_EQ(buffer[zmij::float_buffer_size], '?');
  EXPECT_EQ(buffer[zmij::float_buffer_size - 1], '\0');
}

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
