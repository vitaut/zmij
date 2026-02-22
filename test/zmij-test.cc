// Tests for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

// Include zmij.cc instead of linking with the library to test multiple
// configurations without building multiple versions of the library and to test
// internal functions.
#ifndef ZMIJ_C
#define ZMIJ_C 0
#include "../zmij.cc"
#else
#define _Alignas(x) alignas(x)
#include "../zmij.c"

namespace zmij {
enum {
  double_buffer_size = 25,
  float_buffer_size = 16,
};

auto write(char* out, size_t n, double value) noexcept -> size_t {
  return zmij_write_double(out, n, value);
}
auto write(char* out, size_t n, float value) noexcept -> size_t {
  return zmij_write_float(out, n, value);
}
}
#endif

#include <gtest/gtest.h>

#include <limits>  // std::numeric_limits
#include <string>  // std::string

#include "dragonbox/dragonbox_to_chars.h"


auto dtoa(double value) -> std::string {
  char buffer[zmij::double_buffer_size + 1] = {};
  memset(buffer, '?', sizeof(buffer));
  auto n = zmij::write(buffer + 1, sizeof(buffer), value);
  if (buffer[0] != '?') throw std::runtime_error("buffer underrun");
  return {buffer + 1, n};
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

TEST(dtoa_test, normal) {
  EXPECT_EQ(dtoa(6.62607015e-34), "6.62607015e-34");

  // Exact half-ulp tie when rounding to nearest integer.
  EXPECT_EQ(dtoa(5.444310685350916e+14), "544431068535091.6");
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
  const char* fixed[] = {
    "0.0001220703125",
    "0.000244140625",
    "0.00048828125",
    "0.0009765625",
    "0.001953125",
    "0.00390625",
    "0.0078125",
    "0.015625",
    "0.03125",
    "0.0625",
    "0.125",
    "0.25",
    "0.5"
  };
  for (uint64_t exp = 1; exp < 0x3ff; ++exp) {
    uint64_t bits = exp << 52;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    int fixed_start = 1010, fixed_end = 1022;
    if (exp >= fixed_start && exp <= fixed_end) {
      EXPECT_EQ(dtoa(value), fixed[exp - fixed_start]);
      continue;
    }

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(dtoa(value), expected) << exp;
  }
}

TEST(dtoa_test, all_exponents) {
  const char* fixed[] = {
    "0.00012207031250000003",
    "0.00024414062500000005",
    "0.0004882812500000001",
    "0.0009765625000000002",
    "0.0019531250000000004",
    "0.003906250000000001",
    "0.007812500000000002",
    "0.015625000000000003",
    "0.03125000000000001",
    "0.06250000000000001",
    "0.12500000000000003",
    "0.25000000000000006",
    "0.5000000000000001",
    "1.0000000000000002"
  };
  for (uint64_t exp = 0; exp <= 0x3ff; ++exp) {
    uint64_t bits = (exp << 52) | 1;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    int fixed_start = 1010, fixed_end = 1023;
    if (exp >= fixed_start && exp <= fixed_end) {
      EXPECT_EQ(dtoa(value), fixed[exp - fixed_start]);
      continue;
    }

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(dtoa(value), expected) << exp;
  }
}

TEST(dtoa_test, small_int) { EXPECT_EQ(dtoa(1), "1"); }

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

#if !ZMIJ_C
TEST(dtoa_test, no_buffer) {
  double value = 6.62607015e-34;
  auto n = zmij::write(nullptr, 0, value);
  std::string result(n, '\0');
  zmij::write(&result[0], n, value);
  EXPECT_EQ(result, "6.62607015e-34");
}

TEST(ftoa_test, no_buffer) {
  float value = 6.62607e-34;
  auto n = zmij::write(nullptr, 0, value);
  std::string result(n, '\0');
  zmij::write(&result[0], n, value);
  EXPECT_EQ(result, "6.62607e-34");
}

TEST(dtoa_test, to_decimal) {
  zmij::dec_fp dec = zmij::to_decimal(6.62607015e-34);
  EXPECT_EQ(dec.sig, 66260701500000000);
  EXPECT_EQ(dec.exp, -50);
  EXPECT_EQ(dec.negative, false);

  dec = zmij::to_decimal(-6.62607015e-34);
  EXPECT_EQ(dec.sig, 66260701500000000);
  EXPECT_EQ(dec.exp, -50);
  EXPECT_EQ(dec.negative, true);

  dec = zmij::to_decimal(-0.0);
  EXPECT_EQ(dec.sig, 0);
  EXPECT_EQ(dec.exp, 0);
  EXPECT_EQ(dec.negative, true);

  uint32_t garlic = 0;
  memcpy(&garlic, "🧄", 4);
  uint64_t bits = 0x7FF0000000000000 | garlic;
  double garlic_nan = 0;
  memcpy(&garlic_nan, &bits, sizeof(bits));
  dec = zmij::to_decimal(garlic_nan);
  EXPECT_EQ(dec.sig, garlic);
}
#endif  // ZMIJ_C

TEST(dtoa_test, no_overrun) {
  char buffer[zmij::double_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  auto n = zmij::write(buffer, zmij::double_buffer_size, -1.2345678901234567e+123);
  EXPECT_EQ(std::string(buffer, n), std::string("-1.2345678901234567e+123"));
  EXPECT_EQ(buffer[zmij::double_buffer_size], '?');
  //EXPECT_EQ(buffer[zmij::double_buffer_size - 1], '?');
}

TEST(dtoa_test, no_underrun) {
  dtoa(9.061488e+15);
}

TEST(ftoa_test, normal) {
  EXPECT_EQ(ftoa(6.62607e-34f), "6.62607e-34");
  EXPECT_EQ(ftoa(1.342178e+08f), "1.342178e+08");
  EXPECT_EQ(ftoa(1.3421781e+08f), "1.3421781e+08");
}

TEST(ftoa_test, subnormal) {
  EXPECT_EQ(ftoa(std::numeric_limits<float>::denorm_min()), "1e-45");
}

TEST(ftoa_test, no_overrun) {
  char buffer[zmij::float_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  auto n = zmij::write(buffer, zmij::float_buffer_size, -1.00000005e+15f);
  EXPECT_EQ(std::string(buffer, n), std::string("-1.00000005e+15"));
  EXPECT_EQ(buffer[zmij::float_buffer_size], '?');
  //EXPECT_EQ(buffer[zmij::float_buffer_size - 1], '?');
}

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
