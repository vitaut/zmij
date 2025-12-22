#include "../zmij.cc"

#include <gtest/gtest.h>

#include <math.h>
#include <stdio.h>
#include <limits>
#include <string>

auto dtoa(double value) -> std::string {
  char buffer[zmij::buffer_size];
  zmij::dtoa(value, buffer);
  return buffer;
}

TEST(zmij_test, utilities) {
  EXPECT_EQ(countl_zero(1), 63);
  EXPECT_EQ(countl_zero(~0ull), 0);

  EXPECT_EQ(count_trailing_nonzeros(0x30303030'30303030ull), 0);
  EXPECT_EQ(count_trailing_nonzeros(0x30303030'30303031ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x30303030'30303039ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x30393030'39303030ull), 7);
  EXPECT_EQ(count_trailing_nonzeros(0x31303030'30303030ull), 8);
  EXPECT_EQ(count_trailing_nonzeros(0x39303030'30303030ull), 8);
}

TEST(zmij_test, umul192_upper64_inexact_to_odd) {
  auto pow10 = pow10_significands[0];
  EXPECT_EQ(umul192_upper64_inexact_to_odd(pow10.hi, pow10.lo,
                                           0x1234567890abcdef << 1),
            0x24554a3ce60a45f5);
  EXPECT_EQ(umul192_upper64_inexact_to_odd(pow10.hi, pow10.lo,
                                           0x1234567890abce16 << 1),
            0x24554a3ce60a4643);
}

TEST(zmij_test, normal) { EXPECT_EQ(dtoa(6.62607015e-34), "6.62607015e-34"); }

TEST(zmij_test, subnormal) {
  EXPECT_EQ(dtoa(1e-323), "1.e-323");
  EXPECT_EQ(dtoa(1.2e-322), "1.2e-322");
  EXPECT_EQ(dtoa(1.5e-323), "1.5e-323");
  EXPECT_EQ(dtoa(1.24e-322), "1.24e-322");
  EXPECT_EQ(dtoa(1.234e-320), "1.234e-320");
}

TEST(zmij_test, small_int) { EXPECT_EQ(dtoa(1), "1e+00"); }

TEST(zmij_test, zero) {
  EXPECT_EQ(dtoa(0), "0");
  EXPECT_EQ(dtoa(-0.0), "-0");
}

TEST(zmij_test, inf) {
  EXPECT_EQ(dtoa(std::numeric_limits<double>::infinity()), "inf");
  EXPECT_EQ(dtoa(-std::numeric_limits<double>::infinity()), "-inf");
}

TEST(zmij_test, nan) {
  EXPECT_EQ(dtoa(std::numeric_limits<double>::quiet_NaN()), "nan");
  EXPECT_EQ(dtoa(-std::numeric_limits<double>::quiet_NaN()), "-nan");
}

TEST(zmij_test, shorter) {
  // A possibly shorter underestimate is picked (u' in Schubfach).
  EXPECT_EQ(dtoa(-4.932096661796888e-226), "-4.932096661796888e-226");

  // A possibly shorter overestimate is picked (w' in Schubfach).
  EXPECT_EQ(dtoa(3.439070283483335e+35), "3.439070283483335e+35");
}

TEST(zmij_test, single_candidate) {
  // Only an underestimate is in the rounding region (u in Schubfach).
  EXPECT_EQ(dtoa(6.606854224493745e-17), "6.606854224493745e-17");

  // Only an overestimate is in the rounding region (w in Schubfach).
  EXPECT_EQ(dtoa(6.079537928711555e+61), "6.079537928711555e+61");
}

TEST(zmij_test, all_exponents) {
  using limits = std::numeric_limits<double>;
  for (int exp = limits::min_exponent; exp < limits::max_exponent; ++exp) {
    double expected = ldexp(1, exp);
    double actual = 0;
    sscanf(dtoa(expected).c_str(), "%lg", &actual);
    EXPECT_EQ(actual, expected);
  }
}

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
