#include "zmij.cc"

#include <gtest/gtest.h>

#include <string>

auto dtoa(double value) -> std::string {
  char buffer[zmij::buffer_size];
  zmij::dtoa(value, buffer);
  return buffer;
}

TEST(zmij_test, umul192_upper64_modified) {
  auto pow10 = pow10_significands[0];
  EXPECT_EQ(
      umul192_upper64_modified(pow10.hi, pow10.lo, 0x1234567890abcdef << 1),
      0x24554a3ce60a45f5);
  EXPECT_EQ(
      umul192_upper64_modified(pow10.hi, pow10.lo, 0x1234567890abce16 << 1),
      0x24554a3ce60a4643);
}

TEST(zmij_test, normal) { EXPECT_EQ(dtoa(6.62607015e-34), "6.62607015e-34"); }

TEST(zmij_test, zero) {
  EXPECT_EQ(dtoa(0), "0");
  EXPECT_EQ(dtoa(-0.0), "-0");
}

TEST(zmij_test, inf) {
  EXPECT_EQ(dtoa(std::numeric_limits<double>::infinity()), "inf");
  EXPECT_EQ(dtoa(-std::numeric_limits<double>::infinity()), "-inf");
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

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
