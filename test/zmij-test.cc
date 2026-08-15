// Tests for https://github.com/vitaut/zmij/.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#ifndef ZMIJ_C
#  include "zmij.h"

#  include "zmij-to-chars.h"
#  define ZMIJ_C 0
#else
extern "C" {
#  include "zmij-c.h"
}

namespace zmij {
enum {
  float_buffer_size = zmij_float_buffer_size,
  double_buffer_size = zmij_double_buffer_size,
};

auto write(char* out, size_t n, double value) noexcept -> char* {
  return zmij_write_double(out, n, value);
}
auto write(char* out, size_t n, float value) noexcept -> char* {
  return zmij_write_float(out, n, value);
}
}  // namespace zmij
#endif

#include <gtest/gtest.h>
#include <stdint.h>  // uint64_t
#include <stdio.h>   // snprintf
#include <stdlib.h>  // atoi, strtold

#include <cmath>   // std::ldexp, std::isinf
#include <limits>  // std::numeric_limits
#include <string>  // std::string

#include "dragonbox/dragonbox_to_chars.h"
#include "fmt/format.h"

typedef long double long_double;

auto to_shortest(double value) -> std::string {
  char buffer[zmij::double_buffer_size + 1] = {};
  memset(buffer, '?', sizeof(buffer));
  auto end = zmij::write(buffer + 1, sizeof(buffer), value);
  if (buffer[0] != '?') throw std::runtime_error("buffer underrun");
  return {buffer + 1, end};
}

auto to_shortest(float value) -> std::string {
  char buffer[zmij::float_buffer_size] = {};
  auto end = zmij::write(buffer, sizeof(buffer), value);
  return {buffer, end};
}

TEST(float_test, normal) {
  EXPECT_EQ(to_shortest(6.62607e-34f), "6.62607e-34");
  EXPECT_EQ(to_shortest(1.342178e+08f), "1.342178e+08");
  EXPECT_EQ(to_shortest(1.3421781e+08f), "1.3421781e+08");
}

TEST(float_test, subnormal) {
  EXPECT_EQ(to_shortest(std::numeric_limits<float>::denorm_min()), "1e-45");
}

TEST(float_test, no_overrun) {
  char buffer[zmij::float_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  auto end = zmij::write(buffer, zmij::float_buffer_size, -1.00000005e+15f);
  EXPECT_EQ(std::string(buffer, end), std::string("-1.00000005e+15"));
  EXPECT_EQ(buffer[zmij::float_buffer_size], '?');
}

TEST(float_test, no_buffer) {
  float value = 6.62607e-34;
  char buffer[zmij::float_buffer_size];
  auto end = zmij::write(buffer, sizeof(buffer), value);
  std::string result(buffer, end);
  EXPECT_EQ(result, "6.62607e-34");
}

TEST(float_test, fixed_with_zeros) {
  EXPECT_EQ(to_shortest(43210.0f), "43210");
  EXPECT_EQ(to_shortest(43210.1f), "43210.1");
  EXPECT_EQ(to_shortest(10000.f), "10000");
}

#if !ZMIJ_C
// Writes `value` with `precision` digits after the point in scientific format.
static auto to_scientific(float value, int precision) -> std::string {
  char buffer[zmij::buffer_sizes<float>::scientific];
  return {buffer,
          zmij::write_scientific(buffer, sizeof(buffer), value, precision)};
}
static auto to_scientific(double value, int precision) -> std::string {
  char buffer[zmij::buffer_sizes<double>::scientific];
  return {buffer,
          zmij::write_scientific(buffer, sizeof(buffer), value, precision)};
}

// Writes `value` with up to `precision` significant digits in general format.
static auto to_general(float value, int precision) -> std::string {
  char buffer[zmij::buffer_sizes<float>::scientific];
  return {buffer,
          zmij::write_general(buffer, sizeof(buffer), value, precision)};
}
static auto to_general(double value, int precision) -> std::string {
  char buffer[zmij::buffer_sizes<double>::scientific];
  return {buffer,
          zmij::write_general(buffer, sizeof(buffer), value, precision)};
}

TEST(float_test, to_chars) {
  char buffer[zmij::float_buffer_size];
  auto result = zmij::to_chars(buffer, buffer + sizeof(buffer), 6.62607e-34f);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "6.62607e-34");

  // Too small: truncated output, ptr == last, value_too_large.
  char small[3] = {'?', '?', '?'};
  result = zmij::to_chars(small, small + 2, 1.25f);
  EXPECT_EQ(result.ec, std::errc::value_too_large);
  EXPECT_EQ(result.ptr, small + 2);
  EXPECT_EQ(std::string(small, sizeof(small)), "1.?");
}

TEST(float_test, to_decimal) {
  zmij::dec_fp dec = zmij::to_decimal(6.62607e-34f);
  EXPECT_EQ(dec.sig, 66260700);
  EXPECT_EQ(dec.exp, -41);
  EXPECT_EQ(dec.negative, false);

  dec = zmij::to_decimal(-1.5f);
  EXPECT_EQ(dec.sig, 15000000);
  EXPECT_EQ(dec.exp, -7);
  EXPECT_EQ(dec.negative, true);

  dec = zmij::to_decimal(-0.0f);
  EXPECT_EQ(dec.sig, 0);
  EXPECT_EQ(dec.exp, 0);
  EXPECT_EQ(dec.negative, true);

  uint32_t garlic = 0;
  memcpy(&garlic, "🧄", 4);
  uint32_t bits = 0x7F800000 | (garlic & 0x7FFFFF);
  float garlic_nan = 0;
  memcpy(&garlic_nan, &bits, sizeof(bits));
  dec = zmij::to_decimal(garlic_nan);
  EXPECT_EQ(dec.exp, zmij::non_finite_exp);
  EXPECT_EQ(dec.sig, garlic & 0x7FFFFF);
}

TEST(float_test, to_chars_format) {
  char buffer[zmij::buffer_sizes<float>::fixed];
  auto result = zmij::to_chars(buffer, buffer + sizeof(buffer), 1.5f,
                               zmij::chars_format::scientific, 2);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "1.50e+00");

  // Matching printf, `general` treats precision 0 as 1 significant digit.
  result = zmij::to_chars(buffer, buffer + sizeof(buffer), 1.5f,
                          zmij::chars_format::general, 0);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "2");

  // Large precision routes through the exact writer and matches printf (which
  // promotes the float to the same double value).
  char big[64], ref[64];
  result = zmij::to_chars(big, big + sizeof(big), 0.1f,
                          zmij::chars_format::scientific, 20);
  EXPECT_EQ(result.ec, std::errc());
  snprintf(ref, sizeof(ref), "%.20e", 0.1f);
  EXPECT_EQ(std::string(big, result.ptr), std::string(ref));

  // `hex` writes `precision` fractional hex digits (no 0x prefix); a negative
  // precision selects the shortest form.
  result = zmij::to_chars(buffer, buffer + sizeof(buffer), 1.5f,
                          zmij::chars_format::hex, 10);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "1.8000000000p+0");
  result = zmij::to_chars(buffer, buffer + sizeof(buffer), 1.5f,
                          zmij::chars_format::hex, -1);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "1.8p+0");

  // Format without precision writes the shortest round-tripping form, same as
  // the double path (see double_test.to_chars_format for the %g rule).
  auto shortest = [&](zmij::chars_format f, float value) {
    auto r = zmij::to_chars(buffer, buffer + sizeof(buffer), value, f);
    EXPECT_EQ(r.ec, std::errc());
    return std::string(buffer, r.ptr);
  };
  EXPECT_EQ(shortest(zmij::chars_format::hex, 1.5f), "1.8p+0");
  EXPECT_EQ(shortest(zmij::chars_format::scientific, 1.5f), "1.5e+00");
  EXPECT_EQ(shortest(zmij::chars_format::fixed, 0.0001f), "0.0001");
  EXPECT_EQ(shortest(zmij::chars_format::general, 100.0f), "1e+02");
  EXPECT_EQ(shortest(zmij::chars_format::general, 1234567.0f), "1234567");
}

TEST(float_test, write_precision) {
  EXPECT_EQ(to_scientific(1.5f, 1), "1.5e+00");
  EXPECT_EQ(to_scientific(9.99f, 1), "1.0e+01");   // carry
  EXPECT_EQ(to_scientific(2.5f, 0), "2e+00");      // round half to even
  EXPECT_EQ(to_scientific(-1.5f, 1), "-1.5e+00");  // sign preserved
  EXPECT_EQ(to_scientific(std::numeric_limits<float>::denorm_min(), 0),
            "1e-45");  // subnormal path
  EXPECT_EQ(to_scientific(std::numeric_limits<float>::max(), 8),
            "3.40282347e+38");
}

// Big precision (> 18) routes write_scientific, write_general, and write_fixed
// through write_big; all must match printf's %e, %g, and %f.
TEST(float_test, write_big) {
  auto check = [](float value, int precision) {
    char buf[200], ref[200];
    char* end = zmij::write_scientific(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*e", precision, double(value));
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "scientific value=" << value << " precision=" << precision;
    end = zmij::write_general(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*g", precision, double(value));
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "general value=" << value << " precision=" << precision;
    end = zmij::write_fixed(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*f", precision, double(value));
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "fixed value=" << value << " precision=" << precision;
  };
  const float values[] = {1.0f, 0.1f, 1.5f, 3.14159f, 3.4028235e38f, 1.4e-45f};
  for (float value : values) {
    for (int precision : {19, 20, 30, 50, 100}) check(value, precision);
  }
}
#endif  // !ZMIJ_C

TEST(double_test, normal) {
  EXPECT_EQ(to_shortest(6.62607015e-34), "6.62607015e-34");

  // Exact half-ulp tie when rounding to nearest integer.
  EXPECT_EQ(to_shortest(5.444310685350916e+14), "544431068535091.6");
}

TEST(double_test, subnormal) {
  EXPECT_EQ(to_shortest(std::numeric_limits<double>::denorm_min()), "5e-324");
  EXPECT_EQ(to_shortest(1e-323), "1e-323");
  EXPECT_EQ(to_shortest(1.2e-322), "1.2e-322");
  EXPECT_EQ(to_shortest(1.5e-323), "1.5e-323");
  EXPECT_EQ(to_shortest(1.24e-322), "1.24e-322");
  EXPECT_EQ(to_shortest(1.234e-320), "1.234e-320");
  EXPECT_EQ(to_shortest(2.2250738585072004e-308), "2.2250738585072004e-308");
}

TEST(double_test, irregular) {
  const char* fixed[] = {"0.0001220703125",
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
                         "0.5"};
  for (uint64_t exp = 1; exp < 0x3ff; ++exp) {
    uint64_t bits = exp << 52;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    int fixed_start = 1010, fixed_end = 1022;
    if (exp >= fixed_start && exp <= fixed_end) {
      EXPECT_EQ(to_shortest(value), fixed[exp - fixed_start]);
      continue;
    }

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(to_shortest(value), expected) << exp;
  }
}

TEST(double_test, exponents) {
  const char* fixed[] = {"0.00012207031250000003", "0.00024414062500000005",
                         "0.0004882812500000001",  "0.0009765625000000002",
                         "0.0019531250000000004",  "0.003906250000000001",
                         "0.007812500000000002",   "0.015625000000000003",
                         "0.03125000000000001",    "0.06250000000000001",
                         "0.12500000000000003",    "0.25000000000000006",
                         "0.5000000000000001",     "1.0000000000000002"};
  for (uint64_t exp = 0; exp <= 0x3ff; ++exp) {
    uint64_t bits = (exp << 52) | 1;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));

    int fixed_start = 1010, fixed_end = 1023;
    if (exp >= fixed_start && exp <= fixed_end) {
      EXPECT_EQ(to_shortest(value), fixed[exp - fixed_start]);
      continue;
    }

    char expected[32] = {};
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    EXPECT_EQ(to_shortest(value), expected) << exp;
  }
}

TEST(double_test, small_int) { EXPECT_EQ(to_shortest(1.0), "1"); }

TEST(double_test, zero) {
  EXPECT_EQ(to_shortest(0.0), "0");
  EXPECT_EQ(to_shortest(-0.0), "-0");
}

TEST(double_test, inf) {
  EXPECT_EQ(to_shortest(std::numeric_limits<double>::infinity()), "inf");
}

TEST(double_test, nan) {
  EXPECT_EQ(to_shortest(-std::numeric_limits<double>::quiet_NaN()), "-nan");
}

TEST(double_test, shorter) {
  // A possibly shorter underestimate is picked (u' in Schubfach).
  EXPECT_EQ(to_shortest(-4.932096661796888e-226), "-4.932096661796888e-226");

  // A possibly shorter overestimate is picked (w' in Schubfach).
  EXPECT_EQ(to_shortest(3.439070283483335e+35), "3.439070283483335e+35");
}

TEST(double_test, single_candidate) {
  // Only an underestimate is in the rounding region (u in Schubfach).
  EXPECT_EQ(to_shortest(6.606854224493745e-17), "6.606854224493745e-17");

  // Only an overestimate is in the rounding region (w in Schubfach).
  EXPECT_EQ(to_shortest(6.079537928711555e+61), "6.079537928711555e+61");
}

// Rounding-boundary doubles enumerated by verify.py (see --dump-boundaries).
// boundary-bits.h is a bare initializer list, one bit pattern per line.
static const uint64_t boundary_bits[] = {
#include "boundary-bits.h"
};

// Check zmij against dragonbox on every rounding-boundary double verify.py
// enumerates, using dragonbox's to_decimal as an independent oracle.
TEST(double_test, boundaries) {
  auto to_string = [](uint64_t sig, int dec_exp) -> std::string {
    std::string digits = std::to_string(sig);
    int num_digits = int(digits.size());
    dec_exp += num_digits - 1;           // exponent of the leading digit
    if (dec_exp < -4 || dec_exp > 15) {  // scientific
      std::string sig_str = num_digits == 1
                                ? digits
                                : digits.substr(0, 1) + "." + digits.substr(1);
      return sig_str + fmt::format("e{:+03d}", dec_exp);
    }
    int point = dec_exp + 1;  // digits left of the decimal point
    if (point <= 0) return "0." + std::string(-point, '0') + digits;
    if (point >= num_digits)
      return digits + std::string(point - num_digits, '0');
    return digits.substr(0, point) + "." + digits.substr(point);
  };

  for (uint64_t bits : boundary_bits) {
    double value = 0;
    memcpy(&value, &bits, sizeof(value));
    auto ref = jkj::dragonbox::to_decimal(value);
    EXPECT_EQ(to_shortest(value), to_string(ref.significand, ref.exponent))
        << "bits=" << bits;
  }
}

TEST(double_test, fixed_with_zeros) {
  EXPECT_EQ(to_shortest(43210.0), "43210");
  EXPECT_EQ(to_shortest(43210.1), "43210.1");
  EXPECT_EQ(to_shortest(10000.0), "10000");
  EXPECT_EQ(to_shortest(-5942736479622170.0), "-5942736479622170");
}

TEST(double_test, no_overrun) {
  char buffer[zmij::double_buffer_size + 1];
  memset(buffer, '?', sizeof(buffer));
  auto end =
      zmij::write(buffer, zmij::double_buffer_size, -1.2345678901234567e+123);
  EXPECT_EQ(std::string(buffer, end), std::string("-1.2345678901234567e+123"));
  EXPECT_EQ(buffer[zmij::double_buffer_size], '?');
}

TEST(double_test, no_underrun) { to_shortest(9.061488e+15); }

TEST(double_test, no_buffer) {
  double value = 6.62607015e-34;
  char buffer[zmij::double_buffer_size];
  auto end = zmij::write(buffer, sizeof(buffer), value);
  std::string result(buffer, end);
  EXPECT_EQ(result, "6.62607015e-34");
}

#if !ZMIJ_C
TEST(double_test, to_chars) {
  char buffer[zmij::double_buffer_size];
  auto result = zmij::to_chars(buffer, buffer + sizeof(buffer), 6.62607015e-34);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "6.62607015e-34");

  // Exact fit succeeds ("1.25" is 4 characters).
  result = zmij::to_chars(buffer, buffer + 4, 1.25);
  EXPECT_EQ(result.ec, std::errc());
  EXPECT_EQ(std::string(buffer, result.ptr), "1.25");

  // Too small: truncated output, ptr == last, value_too_large.
  char small[3] = {'?', '?', '?'};
  result = zmij::to_chars(small, small + 2, 1.25);
  EXPECT_EQ(result.ec, std::errc::value_too_large);
  EXPECT_EQ(result.ptr, small + 2);
  EXPECT_EQ(std::string(small, sizeof(small)), "1.?");
}

TEST(double_test, to_chars_format) {
  char buffer[zmij::buffer_sizes<double>::fixed];
  auto fmt = [&](zmij::chars_format f, int precision, double value) {
    auto r =
        zmij::to_chars(buffer, buffer + sizeof(buffer), value, f, precision);
    EXPECT_EQ(r.ec, std::errc());
    return std::string(buffer, r.ptr);
  };
  EXPECT_EQ(fmt(zmij::chars_format::fixed, 2, 1.5), "1.50");
  EXPECT_EQ(fmt(zmij::chars_format::fixed, 0, 2.5), "2");  // ties to even
  EXPECT_EQ(fmt(zmij::chars_format::scientific, 4, 1234.5678), "1.2346e+03");
  EXPECT_EQ(fmt(zmij::chars_format::scientific, 0, 2.5), "2e+00");
  EXPECT_EQ(fmt(zmij::chars_format::general, 6, 1234.5678), "1234.57");

  // Matching printf: `general` treats precision 0 as 1, and a negative
  // precision defaults to 6.
  EXPECT_EQ(fmt(zmij::chars_format::general, 0, 1234.5678), "1e+03");
  EXPECT_EQ(fmt(zmij::chars_format::fixed, -1, 1.5), "1.500000");
  EXPECT_EQ(fmt(zmij::chars_format::scientific, -1, 1.5), "1.500000e+00");

  // `hex` writes `precision` fractional hex digits (no 0x prefix); a negative
  // precision selects the shortest form. Specials keep their "inf"/"nan"
  // spelling (no prefix to strip).
  EXPECT_EQ(fmt(zmij::chars_format::hex, 10, 1.5), "1.8000000000p+0");
  EXPECT_EQ(fmt(zmij::chars_format::hex, 0, -2.0), "-1p+1");
  EXPECT_EQ(fmt(zmij::chars_format::hex, 6, 0.0), "0.000000p+0");
  EXPECT_EQ(fmt(zmij::chars_format::hex, -1, 1.5), "1.8p+0");  // shortest
  EXPECT_EQ(fmt(zmij::chars_format::hex, 6,
                std::numeric_limits<double>::infinity()),
            "inf");
  EXPECT_EQ(fmt(zmij::chars_format::hex, 6,
                -std::numeric_limits<double>::quiet_NaN()),
            "-nan");

  // Format without precision writes the shortest round-tripping form.
  auto shortest = [&](zmij::chars_format f, double value) {
    auto r = zmij::to_chars(buffer, buffer + sizeof(buffer), value, f);
    EXPECT_EQ(r.ec, std::errc());
    return std::string(buffer, r.ptr);
  };
  EXPECT_EQ(shortest(zmij::chars_format::hex, 1.5), "1.8p+0");
  EXPECT_EQ(shortest(zmij::chars_format::scientific, 1.5), "1.5e+00");
  EXPECT_EQ(shortest(zmij::chars_format::fixed, 1.5), "1.5");

  // `general` without precision follows the printf %g rule with the precision
  // set to the shortest significant-digit count, as the standard requires:
  // fixed when the leading exponent is in [-4, num_sig), else scientific. Some
  // libc++ builds instead apply %g's default precision of 6, so std::to_chars
  // there disagrees, e.g. printing "100" and "1.234567e+06" for these cases.
  EXPECT_EQ(shortest(zmij::chars_format::general, 100.0), "1e+02");
  EXPECT_EQ(shortest(zmij::chars_format::general, 1234567.0), "1234567");
  EXPECT_EQ(shortest(zmij::chars_format::general, 0.0001), "0.0001");
  EXPECT_EQ(shortest(zmij::chars_format::general, 1.5), "1.5");

  // Output too small: truncated result, ptr == last, value_too_large.
  char small[8];
  memset(small, '?', sizeof(small));
  auto result =
      zmij::to_chars(small, small + 3, 1234.5678, zmij::chars_format::fixed, 2);
  EXPECT_EQ(result.ec, std::errc::value_too_large);
  EXPECT_EQ(result.ptr, small + 3);
  EXPECT_EQ(std::string(small, 3), "123");  // "1234.57" truncated to 3 chars
}

// Precision beyond the fast-path limit routes through the exact big-integer
// writer and still matches printf, with truncation reported as before.
TEST(double_test, to_chars_large_precision) {
  char buf[512], ref[512];
  auto check = [&](zmij::chars_format f, char conv, int precision,
                   double value) {
    auto r = zmij::to_chars(buf, buf + sizeof(buf), value, f, precision);
    EXPECT_EQ(r.ec, std::errc());
    char spec[16];
    snprintf(spec, sizeof(spec), "%%.%d%c", precision, conv);
    snprintf(ref, sizeof(ref), spec, value);
    EXPECT_EQ(std::string(buf, r.ptr), std::string(ref))
        << conv << " precision=" << precision << " value=" << value;
  };
  for (double value : {1.5, 0.1, 1234.5678, 6.62607015e-34, 1e300}) {
    check(zmij::chars_format::scientific, 'e', 30, value);
    check(zmij::chars_format::fixed, 'f', 25, value);
    check(zmij::chars_format::general, 'g', 30, value);
  }

  // Truncation past the fast path still fills the output and reports overflow.
  char small[5];
  memset(small, '?', sizeof(small));
  auto r = zmij::to_chars(small, small + sizeof(small), 1.5,
                          zmij::chars_format::scientific, 30);
  EXPECT_EQ(r.ec, std::errc::value_too_large);
  EXPECT_EQ(r.ptr, small + sizeof(small));
  EXPECT_EQ(std::string(small, sizeof(small)), "1.500");
}

TEST(double_test, to_decimal) {
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

TEST(double_test, write_precision) {
  EXPECT_EQ(to_scientific(1.5, 1), "1.5e+00");
  EXPECT_EQ(to_scientific(1.0, 0), "1e+00");       // no point when precision 0
  EXPECT_EQ(to_scientific(0.0, 4), "0.0000e+00");  // zero
  EXPECT_EQ(to_scientific(std::numeric_limits<double>::infinity(), 2), "inf");

  // Overshoot: values >= 10 still normalize to a single leading digit.
  EXPECT_EQ(to_scientific(12.0, 1), "1.2e+01");
  EXPECT_EQ(to_scientific(123.0, 2), "1.23e+02");
  EXPECT_EQ(to_scientific(12345.678, 2), "1.23e+04");

  // Carry: rounding 9...9 up rolls into a new leading digit.
  EXPECT_EQ(to_scientific(9.99, 1), "1.0e+01");
  EXPECT_EQ(to_scientific(99.9, 1), "1.0e+02");

  // Round half-to-even.
  EXPECT_EQ(to_scientific(0.125, 1), "1.2e-01");  // 1.25 -> 1.2
  EXPECT_EQ(to_scientific(2.5, 0), "2e+00");      // -> 2 (even)
  EXPECT_EQ(to_scientific(3.5, 0), "4e+00");      // -> 4 (even)

  // Sign is carried through.
  EXPECT_EQ(to_scientific(-9.99, 1), "-1.0e+01");

  // Subnormals take a separate normalization path, so check both boundaries
  // (smallest and largest) at low and full precision.
  EXPECT_EQ(to_scientific(5e-324, 0), "5e-324");    // DBL_TRUE_MIN
  EXPECT_EQ(to_scientific(-5e-324, 0), "-5e-324");  // sign preserved
  // Smallest subnormal at full precision (exercises the widened table top).
  EXPECT_EQ(to_scientific(5e-324, 17), "4.94065645841246544e-324");
  // Largest subnormal, round-tripped at full precision.
  EXPECT_EQ(to_scientific(2.2250738585072009e-308, 16),
            "2.2250738585072009e-308");
  EXPECT_EQ(to_scientific(2.2250738585072009e-308, 5), "2.22507e-308");

  // Large values at low precision reach the low end of the table.
  EXPECT_EQ(to_scientific(1.7976931348623157e308, 0), "2e+308");  // DBL_MAX
  EXPECT_EQ(to_scientific(1.7976931348623157e308, 1), "1.8e+308");

  // Full-precision round trip.
  EXPECT_EQ(to_scientific(6.62607015e-34, 8), "6.62607015e-34");
}

TEST(double_test, negative_precision) {
  // Pass the same negative/zero precision to printf, which defaults it to 6
  // (and treats 0 as 1 for %g), and check we produce identical output.
  double value = 1234.5678;
  char buf[64], ref[64];
  char* end = zmij::write_scientific(buf, sizeof(buf), value, -1);
  snprintf(ref, sizeof(ref), "%.*e", -1, value);
  EXPECT_EQ(std::string(buf, end), ref);
  end = zmij::write_fixed(buf, sizeof(buf), value, -5);
  snprintf(ref, sizeof(ref), "%.*f", -5, value);
  EXPECT_EQ(std::string(buf, end), ref);
  end = zmij::write_general(buf, sizeof(buf), value, -1);
  snprintf(ref, sizeof(ref), "%.*g", -1, value);
  EXPECT_EQ(std::string(buf, end), ref);
  end = zmij::write_general(buf, sizeof(buf), value, 0);
  snprintf(ref, sizeof(ref), "%.*g", 0, value);
  EXPECT_EQ(std::string(buf, end), ref);
}

TEST(double_test, write_precision_irregular) {
  for (uint64_t exp = 1; exp <= 2046; ++exp) {
    uint64_t bits = exp << 52;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));
    for (int precision = 0; precision <= 18; ++precision) {
      char expected[32];
      snprintf(expected, sizeof(expected), "%.*e", precision, value);
      EXPECT_EQ(to_scientific(value, precision), expected)
          << "value=" << value << " precision=" << precision;
    }
  }
}

// Big precision (> 18) routes write_scientific, write_general, and write_fixed
// through write_big; all must match printf's %e, %g, and %f.
TEST(double_test, write_big) {
  auto check = [](double value, int precision) {
    char buf[1200], ref[1200];
    char* end = zmij::write_scientific(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*e", precision, value);
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "scientific value=" << value << " precision=" << precision;
    end = zmij::write_general(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*g", precision, value);
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "general value=" << value << " precision=" << precision;
    end = zmij::write_fixed(buf, sizeof(buf), value, precision);
    snprintf(ref, sizeof(ref), "%.*f", precision, value);
    EXPECT_EQ(std::string(buf, end), std::string(ref))
        << "fixed value=" << value << " precision=" << precision;
  };
  const double values[] = {1.0,
                           2.0,
                           0.1,
                           0.5,
                           1.5,
                           1.25,
                           3.141592653589793,
                           1234.5678,
                           1e300,
                           1e-300,
                           9.999999999999999e22,
                           1.7976931348623157e308,   // DBL_MAX
                           2.2250738585072014e-308,  // smallest normal
                           5e-324};                  // smallest subnormal
  for (double value : values) {
    for (int precision : {19, 20, 25, 30, 40, 60, 100, 300, 767, 800}) {
      check(value, precision);
      check(-value, precision);
    }
  }
}

// An undersized buffer truncates the big-precision result without overrunning.
TEST(double_test, write_big_truncated) {
  char buf[8];
  memset(buf, '?', sizeof(buf));
  char* end = zmij::write_scientific(buf, 5, 1.5, 30);
  EXPECT_EQ(std::string(buf, end), "1.500");  // first 5 of 1.500...e+00
  EXPECT_EQ(end, buf + 5);
  EXPECT_EQ(buf[5], '?');  // no overrun past the requested size

  memset(buf, '?', sizeof(buf));
  end = zmij::write_general(buf, 5, 0.1, 30);
  EXPECT_EQ(std::string(buf, end), "0.100");  // first 5 of 0.10000...
  EXPECT_EQ(end, buf + 5);
  EXPECT_EQ(buf[5], '?');

  memset(buf, '?', sizeof(buf));
  end = zmij::write_fixed(buf, 5, 1.5, 30);
  EXPECT_EQ(std::string(buf, end), "1.500");  // first 5 of 1.5000...
  EXPECT_EQ(end, buf + 5);
  EXPECT_EQ(buf[5], '?');
}

// write_big with zero fractional digits must not emit a trailing decimal point,
// including on carry (e.g. 9.5 -> 1e+01). This path is only reachable directly,
// since the public API routes low precision through the shortest writers.
TEST(double_test, write_big_no_point) {
  char buf[32], ref[32];
  for (double value : {1.0, 2.5, 9.5, 12.5, 0.5, 1e300, 5e-324}) {
    for (int precision : {0, 1, 2}) {
      size_t len = zmij::detail::write_big(value, precision, buf, sizeof(buf),
                                           zmij::format::scientific);
      snprintf(ref, sizeof(ref), "%.*e", precision, value);
      EXPECT_EQ(std::string(buf, len), std::string(ref))
          << "value=" << value << " precision=" << precision;
    }
  }
}

// write_big returns the full would-be length, even when the output is
// truncated, so callers can detect insufficient space.
TEST(double_test, write_big_reports_true_length) {
  // 1.5 with 30 fractional digits in scientific is
  // "1." + 30 fractional digits + "e+00" = 36 characters.
  const size_t expected = 36;

  char buf[64];
  size_t full = zmij::detail::write_big(1.5, 30, buf, sizeof(buf),
                                        zmij::format::scientific);
  EXPECT_EQ(full, expected);  // fit exactly, no truncation

  // Undersized buffer: the output is truncated to the first 5 chars, but the
  // returned length is the full size that would have been needed.
  char small[8];
  memset(small, '?', sizeof(small));
  size_t needed =
      zmij::detail::write_big(1.5, 30, small, 5, zmij::format::scientific);
  EXPECT_EQ(std::string(small, 5), "1.500");  // first 5 chars only
  EXPECT_EQ(small[5], '?');                   // no overrun past the capacity
  EXPECT_EQ(needed, expected);                // true length, > capacity
}

// Use double-exact values (plain double literals) so the expected output is the
// same whether long double is the x87 80-bit, IEEE binary128 or double format,
// keeping the test portable across platforms.
TEST(long_double_test, write_scientific) {
  char buf[80], ref[80];
  for (long double value : {1.5, 0.0, 1e300, 5e-324}) {
    for (int precision : {-1, 0, 6, 20}) {
      char* end = zmij::write_scientific(buf, sizeof(buf), value, precision);
      snprintf(ref, sizeof(ref), "%.*Le", precision < 0 ? 6 : precision, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "value=" << double(value) << " precision=" << precision;
    }
  }
}

TEST(long_double_test, write_general) {
  char buf[360], ref[360];
  for (long double value : {1.5, 0.0, 123456.0, 1e300, 5e-324}) {
    for (int precision : {-1, 0, 1, 6, 20}) {
      char* end = zmij::write_general(buf, sizeof(buf), value, precision);
      int p = precision < 0 ? 6 : precision;
      snprintf(ref, sizeof(ref), "%.*Lg", p, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "value=" << double(value) << " precision=" << precision;
    }
  }
}

TEST(long_double_test, write_fixed) {
  char buf[360], ref[360];
  for (long double value : {1.5, 0.0, 123456.0, 5e-324}) {
    for (int precision : {-1, 0, 6, 20}) {
      char* end = zmij::write_fixed(buf, sizeof(buf), value, precision);
      snprintf(ref, sizeof(ref), "%.*Lf", precision < 0 ? 6 : precision, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "value=" << double(value) << " precision=" << precision;
    }
  }
}

// Exercise the actual extended-precision path (x87 80-bit / IEEE binary128)
// with values carrying more precision and range than double. snprintf's %L
// output is the oracle, so the test also holds where long double is just
// double.
TEST(long_double_test, extended) {
  char buf[8192], ref[8192];
  long double values[] = {
      3.14159265358979323846264338327950288L,  // pi beyond double precision
      1.0L + 0x1p-60L,  // differs from 1.0 only when extended
      1.23456789012345678901234567890123L,             // 33 significant digits
      std::numeric_limits<long double>::max(),         // extreme range
      std::numeric_limits<long double>::denorm_min(),  // smallest subnormal
  };
  for (long double value : values) {
    for (int precision : {-1, 0, 1, 20, 40}) {
      int p = precision < 0 ? 6 : precision;
      char* end = zmij::write_scientific(buf, sizeof(buf), value, precision);
      snprintf(ref, sizeof(ref), "%.*Le", p, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "scientific value=" << value << " precision=" << precision;
      end = zmij::write_fixed(buf, sizeof(buf), value, precision);
      snprintf(ref, sizeof(ref), "%.*Lf", p, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "fixed value=" << value << " precision=" << precision;
      end = zmij::write_general(buf, sizeof(buf), value, precision);
      snprintf(ref, sizeof(ref), "%.*Lg", p, value);
      EXPECT_EQ(std::string(buf, end), std::string(ref))
          << "general value=" << value << " precision=" << precision;
    }
  }
}

// Shortest to_chars for long double must match write and report truncation.
TEST(long_double_test, to_chars) {
  char buf[64], ref[64];
  for (long double value : {1.5L, 0.0L, 1e300L, 5e-324L}) {
    auto r = zmij::to_chars(buf, buf + sizeof(buf), value);
    char* end = zmij::write(ref, sizeof(ref), value);
    EXPECT_EQ(r.ec, std::errc());
    EXPECT_EQ(std::string(buf, r.ptr), std::string(ref, end))
        << "value=" << double(value);
  }

  if (LDBL_MANT_DIG != DBL_MANT_DIG) {
    // An extended value drives write_big (shortest) not the double path.
    long double extended = 1.0L + 0x1p-63L;
    auto r = zmij::to_chars(buf, buf + sizeof(buf), extended);
    char* end = zmij::write(ref, sizeof(ref), extended);
    EXPECT_EQ(r.ec, std::errc());
    EXPECT_EQ(std::string(buf, r.ptr), std::string(ref, end));
  }

  // Too small: truncated output, ptr == last, value_too_large.
  char small[3] = {'?', '?', '?'};
  auto result = zmij::to_chars(small, small + 2, 1.25L);
  EXPECT_EQ(result.ec, std::errc::value_too_large);
  EXPECT_EQ(result.ptr, small + 2);
  EXPECT_EQ(std::string(small, sizeof(small)), "1.?");
}

// to_chars with a format and precision matches printf's %L across formats and
// precisions (including beyond the fast path) and reports truncation.
TEST(long_double_test, to_chars_format) {
  char buf[8192], ref[8192];
  auto check = [&](zmij::chars_format f, char conv, int precision,
                   long double value) {
    auto r = zmij::to_chars(buf, buf + sizeof(buf), value, f, precision);
    EXPECT_EQ(r.ec, std::errc());
    char spec[16];
    snprintf(spec, sizeof(spec), "%%.%dL%c", precision, conv);
    snprintf(ref, sizeof(ref), spec, value);
    EXPECT_EQ(std::string(buf, r.ptr), std::string(ref))
        << conv << " precision=" << precision << " value=" << double(value);
  };
  long double values[] = {
      1.5L, 0.0L, 1234.5678L, 1e300L,
      3.14159265358979323846264338327950288L,  // beyond double precision
      1.0L + 0x1p-60L,                          // differs from 1.0 if extended
  };
  for (long double value : values) {
    for (int precision : {0, 6, 20, 40}) {
      check(zmij::chars_format::scientific, 'e', precision, value);
      check(zmij::chars_format::fixed, 'f', precision, value);
      check(zmij::chars_format::general, 'g', precision, value);
    }
  }

  // `hex` writes `precision` fractional hex digits (no 0x prefix); a negative
  // precision selects the shortest form. Not compared to %La: glibc's 80-bit
  // %La uses a different form.
  auto hex = [&](int precision, long double value) {
    auto r = zmij::to_chars(buf, buf + sizeof(buf), value,
                            zmij::chars_format::hex, precision);
    EXPECT_EQ(r.ec, std::errc());
    return std::string(buf, r.ptr);
  };
  EXPECT_EQ(hex(10, 1.5L), "1.8000000000p+0");
  EXPECT_EQ(hex(0, 1024.0L), "1p+10");
  EXPECT_EQ(hex(-1, 1.5L), "1.8p+0");  // shortest

  // Too small: truncated result, ptr == last, value_too_large.
  char small[5];
  memset(small, '?', sizeof(small));
  auto r = zmij::to_chars(small, small + sizeof(small), 1.5L,
                          zmij::chars_format::scientific, 30);
  EXPECT_EQ(r.ec, std::errc::value_too_large);
  EXPECT_EQ(r.ptr, small + sizeof(small));
  EXPECT_EQ(std::string(small, sizeof(small)), "1.500");
}

// Number of significant decimal digits in a shortest-formatted string.
static int count_sig_digits(const std::string& s) {
  size_t e = s.find_first_of("eE");
  std::string m = s.substr(0, e == std::string::npos ? s.size() : e);
  std::string d;
  for (char c : m)
    if (c >= '0' && c <= '9') d += c;
  size_t first = d.find_first_not_of('0');
  if (first == std::string::npos) return 1;  // the value 0
  size_t last = d.find_last_not_of('0');
  return int(last - first + 1);
}

// Exercises the extended-precision shortest path: every output must round-trip
// through strtold, and dropping a significant digit must break the round-trip
// (minimality). Uses random full-width significands not exactly representable
// as double, so it drives write_big (shortest) rather than the double fast
// path.
TEST(long_double_test, write_shortest) {
  if (LDBL_MANT_DIG == DBL_MANT_DIG)
    GTEST_SKIP() << "long double is double; no extended path";
  auto check = [](long double value) {
    char buf[zmij::long_double_buffer_size + 1];
    memset(buf, '?', sizeof(buf));
    char* end = zmij::write(buf + 1, sizeof(buf) - 1, value);
    ASSERT_EQ(buf[0], '?') << "buffer underrun";
    std::string s(buf + 1, end);
    EXPECT_EQ(strtold(s.c_str(), nullptr), value) << "roundtrip " << s;
    int digits = count_sig_digits(s);
    if (digits > 1) {  // a shorter decimal must not round-trip
      char ref[64];
      snprintf(ref, sizeof(ref), "%.*Le", digits - 2, value);
      EXPECT_NE(strtold(ref, nullptr), value) << "not minimal " << s;
    }
  };

  long double edges[] = {
      3.14159265358979323846264338327950288L,
      1.0L + 0x1p-63L,
      1.23456789012345678901234567890123L,
      std::numeric_limits<long double>::max(),
      std::numeric_limits<long double>::min(),
      std::numeric_limits<long double>::denorm_min(),
  };
  for (long double value : edges) {
    check(value);
    check(-value);
  }

  // Powers of two exercise the irregular (asymmetric-gap) path over the full
  // exponent range. A large one (2**13969 ~ 1.2246e4205) once produced a
  // non-minimal result: its wide ulp overflowed the 128-bit trim comparison.
  using lim = std::numeric_limits<long double>;
  for (int e = lim::min_exponent - lim::digits; e < lim::max_exponent; ++e) {
    long double value = std::ldexp(1.0L, e - 1);
    if (value == 0 || std::isinf(value)) continue;
    check(value);
    check(-value);
  }

  uint64_t state = 0x9e3779b97f4a7c15ull;
  auto next = [&state] {  // xorshift64
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int i = 0; i < 200000; ++i) {
    long double sig = long_double(next());
    sig = sig * 0x1p64L + long_double(next());  // up to 128 significand bits
    int exp = int(next() % 30000) - 15000;
    long double value = std::ldexp(sig, exp - 128);
    if (value == 0 || std::isinf(value)) continue;
    check((next() & 1) ? value : -value);
  }

  // Regression (binary128): for this value v - half_ulp lands one nibble past
  // the trim boundary, yet the packed 124-bit comparison read c == half_ulp as
  // a tie and trimmed to even, dropping the last digit. The refined tie-break
  // keeps it. bin_sig = 0x00012caaef34c608080750fd906c8100, exponent -2266.
  if (LDBL_MANT_DIG == 113) {
    long double misround = long_double(0x00012caaef34c608ull) * 0x1p64L +
                           long_double(0x080750fd906c8100ull);
    misround = std::ldexp(misround, -2266);
    check(misround);
    check(-misround);
  }
}

TEST(float_test, write_general) {
  EXPECT_EQ(to_general(1.5f, 6), "1.5");
  EXPECT_EQ(to_general(0.0001f, 6), "0.0001");  // exp10 == -4 -> fixed
  EXPECT_EQ(to_general(0.00001f, 6), "1e-05");  // exp10 == -5 -> scientific
  EXPECT_EQ(to_general(-1.5f, 6), "-1.5");      // sign preserved
  EXPECT_EQ(to_general(std::numeric_limits<float>::denorm_min(), 1),
            "1e-45");  // subnormal path
}

TEST(double_test, write_general) {
  // Fixed range: decimal exponent in [-4, precision).
  EXPECT_EQ(to_general(1.5, 6), "1.5");
  EXPECT_EQ(to_general(100.0, 6), "100");
  EXPECT_EQ(to_general(123456.0, 6), "123456");
  EXPECT_EQ(to_general(0.0001, 6), "0.0001");  // exp10 == -4 -> fixed
  EXPECT_EQ(to_general(0.00001, 6), "1e-05");  // exp10 == -5 -> scientific
  EXPECT_EQ(to_general(1234567.0, 6),
            "1.23457e+06");  // exp10 == precision -> sci

  // Trailing zeros are trimmed, and the point with them.
  EXPECT_EQ(to_general(1.2000, 6), "1.2");
  EXPECT_EQ(to_general(1.0, 6), "1");
  EXPECT_EQ(to_general(1024.0, 6), "1024");

  // Zero and sign.
  EXPECT_EQ(to_general(0.0, 6), "0");
  EXPECT_EQ(to_general(-0.0, 6), "-0");
  EXPECT_EQ(to_general(-1.5, 6), "-1.5");

  // Rounding rolls into a new leading digit and bumps the format to scientific.
  EXPECT_EQ(to_general(999999.0, 5), "1e+06");

  // Specials.
  EXPECT_EQ(to_general(std::numeric_limits<double>::infinity(), 6), "inf");

  // Full-precision round trips.
  EXPECT_EQ(to_general(6.62607015e-34, 9), "6.62607015e-34");
  EXPECT_EQ(to_general(3.14159265358979, 15), "3.14159265358979");
}

TEST(double_test, write_general_irregular) {
  for (uint64_t exp = 1; exp <= 2046; ++exp) {
    uint64_t bits = exp << 52;
    double value = 0;
    memcpy(&value, &bits, sizeof(double));
    for (int precision = 1; precision <= 18; ++precision) {
      char expected[32];
      snprintf(expected, sizeof(expected), "%.*g", precision, value);
      EXPECT_EQ(to_general(value, precision), expected)
          << "value=" << value << " precision=" << precision;
    }
  }
}

template <typename Float> static auto to_hex(Float value) -> std::string {
  char buffer[zmij::buffer_sizes<Float>::hex];
  return {buffer, zmij::write_hex(buffer, sizeof(buffer), value)};
}

TEST(float_test, write_hex) {
  EXPECT_EQ(to_hex(1.0f), "0x1p+0");
  EXPECT_EQ(to_hex(-3.5f), "-0x1.cp+1");
  EXPECT_EQ(to_hex(0.1f), "0x1.99999ap-4");
  EXPECT_EQ(to_hex(0.0f), "0x0p+0");
  EXPECT_EQ(to_hex(-0.0f), "-0x0p+0");
  EXPECT_EQ(to_hex(std::numeric_limits<float>::infinity()), "inf");
  EXPECT_EQ(to_hex(std::numeric_limits<float>::quiet_NaN()), "nan");
}

TEST(double_test, write_hex) {
  EXPECT_EQ(to_hex(1.0), "0x1p+0");
  EXPECT_EQ(to_hex(2.0), "0x1p+1");
  EXPECT_EQ(to_hex(0.5), "0x1p-1");
  EXPECT_EQ(to_hex(-1.5), "-0x1.8p+0");
  EXPECT_EQ(to_hex(0.0), "0x0p+0");
  EXPECT_EQ(to_hex(-0.0), "-0x0p+0");
  // Shortest form: trailing zero nibbles are dropped.
  EXPECT_EQ(to_hex(3.14159265358979), "0x1.921fb54442d11p+1");
  // Subnormals are normalized to a leading 0x1, matching printf's %a.
  EXPECT_EQ(to_hex(std::numeric_limits<double>::denorm_min()), "0x1p-1074");
  EXPECT_EQ(to_hex(std::numeric_limits<double>::min() -
                   std::numeric_limits<double>::denorm_min()),
            "0x1.ffffffffffffep-1023");
  EXPECT_EQ(to_hex(std::numeric_limits<double>::infinity()), "inf");
  EXPECT_EQ(to_hex(std::numeric_limits<double>::quiet_NaN()), "nan");
}

TEST(double_test, write_hex_no_buffer) {
  // "-0x1.8p+0" truncated to the first 4 chars, end points past them.
  char buf[4];
  char* end = zmij::write_hex(buf, sizeof(buf), -1.5);
  EXPECT_EQ(std::string(buf, end), "-0x1");
}

// Double-exact values so the expected output is the same whether long double
// is x87 80-bit, IEEE binary128, or plain double. This is not compared against
// snprintf's %La: glibc's 80-bit %La uses a non-leading-1 form (e.g. "0xcp-3"
// for 1.5), whereas write_hex always normalizes to a leading 0x1.
TEST(long_double_test, write_hex) {
  EXPECT_EQ(to_hex(1.5L), "0x1.8p+0");
  EXPECT_EQ(to_hex(-2.0L), "-0x1p+1");
  EXPECT_EQ(to_hex(0.5L), "0x1p-1");
  EXPECT_EQ(to_hex(0.0L), "0x0p+0");
  EXPECT_EQ(to_hex(1024.0L), "0x1p+10");
  EXPECT_EQ(to_hex(3.25L), "0x1.ap+1");
}

#endif  // !ZMIJ_C

auto main(int argc, char** argv) -> int {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
