// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_H_
#define ZMIJ_H_

#include <assert.h>  // assert
#include <stddef.h>  // size_t
#include <string.h>  // memcpy

namespace zmij {
struct dec_fp;

namespace detail {
template <typename Float>
auto to_decimal(Float value, int precision) noexcept -> dec_fp;

template <typename Float>
auto write(Float value, char* buffer) noexcept -> char*;
}  // namespace detail

enum {
  non_finite_exp = int(~0u >> 1),
};

// A decimal floating-point number sig * pow(10, exp).
// If exp is non_finite_exp then the number is a NaN or an infinity.
struct dec_fp {
  long long sig;  // significand
  int exp;        // exponent
  bool negative;
};

/// Converts `value` into the shortest correctly rounded decimal representation.
/// Usage:
///   auto [sig, exp, negative] = to_decimal(6.62607015e-34);
auto to_decimal(double value) noexcept -> dec_fp;

/// Converts `value` into a correctly rounded decimal with exactly `precision`
/// significant digits (sig * 10**exp). `precision` must be in [1, 18];
/// out-of-range values are clamped.
inline auto to_decimal(float value, int precision) noexcept -> dec_fp {
  assert(precision >= 1 && precision <= 18);
  if (precision < 1) precision = 1;
  if (precision > 18) precision = 18;
  return detail::to_decimal(value, precision);
}

/// Converts `value` into a correctly rounded decimal with exactly `precision`
/// significant digits (sig * 10**exp). `precision` must be in [1, 18];
/// out-of-range values are clamped.
inline auto to_decimal(double value, int precision) noexcept -> dec_fp {
  assert(precision >= 1 && precision <= 18);
  if (precision < 1) precision = 1;
  if (precision > 18) precision = 18;
  return detail::to_decimal(value, precision);
}

enum {
  float_buffer_size = 17,
  double_buffer_size = 34,
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out` without a null terminator. Returns a pointer past the last character
/// written; if the representation exceeds `n` characters, only the first `n`
/// are written.
inline auto write(char* out, size_t n, float value) noexcept -> char* {
  if (n >= float_buffer_size) return detail::write(value, out);
  char buffer[float_buffer_size];
  size_t size = detail::write(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out` without a null terminator. Returns a pointer past the last character
/// written; if the representation exceeds `n` characters, only the first `n`
/// are written.
inline auto write(char* out, size_t n, double value) noexcept -> char* {
  if (n >= double_buffer_size) return detail::write(value, out);
  char buffer[double_buffer_size];
  size_t size = detail::write(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

}  // namespace zmij

#endif  // ZMIJ_H_
