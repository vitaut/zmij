// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_H_
#define ZMIJ_H_

#include <stddef.h>  // size_t
#include <string.h>  // memcpy

#ifdef ZMIJ_ENABLE_COMPILETIME_EVALUATION
#  define ZMEJ_AUTO constexpr auto
#else
#  define ZMEJ_AUTO auto
#endif

namespace zmij {
namespace detail {
template <typename Float>
ZMEJ_AUTO write(Float value, char* buffer) noexcept -> char*;
}  // namespace detail

enum {
  non_finite_exp = int(~0u >> 1),
};

// A decimal floating-point number sig * pow(10, exp).
// If exp is non_finite_exp then the number is a NaN or an infinity.
struct dec_fp {
  long long sig;  // significand
  int exp;        // exponent
};

/// Converts `value` into the shortest correctly rounded decimal representation.
/// Usage:
///   auto [sig, exp] = to_decimal(6.62607015e-34);
ZMEJ_AUTO to_decimal(double value) noexcept -> dec_fp;

enum {
  double_buffer_size = 25,
  float_buffer_size = 17,
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`. `out` should point to a buffer of size `n` or larger.
inline auto write(char* out, size_t n, double value) noexcept -> size_t {
  if (n >= double_buffer_size) return detail::write(value, out) - out;
  char buffer[double_buffer_size];
  size_t result = detail::write(value, buffer) - buffer;
  memcpy(out, buffer, n);
  return result;
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`. `out` should point to a buffer of size `n` or larger.
inline auto write(char* out, size_t n, float value) noexcept -> size_t {
  if (n >= float_buffer_size) return detail::write(value, out) - out;
  char buffer[float_buffer_size];
  size_t result = detail::write(value, buffer) - buffer;
  memcpy(out, buffer, n);
  return result;
}

}  // namespace zmij

#if defined(ZMIJ_HEADER_ONLY) || defined(ZMIJ_ENABLE_COMPILETIME_EVALUATION)
#  if __has_include("zmij.cc")
#    include "zmij.cc"
#  elif __has_include(<zmij.cc>)
#    include <zmij.cc>
#  endif
#endif

#undef ZMEJ_AUTO
#endif  // ZMIJ_H_
