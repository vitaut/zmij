// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_H_
#define ZMIJ_H_

#include <stddef.h>  // size_t
#include <string.h>  // memcpy

namespace zmij {
namespace detail {
template <typename Float>
auto write(Float value, char* buffer) noexcept -> char*;
}  // namespace detail

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

#endif  // ZMIJ_H_
