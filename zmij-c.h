// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_C_H_
#define ZMIJ_C_H_

#include <stddef.h>  // size_t
#include <string.h>  // memcpy

// Implementation details, use zmij_write_* instead.
char* zmij_detail_write_float(float value, char* buffer);
char* zmij_detail_write_double(double value, char* buffer);

enum {
  zmij_float_buffer_size = 16,
  zmij_double_buffer_size = 25,
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`. `out` should point to a buffer of size `n` or larger.
static inline size_t zmij_write_float(char* out, size_t n, float value) {
  if (n >= zmij_float_buffer_size)
    return zmij_detail_write_float(value, out) - out;
  char buffer[zmij_float_buffer_size];
  size_t result = zmij_detail_write_float(value, buffer) - buffer;
  memcpy(out, buffer, n);
  return result;
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`. `out` should point to a buffer of size `n` or larger.
static inline size_t zmij_write_double(char* out, size_t n, double value) {
  if (n >= zmij_double_buffer_size)
    return zmij_detail_write_double(value, out) - out;
  char buffer[zmij_double_buffer_size];
  size_t result = zmij_detail_write_double(value, buffer) - buffer;
  memcpy(out, buffer, n);
  return result;
}

#endif  // ZMIJ_C_H_
