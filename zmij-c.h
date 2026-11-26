// A double-to-string conversion library: https://github.com/vitaut/zmij/
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_C_H_
#define ZMIJ_C_H_

#undef ZMIJ_EXT
#ifdef ZMIJ_FFI
#define ZMIJ_EXT extern
#else
#define ZMIJ_EXT static inline
#endif /* ZMIJ_EXT */

#include <stddef.h>  // size_t
#include <string.h>  // memcpy

// Implementation details, use the zmij_write functions instead.
char* zmij_detail_write_f(float value, char* buffer);
char* zmij_detail_write(double value, char* buffer);
char* zmij_detail_write_scientific_f(float value, char* buffer, int num_digits);
char* zmij_detail_write_scientific(double value, char* buffer, int num_digits);
size_t zmij_detail_write_scientific_big(char* out, size_t n, double value,
                                        int precision);
char* zmij_detail_write_general_f(float value, char* buffer, int precision);
char* zmij_detail_write_general(double value, char* buffer, int precision);
size_t zmij_detail_write_general_big(char* out, size_t n, double value,
                                     int precision);
char* zmij_detail_write_fixed_f(float value, char* buffer, int precision);
char* zmij_detail_write_fixed(double value, char* buffer, int precision);
size_t zmij_detail_write_fixed_big(char* out, size_t n, double value,
                                   int precision);
char* zmij_detail_write_hex(double value, char* buffer);

enum {
  // Minimum buffer sizes for zmij_write and zmij_write_f.
  zmij_float_buffer_size = 17,
  zmij_double_buffer_size = 34,
  // Buffer sizes that always suffice for the zmij_write_scientific functions
  // with precision up to 17 and the zmij_write_general ones with precision up
  // to 18; higher precision needs `precision + 8` bytes.
  zmij_float_scientific_buffer_size = 24,
  zmij_double_scientific_buffer_size = 25,
  // Buffer sizes that always suffice for the zmij_write_fixed functions with
  // precision up to 18.
  zmij_float_fixed_buffer_size = 59,
  zmij_double_fixed_buffer_size = 329,
  // Buffer sizes that always suffice for the zmij_write_hex functions.
  zmij_float_hex_buffer_size = 16,
  zmij_double_hex_buffer_size = 24,
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out` without a null terminator. Returns a pointer past the last character
/// written; if the representation exceeds `n` characters, only the first `n`
/// are written.
ZMIJ_EXT char* zmij_write(char* out, size_t n, double value) {
  if (n >= zmij_double_buffer_size) return zmij_detail_write(value, out);
  char buffer[zmij_double_buffer_size];
  size_t size = zmij_detail_write(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

ZMIJ_EXT char* zmij_write_f(char* out, size_t n, float value) {
  if (n >= zmij_float_buffer_size) return zmij_detail_write_f(value, out);
  char buffer[zmij_float_buffer_size];
  size_t size = zmij_detail_write_f(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in scientific format with `precision` digits after the
/// decimal point (e.g. 1.234e+05) to `out`, without a null terminator, like
/// printf's %e. A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
ZMIJ_EXT char* zmij_write_scientific(char* out, size_t n, double value,
                                          int precision) {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    size_t size = zmij_detail_write_scientific_big(out, n, value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_double_scientific_buffer_size)
    return zmij_detail_write_scientific(value, out, precision + 1);
  char buffer[zmij_double_scientific_buffer_size];
  size_t size =
      zmij_detail_write_scientific(value, buffer, precision + 1) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

ZMIJ_EXT char* zmij_write_scientific_f(char* out, size_t n, float value,
                                            int precision) {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    // A float is exact as a double, so both produce the same digits.
    size_t size =
        zmij_detail_write_scientific_big(out, n, (double)value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_float_scientific_buffer_size)
    return zmij_detail_write_scientific_f(value, out, precision + 1);
  char buffer[zmij_float_scientific_buffer_size];
  size_t size =
      zmij_detail_write_scientific_f(value, buffer, precision + 1) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in general format with up to `precision` significant digits
/// and no trailing zeros (e.g. 1.5 or 1.5e+20) to `out`, without a null
/// terminator. Fixed notation is used when `value`'s decimal exponent is in
/// [-4, precision), and scientific otherwise. A negative `precision` defaults
/// to 6 and zero is treated as 1, matching printf.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
ZMIJ_EXT char* zmij_write_general(char* out, size_t n, double value,
                                       int precision) {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    size_t size = zmij_detail_write_general_big(out, n, value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_double_scientific_buffer_size)
    return zmij_detail_write_general(value, out, precision);
  char buffer[zmij_double_scientific_buffer_size];
  size_t size = zmij_detail_write_general(value, buffer, precision) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

ZMIJ_EXT char* zmij_write_general_f(char* out, size_t n, float value,
                                         int precision) {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    // A float is exact as a double, so both produce the same digits.
    size_t size =
        zmij_detail_write_general_big(out, n, (double)value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_float_scientific_buffer_size)
    return zmij_detail_write_general_f(value, out, precision);
  char buffer[zmij_float_scientific_buffer_size];
  size_t size = zmij_detail_write_general_f(value, buffer, precision) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in fixed notation with exactly `precision` digits after the
/// decimal point (e.g. 1.500) to `out`, without a null terminator. The result
/// is the exact value correctly rounded (ties to even), matching printf's %f.
/// A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
ZMIJ_EXT char* zmij_write_fixed(char* out, size_t n, double value,
                                     int precision) {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    size_t size = zmij_detail_write_fixed_big(out, n, value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_double_fixed_buffer_size)
    return zmij_detail_write_fixed(value, out, precision);
  char buffer[zmij_double_fixed_buffer_size];
  size_t size = zmij_detail_write_fixed(value, buffer, precision) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

ZMIJ_EXT char* zmij_write_fixed_f(char* out, size_t n, float value,
                                       int precision) {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    // A float is exact as a double, so both produce the same digits.
    size_t size = zmij_detail_write_fixed_big(out, n, (double)value, precision);
    return out + (size < n ? size : n);
  }
  if (n >= zmij_float_fixed_buffer_size)
    return zmij_detail_write_fixed_f(value, out, precision);
  char buffer[zmij_float_fixed_buffer_size];
  size_t size = zmij_detail_write_fixed_f(value, buffer, precision) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
/// its shortest form (e.g. -0x1.8p+1) to `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
ZMIJ_EXT char* zmij_write_hex(char* out, size_t n, double value) {
  if (n >= zmij_double_hex_buffer_size)
    return zmij_detail_write_hex(value, out);
  char buffer[zmij_double_hex_buffer_size];
  size_t size = zmij_detail_write_hex(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

ZMIJ_EXT char* zmij_write_hex_f(char* out, size_t n, float value) {
  if (n >= zmij_float_hex_buffer_size)
    return zmij_detail_write_hex((double)value, out);
  char buffer[zmij_float_hex_buffer_size];
  size_t size = zmij_detail_write_hex((double)value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

#endif  // ZMIJ_C_H_
