// A double-to-string conversion library: https://github.com/vitaut/zmij/
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_H_
#define ZMIJ_H_

#include <float.h>   // DBL_MANT_DIG, LDBL_MANT_DIG
#include <stddef.h>  // size_t
#include <string.h>  // memcpy

namespace zmij {

// Floating-point formatting style. Values match std::chars_format (hex is
// unsupported), so general == fixed | scientific.
enum class format {
  scientific = 1,
  fixed = 2,
  general = 3,
};

namespace detail {

// `buffer` params require at least buffer_sizes<Float> capacity;
// `out`/`n` params write at most `n` characters.

template <typename Float>
auto write(Float value, char* buffer) noexcept -> char*;

// Writes the shortest decimal representation of `value`, correctly rounded
// (ties to even), into `out`, truncating after `n` chars. Returns the total
// length the result would need; if it exceeds `n` the output was truncated to
// the first `n` chars.
template <typename Float>
auto write_big(Float value, char* out, size_t n) noexcept -> size_t;

// Writes `value` in `fmt` notation with `precision` digits, correctly rounded
// (ties to even), into `out`, truncating after `n` chars. Returns the total
// length the result would need; if it exceeds `n` the output was truncated to
// the first `n` chars, and 0 on allocation failure (only possible for long
// double).
template <typename Float>
auto write_big(Float value, int precision, char* out, size_t n,
               format fmt) noexcept -> size_t;
template <>
inline auto write_big(float value, int precision, char* out, size_t n,
                      format fmt) noexcept -> size_t {
  return write_big(double(value), precision, out, n, fmt);
}
#if LDBL_MANT_DIG == DBL_MANT_DIG
template <>
inline auto write_big(long double value, int precision, char* out, size_t n,
                      format fmt) noexcept -> size_t {
  return write_big(double(value), precision, out, n, fmt);
}
#endif

// Returns the past-the-end pointer after writing min(size, n) chars to `out`.
inline auto clamp_end(char* out, size_t size, size_t n) noexcept -> char* {
  return out + (size < n ? size : n);
}

template <typename Float>
auto write_scientific(Float value, int precision, char* buffer) noexcept
    -> char*;

template <typename Float>
auto write_general(Float value, int precision, char* buffer) noexcept -> char*;

template <typename Float>
auto write_fixed(Float value, int precision, char* buffer) noexcept -> char*;

// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
// its shortest form, e.g. -0x1.8p+1.
template <typename Float>
auto write_hex(Float value, char* buffer) noexcept -> char*;

}  // namespace detail

enum {
  non_finite_exp = int(~0u >> 1),
};

// A decimal floating-point number (negative ? -1 : 1) * sig * pow(10, exp).
// If exp is non_finite_exp then the number is a NaN or an infinity.
struct dec_fp {
  unsigned long long sig;  // significand
  int exp;                 // exponent
  bool negative;
};

/// Converts `value` into the shortest correctly rounded decimal representation.
/// Usage:
///   auto [sig, exp, negative] = to_decimal(6.62607015e-34);
auto to_decimal(double value) noexcept -> dec_fp;

// Minimum buffer sizes for the shortest `write`, one per floating-point type.
enum {
  float_buffer_size = 17,
  double_buffer_size = 34,
  // Worst case is IEEE binary128: 1 sign + 36 digits + '.' + "e-dddd".
  long_double_buffer_size = 44,
};

/// Buffer sizes for the write* functions, usable in generic code as
/// buffer_sizes<Float>::shortest, ::scientific, ::fixed, and ::hex.
/// `scientific` assumes precision up to 17 and `fixed` up to 18; long double
/// sets its own bounds below. Larger precision must be sized by the caller.
template <typename Float> struct buffer_sizes;

template <> struct buffer_sizes<float> {
  static constexpr size_t shortest = float_buffer_size;  // write
  static constexpr size_t scientific = 24;  // write_scientific (and general)
  static constexpr size_t fixed = 59;       // write_fixed
  static constexpr size_t hex = 16;         // write_hex
};
template <> struct buffer_sizes<double> {
  static constexpr size_t shortest = double_buffer_size;  // write
  static constexpr size_t scientific = 25;  // write_scientific (and general)
  static constexpr size_t fixed = 329;      // write_fixed
  static constexpr size_t hex = 24;         // write_hex
};
// long double: `scientific` is sized to round-trip: precision 35 (scientific)
// or 36 (general). `fixed` is omitted as it would need an impractical buffer,
// so callers must size fixed output themselves.
template <> struct buffer_sizes<long double> {
  static constexpr size_t shortest = long_double_buffer_size;  // write
  static constexpr size_t scientific = 44;  // write_scientific (and general)
  // Worst case is IEEE binary128: 1 sign + "0x1." + 28 digits + "p+16383".
  static constexpr size_t hex = 40;  // write_hex
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write(char* out, size_t n, float value) noexcept -> char* {
  if (n >= float_buffer_size) return detail::write(value, out);
  char buffer[float_buffer_size];
  size_t size = detail::write(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write(char* out, size_t n, double value) noexcept -> char* {
  if (n >= double_buffer_size) return detail::write(value, out);
  char buffer[double_buffer_size];
  size_t size = detail::write(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write(char* out, size_t n, long double value) noexcept -> char* {
#if LDBL_MANT_DIG == DBL_MANT_DIG
  return write(out, n, double(value));
#else
  // Shortest cannot fall back to double: a double-shortest decimal may not
  // round-trip at long double precision.
  return detail::clamp_end(out, detail::write_big(value, out, n), n);
#endif
}

/// Writes `value` in scientific format with `precision` digits after the
/// decimal point (e.g. 1.234e+05) to `out`, without a null terminator, like
/// printf's %e. A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_scientific(char* out, size_t n, float value,
                             int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    auto size = detail::write_big(value, precision, out, n, format::scientific);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<float>::scientific)
    return detail::write_scientific(value, precision + 1, out);
  char buffer[buffer_sizes<float>::scientific];
  size_t size = detail::write_scientific(value, precision + 1, buffer) - buffer;
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
inline auto write_scientific(char* out, size_t n, double value,
                             int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    auto size = detail::write_big(value, precision, out, n, format::scientific);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<double>::scientific)
    return detail::write_scientific(value, precision + 1, out);
  char buffer[buffer_sizes<double>::scientific];
  size_t size = detail::write_scientific(value, precision + 1, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in scientific format with `precision` digits after the
/// decimal point (e.g. 1.234e+05) to `out`, without a null terminator, like
/// printf's %e. A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written, or nullptr on allocation
/// failure; if the representation exceeds `n` characters, only the first `n`
/// are written.
inline auto write_scientific(char* out, size_t n, long double value,
                             int precision) noexcept -> char* {
  if (double(value) == value)
    return write_scientific(out, n, double(value), precision);
  if (precision < 0) precision = 6;
  auto size = detail::write_big(value, precision, out, n, format::scientific);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

/// Writes `value` in general format with up to `precision` significant digits
/// and no trailing zeros (e.g. 1.5 or 1.5e+20) to `out`, without a null
/// terminator. Fixed notation is used when `value`'s decimal exponent is in
/// [-4, precision), and scientific otherwise. A negative `precision` defaults
/// to 6 and zero is treated as 1, matching printf.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_general(char* out, size_t n, float value,
                          int precision) noexcept -> char* {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    auto size = detail::write_big(value, precision, out, n, format::general);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<float>::scientific)
    return detail::write_general(value, precision, out);
  char buffer[buffer_sizes<float>::scientific];
  size_t size = detail::write_general(value, precision, buffer) - buffer;
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
inline auto write_general(char* out, size_t n, double value,
                          int precision) noexcept -> char* {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    auto size = detail::write_big(value, precision, out, n, format::general);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<double>::scientific)
    return detail::write_general(value, precision, out);
  char buffer[buffer_sizes<double>::scientific];
  size_t size = detail::write_general(value, precision, buffer) - buffer;
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
/// Returns a pointer past the last character written, or nullptr on allocation
/// failure; if the representation exceeds `n` characters, only the first `n`
/// are written.
inline auto write_general(char* out, size_t n, long double value,
                          int precision) noexcept -> char* {
  if (double(value) == value)
    return write_general(out, n, double(value), precision);
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  auto size = detail::write_big(value, precision, out, n, format::general);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

/// Writes `value` in fixed notation with exactly `precision` digits after the
/// decimal point (e.g. 1.500) to `out`, without a null terminator. The result
/// is the exact value correctly rounded (ties to even), matching printf's %f.
/// A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_fixed(char* out, size_t n, float value,
                        int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    auto size = detail::write_big(value, precision, out, n, format::fixed);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<float>::fixed)
    return detail::write_fixed(value, precision, out);
  char buffer[buffer_sizes<float>::fixed];
  size_t size = detail::write_fixed(value, precision, buffer) - buffer;
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
inline auto write_fixed(char* out, size_t n, double value,
                        int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    auto size = detail::write_big(value, precision, out, n, format::fixed);
    return detail::clamp_end(out, size, n);
  }
  if (n >= buffer_sizes<double>::fixed)
    return detail::write_fixed(value, precision, out);
  char buffer[buffer_sizes<double>::fixed];
  size_t size = detail::write_fixed(value, precision, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in fixed notation with exactly `precision` digits after the
/// decimal point (e.g. 1.500) to `out`, without a null terminator. The result
/// is the exact value correctly rounded (ties to even), matching printf's %f.
/// A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written, or nullptr on allocation
/// failure; if the representation exceeds `n` characters, only the first `n`
/// are written.
inline auto write_fixed(char* out, size_t n, long double value,
                        int precision) noexcept -> char* {
  if (double(value) == value)
    return write_fixed(out, n, double(value), precision);
  if (precision < 0) precision = 6;
  auto size = detail::write_big(value, precision, out, n, format::fixed);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

auto write_hex(char* out, size_t n, double value) noexcept -> char*;

/// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
/// its shortest form (e.g. -0x1.8p+1) to `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_hex(char* out, size_t n, float value) noexcept -> char* {
  return write_hex(out, n, double(value));
}

/// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
/// its shortest form (e.g. -0x1.8p+1) to `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_hex(char* out, size_t n, double value) noexcept -> char* {
  if (n >= buffer_sizes<double>::hex) return detail::write_hex(value, out);
  char buffer[buffer_sizes<double>::hex];
  size_t size = detail::write_hex(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
}

/// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
/// its shortest form (e.g. -0x1.8p+1) to `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_hex(char* out, size_t n, long double value) noexcept
    -> char* {
#if LDBL_MANT_DIG == DBL_MANT_DIG
  return write_hex(out, n, double(value));
#else
  if (n >= buffer_sizes<long double>::hex) return detail::write_hex(value, out);
  char buffer[buffer_sizes<long double>::hex];
  size_t size = detail::write_hex(value, buffer) - buffer;
  if (size > n) size = n;
  memcpy(out, buffer, size);
  return out + size;
#endif
}

}  // namespace zmij

#endif  // ZMIJ_H_
