// A double-to-string conversion library: https://github.com/vitaut/zmij/
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_H_
#define ZMIJ_H_

#include <float.h>   // DBL_MANT_DIG, LDBL_MANT_DIG
#include <stddef.h>  // size_t
#include <stdint.h>  // uint64_t
#include <string.h>  // memcpy

namespace zmij {

enum {
  nonfinite_exp = int(~0u >> 1),
};

// A decimal floating-point number (negative ? -1 : 1) * sig * pow(10, exp).
// If exp is nonfinite_exp then the number is a NaN or an infinity.
template <typename Sig = uint64_t> struct dec_fp {
  Sig sig;  // significand
  int exp;  // exponent
  bool negative;
};

// Floating-point formatting style. Values match std::chars_format, so
// general == fixed | scientific.
enum class format {
  scientific = 1,
  fixed = 2,
  general = 3,
  hex = 4,
};

// Minimum buffer sizes for the shortest `write`, one per floating-point type.
enum {
  float_buffer_size = 17,
  double_buffer_size = 34,
  // Worst case is IEEE binary128: 1 sign + 36 digits + '.' + "e-dddd".
  long_double_buffer_size = 44,
};

namespace detail {

struct uint128 {
  uint64_t lo;
  uint64_t hi;

  uint128() = default;
  constexpr uint128(uint64_t hi, uint64_t lo) noexcept : lo(lo), hi(hi) {}
  constexpr uint128(uint64_t lo) noexcept : lo(lo), hi(0) {}

  explicit constexpr operator uint64_t() const noexcept { return lo; }

  constexpr auto operator+(uint128 rhs) const noexcept -> uint128 {
    uint64_t s = lo + rhs.lo;
    return {hi + rhs.hi + (s < lo), s};
  }
  constexpr auto operator-(uint128 rhs) const noexcept -> uint128 {
    uint64_t d = lo - rhs.lo;
    return {hi - rhs.hi - (d > lo), d};
  }
  constexpr auto operator%(uint32_t d) const noexcept -> uint64_t {
    uint64_t m = (UINT64_MAX % d + 1) % d;  // 2**64 mod d, fits in 32 bits
    return ((hi % d) * m + lo % d) % d;
  }
  constexpr auto operator&(uint128 rhs) const noexcept -> uint128 {
    return {hi & rhs.hi, lo & rhs.lo};
  }
  constexpr auto operator|(uint128 rhs) const noexcept -> uint128 {
    return {hi | rhs.hi, lo | rhs.lo};
  }
  constexpr auto operator^(uint128 rhs) const noexcept -> uint128 {
    return {hi ^ rhs.hi, lo ^ rhs.lo};
  }
  constexpr auto operator<<(int s) const noexcept -> uint128 {
    if (s == 0) return *this;
    if (s < 64) return {hi << s | lo >> (64 - s), lo << s};
    if (s < 128) return {lo << (s - 64), 0};
    return {0, 0};
  }
  constexpr auto operator>>(int s) const noexcept -> uint128 {
    if (s == 0) return *this;
    if (s < 64) return {hi >> s, lo >> s | hi << (64 - s)};
    if (s < 128) return {0, hi >> (s - 64)};
    return {0, 0};
  }

  constexpr auto operator==(uint128 rhs) const noexcept -> bool {
    return hi == rhs.hi && lo == rhs.lo;
  }
  constexpr auto operator!=(uint128 rhs) const noexcept -> bool {
    return hi != rhs.hi || lo != rhs.lo;
  }
  constexpr auto operator<(uint128 rhs) const noexcept -> bool {
    return hi != rhs.hi ? hi < rhs.hi : lo < rhs.lo;
  }
  constexpr auto operator>(uint128 rhs) const noexcept -> bool {
    return rhs < *this;
  }
  constexpr auto operator<=(uint128 rhs) const noexcept -> bool {
    return !(rhs < *this);
  }
  constexpr auto operator>=(uint128 rhs) const noexcept -> bool {
    return !(*this < rhs);
  }

  auto operator++() noexcept -> uint128& {
    if (++lo == 0) ++hi;
    return *this;
  }
};

#ifdef ZMIJ_USE_INT128
// Use the provided definition.
#elif defined(__SIZEOF_INT128__)
#  define ZMIJ_USE_INT128 1
#else
#  define ZMIJ_USE_INT128 0
#endif

#if ZMIJ_USE_INT128
using uint128_t = unsigned __int128;
#else
using uint128_t = uint128;
#endif  // ZMIJ_USE_INT128

// Divides x by 10 in place and returns the remainder.
inline auto divmod10(uint128_t& x) noexcept -> uint64_t {
#if ZMIJ_USE_INT128
  uint64_t r = uint64_t(x % 10);
  x /= 10;
  return r;
#else
  auto div = [](uint64_t& w, uint64_t rem) -> uint64_t {
    uint64_t hi = rem << 32 | w >> 32;
    uint64_t lo = (hi % 10) << 32 | uint32_t(w);
    w = (hi / 10) << 32 | (lo / 10);
    return lo % 10;
  };
  return div(x.lo, div(x.hi, 0));
#endif
}

// `buffer` params require at least buffer_sizes<Float> capacity;
// `out`/`n` params write at most `n` characters.

// Converts `value` to the shortest correctly rounded decimal (see to_decimal).
template <typename Float> auto to_decimal(Float value) noexcept -> dec_fp<>;

// Wide-significand variant for long double (see to_decimal).
template <typename Float>
auto to_decimal_big(Float value) noexcept -> dec_fp<uint128_t>;

template <typename Float>
auto write(char* buffer, Float value) noexcept -> char*;

// Writes the shortest decimal representation of `value`, correctly rounded
// (ties to even), into `out`, truncating after `n` chars. Returns the total
// length the result would need; if it exceeds `n` the output was truncated to
// the first `n` chars.
template <typename Float>
auto write_big(char* out, size_t n, Float value) noexcept -> size_t;

// Writes `value` in `fmt` notation with `precision` digits, correctly rounded
// (ties to even), into `out`, truncating after `n` chars. Returns the total
// length the result would need; if it exceeds `n` the output was truncated to
// the first `n` chars, and 0 on allocation failure (only possible for long
// double).
template <typename Float>
auto write_big(char* out, size_t n, Float value, int precision,
               format fmt) noexcept -> size_t;
template <>
inline auto write_big(char* out, size_t n, float value, int precision,
                      format fmt) noexcept -> size_t {
  return write_big(out, n, double(value), precision, fmt);
}

template <typename Float>
auto write_scientific(char* buffer, Float value, int precision) noexcept
    -> char*;

template <typename Float>
auto write_general(char* buffer, Float value, int precision) noexcept -> char*;

template <typename Float>
auto write_fixed(char* buffer, Float value, int precision) noexcept -> char*;

// Writes the decimal exponent as 'e', a sign and at least two digits, up to
// four (e.g. e+05 or e+4932, enough for extended long double).
auto write_big_exp(char* buffer, int dec_exp) noexcept -> char*;

// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
// its shortest form, e.g. -0x1.8p+1. If `prefix` is false the leading "0x" is
// omitted (e.g. -1.8p+1).
template <typename Float>
auto write_hex(char* buffer, Float value, bool prefix = true) noexcept -> char*;

// Writes `value` in hexadecimal floating-point notation with `precision` hex
// digits after the point, correctly rounded (ties to even), e.g. -0x1.80p+1,
// into `out`, truncating after `n` chars. Returns the total length the result
// would need. If `prefix` is false the leading "0x" is omitted.
template <typename Float>
auto write_hex(char* out, size_t n, Float value, int precision,
               bool prefix = true) noexcept -> size_t;

// When long double == double it has no explicit instantiations, so forward the
// long double detail writers to their double counterparts.
#if LDBL_MANT_DIG == DBL_MANT_DIG
// There is no write_big for double, so route shortest through write.
template <>
inline auto write_big(char* out, size_t n, long double value) noexcept
    -> size_t {
  char buffer[double_buffer_size];
  size_t size = size_t(write(buffer, double(value)) - buffer);
  memcpy(out, buffer, size < n ? size : n);
  return size;
}
template <>
inline auto write_big(char* out, size_t n, long double value, int precision,
                      format fmt) noexcept -> size_t {
  return write_big(out, n, double(value), precision, fmt);
}
template <>
inline auto write_hex(char* buffer, long double value, bool prefix) noexcept
    -> char* {
  return write_hex(buffer, double(value), prefix);
}
template <>
inline auto write_hex(char* out, size_t n, long double value, int precision,
                      bool prefix) noexcept -> size_t {
  return write_hex(out, n, double(value), precision, prefix);
}
#endif  // LDBL_MANT_DIG == DBL_MANT_DIG

// Returns the past-the-end pointer after writing min(size, n) chars to `out`.
inline auto clamp_end(char* out, size_t size, size_t n) noexcept -> char* {
  return out + (size < n ? size : n);
}

// Copies the result in [`buffer`, `end`) to `out`, truncating after `n` chars,
// and returns the past-the-end pointer.
inline auto copy_clamped(char* out, size_t n, const char* buffer,
                         const char* end) noexcept -> char* {
  size_t size = size_t(end - buffer);
  memcpy(out, buffer, size < n ? size : n);
  return clamp_end(out, size, n);
}

}  // namespace detail

/// Converts `value` into the shortest correctly rounded decimal representation.
/// Usage:
///   auto [sig, exp, negative] = to_decimal(6.62607015e-34);
inline auto to_decimal(float value) noexcept -> dec_fp<> {
  return detail::to_decimal(value);
}
inline auto to_decimal(double value) noexcept -> dec_fp<> {
  return detail::to_decimal(value);
}
inline auto to_decimal(long double value) noexcept
    -> dec_fp<detail::uint128_t> {
#if LDBL_MANT_DIG == DBL_MANT_DIG
  dec_fp<> dec = detail::to_decimal(double(value));
  return {dec.sig, dec.exp, dec.negative};
#else
  return detail::to_decimal_big(value);
#endif
}

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
inline auto write(char* out, size_t n, double value) noexcept -> char* {
  char buffer[double_buffer_size];
  if (n >= sizeof(buffer)) return detail::write(out, value);
  return detail::copy_clamped(out, n, buffer, detail::write(buffer, value));
}

inline auto write(char* out, size_t n, float value) noexcept -> char* {
  char buffer[float_buffer_size];
  if (n >= sizeof(buffer)) return detail::write(out, value);
  return detail::copy_clamped(out, n, buffer, detail::write(buffer, value));
}

inline auto write(char* out, size_t n, long double value) noexcept -> char* {
  if (LDBL_MANT_DIG == DBL_MANT_DIG) return write(out, n, double(value));
  return detail::clamp_end(out, detail::write_big(out, n, value), n);
}

/// Writes `value` in scientific format with `precision` digits after the
/// decimal point (e.g. 1.234e+05) to `out`, without a null terminator, like
/// printf's %e. A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written. The `long double`
/// overload returns nullptr on allocation failure.
inline auto write_scientific(char* out, size_t n, double value,
                             int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    auto size = detail::write_big(out, n, value, precision, format::scientific);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<double>::scientific];
  if (n >= sizeof(buffer))
    return detail::write_scientific(out, value, precision + 1);
  return detail::copy_clamped(
      out, n, buffer, detail::write_scientific(buffer, value, precision + 1));
}

inline auto write_scientific(char* out, size_t n, float value,
                             int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision >= 18) {
    auto size = detail::write_big(out, n, value, precision, format::scientific);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<float>::scientific];
  if (n >= sizeof(buffer))
    return detail::write_scientific(out, value, precision + 1);
  return detail::copy_clamped(
      out, n, buffer, detail::write_scientific(buffer, value, precision + 1));
}

inline auto write_scientific(char* out, size_t n, long double value,
                             int precision) noexcept -> char* {
  if (double(value) == value)
    return write_scientific(out, n, double(value), precision);
  if (precision < 0) precision = 6;
  auto size = detail::write_big(out, n, value, precision, format::scientific);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

/// Writes `value` in general format with up to `precision` significant digits
/// and no trailing zeros (e.g. 1.5 or 1.5e+20) to `out`, without a null
/// terminator. Fixed notation is used when `value`'s decimal exponent is in
/// [-4, precision), and scientific otherwise. A negative `precision` defaults
/// to 6 and zero is treated as 1, matching printf.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written. The `long double`
/// overload returns nullptr on allocation failure.
inline auto write_general(char* out, size_t n, double value,
                          int precision) noexcept -> char* {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    auto size = detail::write_big(out, n, value, precision, format::general);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<double>::scientific];
  if (n >= sizeof(buffer)) return detail::write_general(out, value, precision);
  return detail::copy_clamped(out, n, buffer,
                              detail::write_general(buffer, value, precision));
}

inline auto write_general(char* out, size_t n, float value,
                          int precision) noexcept -> char* {
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  if (precision > 18) {
    auto size = detail::write_big(out, n, value, precision, format::general);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<float>::scientific];
  if (n >= sizeof(buffer)) return detail::write_general(out, value, precision);
  return detail::copy_clamped(out, n, buffer,
                              detail::write_general(buffer, value, precision));
}

inline auto write_general(char* out, size_t n, long double value,
                          int precision) noexcept -> char* {
  if (double(value) == value)
    return write_general(out, n, double(value), precision);
  if (precision <= 1) precision = precision < 0 ? 6 : 1;
  auto size = detail::write_big(out, n, value, precision, format::general);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

/// Writes `value` in fixed notation with exactly `precision` digits after the
/// decimal point (e.g. 1.500) to `out`, without a null terminator. The result
/// is the exact value correctly rounded (ties to even), matching printf's %f.
/// A negative `precision` defaults to 6.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written. The `long double`
/// overload returns nullptr on allocation failure.
inline auto write_fixed(char* out, size_t n, double value,
                        int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    auto size = detail::write_big(out, n, value, precision, format::fixed);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<double>::fixed];
  if (n >= sizeof(buffer)) return detail::write_fixed(out, value, precision);
  return detail::copy_clamped(out, n, buffer,
                              detail::write_fixed(buffer, value, precision));
}

inline auto write_fixed(char* out, size_t n, float value,
                        int precision) noexcept -> char* {
  if (precision < 0) precision = 6;
  if (precision > 18) {
    auto size = detail::write_big(out, n, value, precision, format::fixed);
    return detail::clamp_end(out, size, n);
  }
  char buffer[buffer_sizes<float>::fixed];
  if (n >= sizeof(buffer)) return detail::write_fixed(out, value, precision);
  return detail::copy_clamped(out, n, buffer,
                              detail::write_fixed(buffer, value, precision));
}

inline auto write_fixed(char* out, size_t n, long double value,
                        int precision) noexcept -> char* {
  if (double(value) == value)
    return write_fixed(out, n, double(value), precision);
  if (precision < 0) precision = 6;
  auto size = detail::write_big(out, n, value, precision, format::fixed);
  return size != 0 ? detail::clamp_end(out, size, n) : nullptr;
}

/// Writes `value` in hexadecimal floating-point notation (like printf's %a) in
/// its shortest form (e.g. -0x1.8p+1) to `out`, without a null terminator.
///
/// Returns a pointer past the last character written; if the representation
/// exceeds `n` characters, only the first `n` are written.
inline auto write_hex(char* out, size_t n, double value) noexcept -> char* {
  char buffer[buffer_sizes<double>::hex];
  if (n >= sizeof(buffer)) return detail::write_hex(out, value);
  return detail::copy_clamped(out, n, buffer, detail::write_hex(buffer, value));
}

inline auto write_hex(char* out, size_t n, float value) noexcept -> char* {
  return write_hex(out, n, double(value));
}

inline auto write_hex(char* out, size_t n, long double value) noexcept
    -> char* {
  char buffer[buffer_sizes<long double>::hex];
  if (n >= sizeof(buffer)) return detail::write_hex(out, value);
  return detail::copy_clamped(out, n, buffer, detail::write_hex(buffer, value));
}

}  // namespace zmij

#endif  // ZMIJ_H_
