// A double-to-string conversion algorithm based on Schubfach and yy.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.
// https://github.com/vitaut/zmij/

#if __has_include("zmij.h")
#  include "zmij.h"
#else
namespace zmij {
struct dec_fp {
  long long sig;
  int exp;
};
}  // namespace zmij
#endif

#include <assert.h>  // assert
#include <stddef.h>  // size_t
#include <stdint.h>  // uint64_t
#include <string.h>  // memcpy

#include <limits>       // std::numeric_limits
#include <type_traits>  // std::conditional_t

#ifdef __ARM_NEON
#  include <arm_neon.h>
#endif

#ifdef _MSC_VER
#  include <intrin.h>  // __lzcnt64/_umul128/__umulh
#endif

#ifdef __clang__
#  pragma clang diagnostic ignored "-Wc++17-extensions"
#endif

#ifndef ZMIJ_USE_SIMD
#  define ZMIJ_USE_SIMD 1
#endif

#ifdef __has_builtin
#  define ZMIJ_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define ZMIJ_HAS_BUILTIN(x) 0
#endif
#ifdef __has_attribute
#  define ZMIJ_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#  define ZMIJ_HAS_ATTRIBUTE(x) 0
#endif
#ifdef __has_cpp_attribute
#  define ZMIJ_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#  define ZMIJ_HAS_CPP_ATTRIBUTE(x) 0
#endif

#if ZMIJ_HAS_CPP_ATTRIBUTE(likely) && ZMIJ_HAS_CPP_ATTRIBUTE(unlikely)
#  define ZMIJ_LIKELY likely
#  define ZMIJ_UNLIKELY unlikely
#else
#  define ZMIJ_LIKELY
#  define ZMIJ_UNLIKELY
#endif

#if ZMIJ_HAS_ATTRIBUTE(always_inline)
#  define ZMIJ_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#  define ZMIJ_INLINE __forceinline
#else
#  define ZMIJ_INLINE inline
#endif

namespace {

inline auto is_big_endian() noexcept -> bool {
  int n = 1;
  return *reinterpret_cast<char*>(&n) != 1;
}

inline auto bswap64(uint64_t x) noexcept -> uint64_t {
#if ZMIJ_HAS_BUILTIN(__builtin_bswap64)
  return __builtin_bswap64(x);
#elif defined(_MSC_VER)
  return _byteswap_uint64(x);
#else
  return ((x & 0xff00000000000000) >> 56) | ((x & 0x00ff000000000000) >> 40) |
         ((x & 0x0000ff0000000000) >> 24) | ((x & 0x000000ff00000000) >> +8) |
         ((x & 0x00000000ff000000) << +8) | ((x & 0x0000000000ff0000) << 24) |
         ((x & 0x000000000000ff00) << 40) | ((x & 0x00000000000000ff) << 56);
#endif
}

inline auto clz(uint64_t x) noexcept -> int {
  assert(x != 0);
#if ZMIJ_HAS_BUILTIN(__builtin_clzll)
  return __builtin_clzll(x);
#elif defined(_MSC_VER) && defined(__AVX2__) && defined(_M_AMD64)
  // Use lzcnt only on AVX2-capable CPUs that have this BMI instruction.
  return __lzcnt64(x);
#elif defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_ARM64))
  unsigned long idx;
  _BitScanReverse64(&idx, x);  // Fallback to the BSR instruction.
  return 63 - idx;
#elif defined(_MSC_VER)
  // Fallback to the 32-bit BSR instruction.
  unsigned long idx;
  if (_BitScanReverse(&idx, uint32_t(x >> 32))) return 31 - idx;
  _BitScanReverse(&idx, uint32_t(x));
  return 63 - idx;
#else
  int n = 64;
  for (; x > 0; x >>= 1) --n;
  return n;
#endif
}

struct uint128 {
  uint64_t hi;
  uint64_t lo;

  [[maybe_unused]] explicit constexpr operator uint64_t() const noexcept {
    return lo;
  }

  [[maybe_unused]] constexpr auto operator>>(int shift) const noexcept
      -> uint128 {
    if (shift == 32) {
      uint64_t hilo = uint32_t(hi);
      return {hi >> 32, (hilo << 32) | (lo >> 32)};
    }
    assert(shift >= 64 && shift < 128);
    return {0, hi >> (shift - 64)};
  }
};

[[maybe_unused]] inline auto operator+(uint128 lhs, uint128 rhs) noexcept
    -> uint128 {
#if defined(_MSC_VER) && defined(_M_AMD64)
  uint64_t lo, hi;
  _addcarry_u64(_addcarry_u64(0, lhs.lo, rhs.lo, &lo), lhs.hi, rhs.hi, &hi);
  return {hi, lo};
#else
  uint64_t lo = lhs.lo + rhs.lo;
  return {lhs.hi + rhs.hi + (lo < lhs.lo), lo};
#endif
}

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

// Computes 128-bit result of multiplication of two 64-bit unsigned integers.
constexpr auto umul128(uint64_t x, uint64_t y) noexcept -> uint128_t {
#if ZMIJ_USE_INT128
  return uint128_t(x) * y;
#else
#  if defined(_MSC_VER) && defined(_M_AMD64)
  if (!__builtin_is_constant_evaluated()) {
    uint64_t hi = 0;
    uint64_t lo = _umul128(x, y, &hi);
    return {hi, lo};
  }
#  elif defined(_MSC_VER) && defined(_M_ARM64)
  if (!__builtin_is_constant_evaluated()) {
    return {__umulh(x, y), x * y};
  }
#  endif
  uint64_t a = x >> 32;
  uint64_t b = uint32_t(x);
  uint64_t c = y >> 32;
  uint64_t d = uint32_t(y);

  uint64_t ac = a * c;
  uint64_t bc = b * c;
  uint64_t ad = a * d;
  uint64_t bd = b * d;

  uint64_t cs = (bd >> 32) + uint32_t(ad) + uint32_t(bc);  // cross sum
  return {ac + (ad >> 32) + (bc >> 32) + (cs >> 32), (cs << 32) + uint32_t(bd)};
#endif  // ZMIJ_USE_INT128
}

inline auto umul192_upper128(uint64_t x_hi, uint64_t x_lo, uint64_t y) noexcept
    -> uint128 {
  uint128_t p = umul128(x_hi, y);
  uint64_t lo = uint64_t(p) + uint64_t(umul128(x_lo, y) >> 64);
  return {uint64_t(p >> 64) + (lo < uint64_t(p)), lo};
}

constexpr auto umul128_upper64(uint64_t x, uint64_t y) noexcept -> uint64_t {
  return uint64_t(umul128(x, y) >> 64);
}

// Computes upper 64 bits of multiplication of x and y, discards the least
// significant bit and rounds to odd, where x = uint128_t(x_hi << 64) | x_lo.
auto umul_upper_inexact_to_odd(uint64_t x_hi, uint64_t x_lo,
                               uint64_t y) noexcept -> uint64_t {
  auto [hi, lo] = umul192_upper128(x_hi, x_lo, y);
  return hi | ((lo >> 1) != 0);
}
auto umul_upper_inexact_to_odd(uint64_t x_hi, uint64_t, uint32_t y) noexcept
    -> uint32_t {
  uint64_t result = uint64_t(umul128(x_hi, y) >> 32);
  return uint32_t(result >> 32) | ((uint32_t(result) >> 1) != 0);
}

template <typename Float> struct float_traits : std::numeric_limits<Float> {
  static_assert(float_traits::is_iec559, "IEEE 754 required");

  static constexpr int num_bits = float_traits::digits == 53 ? 64 : 32;
  static constexpr int num_sig_bits = float_traits::digits - 1;
  static constexpr int num_exp_bits = num_bits - num_sig_bits - 1;
  static constexpr int exp_mask = (1 << num_exp_bits) - 1;
  static constexpr int exp_bias = (1 << (num_exp_bits - 1)) - 1;

  using sig_type = std::conditional_t<num_bits == 64, uint64_t, uint32_t>;
  static constexpr sig_type implicit_bit = sig_type(1) << num_sig_bits;

  static auto to_bits(Float value) noexcept -> sig_type {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(value));
    return bits;
  }

  static auto get_sig(sig_type bits) noexcept -> sig_type {
    return bits & (implicit_bit - 1);
  }
  static auto get_exp(sig_type bits) noexcept -> int {
    return int(bits >> num_sig_bits) & exp_mask;
  }
};

// 128-bit significands of powers of 10 rounded down.
// Generated using 192-bit arithmetic method by Dougall Johnson.
constexpr struct pow10_significands_table {
  uint128 data[617];

  constexpr const uint128& operator[](int i) const { return data[i]; }

  constexpr pow10_significands_table() : data() {
    struct uint192 {
      uint64_t w0, w1, w2;  // w0 = least significant, w2 = most significant
    };

    // first element, rounded up to cancel out rounding down in the
    // multiplication, and minimise significant bits
    uint192 current = {0xe000000000000000, 0x25e8e89c13bb0f7a,
                       0xff77b1fcbebcdc4f};
    uint64_t ten = 0xa000000000000000;
    for (auto& datum : data) {
      datum = {current.w2, current.w1};

      uint64_t h0 = umul128_upper64(current.w0, ten);
      uint64_t h1 = umul128_upper64(current.w1, ten);

      uint64_t c0 = h0 + current.w1 * ten;
      uint64_t c1 = (c0 < h0) + h1 + current.w2 * ten;
      uint64_t c2 =
          (c1 < h1) + umul128_upper64(current.w2, ten);  // dodgy carry

      // normalise
      if (c2 >> 63)
        current = {c0, c1, c2};
      else
        current = {c0 << 1, c1 << 1 | c0 >> 63, c2 << 1 | c1 >> 63};
    }
  }
} pow10_significands;

constexpr int dec_exp_min = -292;

inline auto count_trailing_nonzeros(uint64_t x) noexcept -> int {
  // We count the number of bytes until there are only zeros left.
  // The code is equivalent to
  //   return 8 - clz(x) / 8
  // but if the BSR instruction is emitted (as gcc on x64 does with
  // default settings), subtracting the constant before dividing allows
  // the compiler to combine it with the subtraction which it inserts
  // due to BSR counting in the opposite direction.
  //
  // Additionally, the BSR instruction requires a zero check.  Since the
  // high bit is unused we can avoid the zero check by shifting the
  // datum left by one and inserting a sentinel bit at the end. This can
  // be faster than the automatically inserted range check.
  if (is_big_endian()) x = bswap64(x);
  return (70 - clz((x << 1) | 1)) / 8;
}

// Converts value in the range [0, 100) to a string. GCC generates a bit better
// code when value is pointer-size (https://www.godbolt.org/z/5fEPMT1cc).
inline auto digits2(size_t value) noexcept -> const char* {
  // Align data since unaligned access may be slower when crossing a
  // hardware-specific boundary.
  alignas(2) static const char data[] =
      "0001020304050607080910111213141516171819"
      "2021222324252627282930313233343536373839"
      "4041424344454647484950515253545556575859"
      "6061626364656667686970717273747576777879"
      "8081828384858687888990919293949596979899";
  return &data[value * 2];
}

auto to_bcd8(uint64_t abcdefgh) noexcept -> uint64_t {
  // An optimization from Xiang JunBo.
  // Three steps BCD.  Base 10000 -> base 100 -> base 10.
  // div and mod are evaluated simultaneously as, e.g.
  //   (abcdefgh / 10000) << 32 + (abcdefgh % 10000)
  //      == abcdefgh + (2^32 - 10000) * (abcdefgh / 10000)))
  // where the division on the RHS is implemented by the usual multiply + shift
  // trick and the fractional bits are masked away.
  uint64_t abcd_efgh =
      abcdefgh + (0x100000000 - 10000) * ((abcdefgh * 0x68db8bb) >> 40);
  uint64_t ab_cd_ef_gh =
      abcd_efgh +
      (0x10000 - 100) * (((abcd_efgh * 0x147b) >> 19) & 0x7f0000007f);
  uint64_t a_b_c_d_e_f_g_h =
      ab_cd_ef_gh +
      (0x100 - 10) * (((ab_cd_ef_gh * 0x67) >> 10) & 0xf000f000f000f);
  return is_big_endian() ? a_b_c_d_e_f_g_h : bswap64(a_b_c_d_e_f_g_h);
}

inline auto write_if_nonzero(char* buffer, uint32_t digit) noexcept -> char* {
  *buffer = char('0' + digit);
  return buffer + (digit != 0);
}

inline void write8(char* buffer, uint64_t value) noexcept {
  memcpy(buffer, &value, 8);
}

constexpr uint64_t zeros = 0x0101010101010101u * '0';

// Writes a significand consisting of up to 17 decimal digits (16-17 for
// normals) and removes trailing zeros.
auto write_significand17(char* buffer, uint64_t value) noexcept -> char* {
#if !defined(__ARM_NEON) || !ZMIJ_USE_SIMD
  char* start = buffer;
  // Each digit is denoted by a letter so value is abbccddeeffgghhii.
  uint32_t abbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);
  buffer = write_if_nonzero(buffer, abbccddee / 100'000'000);
  uint64_t bcd = to_bcd8(abbccddee % 100'000'000);
  write8(buffer, bcd | zeros);
  if (ffgghhii == 0) {
    buffer += count_trailing_nonzeros(bcd);
    return buffer - int(buffer - start == 1);
  }
  bcd = to_bcd8(ffgghhii);
  write8(buffer + 8, bcd | zeros);
  return buffer + 8 + count_trailing_nonzeros(bcd);
#else   // __ARM_NEON
  // An optimized version for NEON by Dougall Johnson.
  struct to_string_constants {
    uint64_t mul_const;
    uint64_t hundred_million;
    int32x4_t multipliers32;
    int16x8_t multipliers16;
  };

  static const to_string_constants constants = {
      .mul_const = 0xabcc77118461cefd,
      .hundred_million = 100000000,
      .multipliers32 = {0x68db8bb, -10000 + 0x10000, 0x147b000, -100 + 0x10000},
      .multipliers16 = {0xce0, -10 + 0x100},
  };

  const to_string_constants* c = &constants;

  // Compiler barrier, or clang doesn't load from memory and generates 15 more
  // instructions
  asm("" : "+r"(c));

  uint64_t hundred_million = c->hundred_million;

  // Compiler barrier, or clang narrows the load to 32-bit and unpairs it.
  asm("" : "+r"(hundred_million));

  // Equivalent to abbccddee = value / 100000000, ffgghhii = value % 100000000.
  uint64_t abbccddee = (__uint128_t(value) * c->mul_const) >> 90;
  uint64_t ffgghhii = value - abbccddee * hundred_million;

  // We could probably make this bit faster, but we're preferring to
  // reuse the constants for now.
  uint64_t a = (__uint128_t(abbccddee) * c->mul_const) >> 90;
  abbccddee -= a * hundred_million;

  char* start = buffer;
  buffer = write_if_nonzero(buffer, a);

  uint64x1_t hundredmillions = {abbccddee | (uint64_t(ffgghhii) << 32)};

  int32x2_t high_10000 =
      vshr_n_u32(vqdmulh_n_s32(hundredmillions, c->multipliers32[0]), 9);
  int32x2_t tenthousands =
      vmla_n_s32(hundredmillions, high_10000, c->multipliers32[1]);

  int32x4_t extended = vshll_n_u16(tenthousands, 0);

  // Compiler barrier, or clang breaks the subsequent MLA into UADDW + MUL.
  asm("" : "+w"(extended));

  int32x4_t high_100 = vqdmulhq_n_s32(extended, c->multipliers32[2]);
  int32x4_t hundreds = vmlaq_n_s32(extended, high_100, c->multipliers32[3]);
  int16x8_t high_10 = vqdmulhq_n_s16(hundreds, c->multipliers16[0]);
  int16x8_t digits =
      vrev64q_u8(vmlaq_n_s16(hundreds, high_10, c->multipliers16[1]));
  int16x8_t ascii = vaddq_u16(digits, vdupq_n_s8('0'));

  memcpy(buffer, &ascii, 16);

  uint16x8_t is_zero = vceqzq_u8(digits);
  uint64_t zeroes =
      ~vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(is_zero, 4)), 0);

  buffer += 16 - ((zeroes != 0 ? clz(zeroes) : 64) >> 2);
  return buffer - int(buffer - start == 1);
#endif  // __ARM_NEON
}

// Writes a significand consisting of up to 9 decimal digits (7-9 for normals)
// and removes trailing zeros.
auto write_significand9(char* buffer, uint32_t value) noexcept -> char* {
  char* start = buffer;
  buffer = write_if_nonzero(buffer, value / 100'000'000);
  uint64_t bcd = to_bcd8(value % 100'000'000);
  write8(buffer, bcd | zeros);
  buffer += count_trailing_nonzeros(bcd);
  return buffer - int(buffer - start == 1);
}

// Computes the decimal exponent as floor(log10(2**bin_exp)) if regular or
// floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
constexpr auto compute_dec_exp(int bin_exp, bool regular) noexcept -> int {
  assert(bin_exp >= -1334 && bin_exp <= 2620);
  // log10_3_over_4_sig = -log10(3/4) * 2**log10_2_exp rounded to a power of 2
  constexpr int log10_3_over_4_sig = 131'072;
  // log10_2_sig = round(log10(2) * 2**log10_2_exp)
  constexpr int log10_2_sig = 315'653;
  constexpr int log10_2_exp = 20;
  return (bin_exp * log10_2_sig - !regular * log10_3_over_4_sig) >> log10_2_exp;
}

// Computes a shift so that, after scaling by a power of 10, the intermediate
// result always has a fixed 128-bit fractional part (for double).
//
// Different binary exponents can map to the same decimal exponent, but place
// the decimal point at different bit positions. The shift compensates for this.
//
// For example, both 3 * 2**59 and 3 * 2**60 have dec_exp = 2, but dividing by
// 10^dec_exp puts the decimal point in different bit positions:
//   3 * 2**59 / 100 = 1.72...e+16  (needs shift = 1 + 1)
//   3 * 2**60 / 100 = 3.45...e+16  (needs shift = 2 + 1)
constexpr ZMIJ_INLINE auto compute_exp_shift(int bin_exp, int dec_exp) noexcept
    -> int {
  assert(dec_exp >= -350 && dec_exp <= 350);
  // log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
  constexpr int log2_pow10_sig = 217'707, log2_pow10_exp = 16;
  // pow10_bin_exp = floor(log2(10**-dec_exp))
  int pow10_bin_exp = -dec_exp * log2_pow10_sig >> log2_pow10_exp;
  // pow10 = ((pow10_hi << 64) | pow10_lo) * 2**(pow10_bin_exp - 127)
  return bin_exp + pow10_bin_exp + 1;
}

template <int num_bits>
auto normalize(zmij::dec_fp dec, bool subnormal) noexcept -> zmij::dec_fp {
  if (!subnormal) [[ZMIJ_LIKELY]]
    return dec;
  while (dec.sig < (num_bits == 64 ? uint64_t(1e16) : uint64_t(1e8))) {
    dec.sig *= 10;
    --dec.exp;
  }
  return dec;
}

// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation.
template <typename UInt>
ZMIJ_INLINE auto to_decimal(UInt bin_sig, int bin_exp, int dec_exp,
                            bool regular, bool subnormal) noexcept
    -> zmij::dec_fp {
  constexpr int num_bits = std::numeric_limits<UInt>::digits;
  if (regular & !subnormal) [[ZMIJ_LIKELY]] {
    int exp_shift = compute_exp_shift(bin_exp, dec_exp);
    auto [pow10_hi, pow10_lo] = pow10_significands[-dec_exp - dec_exp_min];

    UInt integral = 0;        // integral part of bin_sig * pow10
    uint64_t fractional = 0;  // fractional part of bin_sig * pow10
    if (num_bits == 64) {
      auto [i, f] = umul192_upper128(pow10_hi, pow10_lo, bin_sig << exp_shift);
      integral = i;
      fractional = f;
    } else {
      uint128_t result = umul128(pow10_hi, bin_sig << exp_shift);
      integral = uint64_t(result >> 64);
      fractional = uint64_t(result);
    }
#if ZMIJ_USE_INT128
    // An optimization of integral % 10 by Dougall Johnson.
    // Relies on range calculation: (max_bin_sig << max_exp_shift) * max_u128.
    uint64_t div10 = (integral * ((uint128_t(1) << 64) / 10 + 1)) >> 64;
    uint64_t digit = integral - div10 * 10;
    asm("" : "+r"(digit));  // or it narrows to 32-bit and doesn't use madd/msub
#else
    uint64_t digit = integral % 10;
#endif

    // Switch to a fixed-point representation with the least significant
    // integral digit in the upper bits and fractional digits in the lower bits.
    constexpr int num_integral_bits = num_bits == 64 ? 4 : 32;
    constexpr int num_fractional_bits = 64 - num_integral_bits;
    constexpr uint64_t ten = uint64_t(10) << num_fractional_bits;
    // Fixed-point remainder of the scaled significand modulo 10.
    uint64_t scaled_sig_mod10 =
        (digit << num_fractional_bits) | (fractional >> num_integral_bits);

    // scaled_half_ulp = 0.5 * pow10 in the fixed-point format.
    // dec_exp is chosen so that 10**dec_exp <= 2**bin_exp < 10**(dec_exp + 1).
    // Since 1ulp == 2**bin_exp it will be in the range [1, 10) after scaling
    // by 10**dec_exp. Add 1 to combine the shift with division by two.
    uint64_t scaled_half_ulp = pow10_hi >> (num_integral_bits - exp_shift + 1);
    uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;
    constexpr uint64_t half_ulp = uint64_t(1) << 63;

    // An optimization from yy by Yaoyuan Guo:
    if (
        // Exact half-ulp tie when rounding to nearest integer.
        fractional != half_ulp &&
        // Exact half-ulp tie when rounding to nearest 10.
        scaled_sig_mod10 != scaled_half_ulp &&
        // Near-boundary case for rounding to nearest 10.
        ten - upper > 1u) [[ZMIJ_LIKELY]] {
      bool round_up = upper >= ten;
      int64_t shorter = int64_t(integral - digit + round_up * 10);
      int64_t longer = int64_t(integral + (fractional >= half_ulp));
      bool use_shorter = (scaled_sig_mod10 <= scaled_half_ulp) + round_up != 0;
      return {use_shorter ? shorter : longer, dec_exp};
    }
  }

  dec_exp = compute_dec_exp(bin_exp, regular);
  int exp_shift = compute_exp_shift(bin_exp, dec_exp);
  auto [pow10_hi, pow10_lo] = pow10_significands[-dec_exp - dec_exp_min];

  // Fallback to Schubfach to guarantee correctness in boundary cases.
  // This requires switching to strict overestimates of powers of 10.
  ++(num_bits == 64 ? pow10_lo : pow10_hi);

  // Shift the significand so that boundaries are integer.
  constexpr int bound_shift = 2;
  UInt bin_sig_shifted = bin_sig << bound_shift;

  // Compute the estimates of lower and upper bounds of the rounding interval
  // by multiplying them by the power of 10 and applying modified rounding.
  UInt lsb = bin_sig & 1;
  UInt lower = (bin_sig_shifted - (regular + 1)) << exp_shift;
  lower = umul_upper_inexact_to_odd(pow10_hi, pow10_lo, lower) + lsb;
  UInt upper = (bin_sig_shifted + 2) << exp_shift;
  upper = umul_upper_inexact_to_odd(pow10_hi, pow10_lo, upper) - lsb;

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  UInt shorter = 10 * ((upper >> bound_shift) / 10);
  if ((shorter << bound_shift) >= lower)
    return normalize<num_bits>({int64_t(shorter), dec_exp}, subnormal);

  UInt scaled_sig = umul_upper_inexact_to_odd(pow10_hi, pow10_lo,
                                              bin_sig_shifted << exp_shift);
  UInt dec_sig_below = scaled_sig >> bound_shift;
  UInt dec_sig_above = dec_sig_below + 1;

  // Pick the closest of dec_sig_below and dec_sig_above and check if it's in
  // the rounding interval.
  using sint = std::make_signed_t<UInt>;
  sint cmp = sint(scaled_sig - ((dec_sig_below + dec_sig_above) << 1));
  bool below_closer = cmp < 0 || (cmp == 0 && (dec_sig_below & 1) == 0);
  bool below_in = (dec_sig_below << bound_shift) >= lower;
  UInt dec_sig = (below_closer & below_in) ? dec_sig_below : dec_sig_above;
  return normalize<num_bits>({int64_t(dec_sig), dec_exp}, subnormal);
}

}  // namespace

namespace zmij {

inline auto to_decimal(double value) noexcept -> dec_fp {
  using traits = float_traits<double>;
  auto bits = traits::to_bits(value);
  auto bin_sig = traits::get_sig(bits);  // binary significand
  auto bin_exp = traits::get_exp(bits);  // binary exponent
  bool regular = bin_sig != 0;
  bool subnormal = false;
  if (((bin_exp + 1) & traits::exp_mask) <= 1) [[ZMIJ_UNLIKELY]] {
    if (bin_exp != 0) return {0, int(~0u >> 1)};
    if (bin_sig == 0) return {0, 0};
    // Handle subnormals.
    bin_sig |= traits::implicit_bit;
    bin_exp = 1;
    subnormal = true;
    regular = true;
  }
  bin_sig ^= traits::implicit_bit;
  bin_exp -= traits::num_sig_bits + traits::exp_bias;
  auto [dec_sig, dec_exp] = ::to_decimal(
      bin_sig, bin_exp, compute_dec_exp(bin_exp, true), regular, subnormal);
  return {bits >> (traits::num_bits - 1) ? -dec_sig : dec_sig, dec_exp};
}

namespace detail {

// It is slightly faster to return a pointer to the end than the size.
template <typename Float>
auto write(Float value, char* buffer) noexcept -> char* {
  using traits = float_traits<Float>;
  auto bits = traits::to_bits(value);
  auto raw_exp = traits::get_exp(bits);  // binary exponent
  auto bin_exp = raw_exp - traits::num_sig_bits - traits::exp_bias;
  // Compute the decimal exponent early to overlap its latency with other work.
  auto dec_exp = compute_dec_exp(bin_exp, true);

  *buffer = '-';
  buffer += bits >> (traits::num_bits - 1);

  auto bin_sig = traits::get_sig(bits);  // binary significand
  bool regular = bin_sig != 0;
  bool subnormal = false;
  if (((raw_exp + 1) & traits::exp_mask) <= 1) [[ZMIJ_UNLIKELY]] {
    if (raw_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return buffer + 3;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return buffer + 1;
    }
    // Handle subnormals.
    bin_exp = 1 - traits::num_sig_bits - traits::exp_bias;
    dec_exp = compute_dec_exp(bin_exp, true);
    bin_sig |= traits::implicit_bit;
    subnormal = true;
    // Setting regular is not redundant: it has a measurable perf impact.
    regular = true;
  }
  bin_sig ^= traits::implicit_bit;

  // Here be 🐉s.
  auto dec = ::to_decimal(bin_sig, bin_exp, dec_exp, regular, subnormal);
  dec_exp = dec.exp;
  uint64_t dec_sig = dec.sig;

  // Write significand.
  char* start = buffer;
  if (traits::num_bits == 64) {
    dec_exp += traits::max_digits10 + (dec_sig >= uint64_t(1e16)) - 2;
    buffer = write_significand17(buffer + 1, dec_sig);
  } else {
    if (dec_sig < uint32_t(1e7)) [[ZMIJ_UNLIKELY]] {
      dec_sig *= 10;
      --dec_exp;
    }
    dec_exp += traits::max_digits10 + (dec_sig >= uint32_t(1e8)) - 2;
    buffer = write_significand9(buffer + 1, dec_sig);
  }
  start[0] = start[1];
  start[1] = '.';

  // Write exponent.
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian()) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  int mask = (dec_exp >= 0) - 1;
  dec_exp = ((dec_exp + mask) ^ mask);  // absolute value
  if constexpr (traits::min_exponent10 >= -99 && traits::max_exponent10 <= 99) {
    memcpy(buffer, digits2(dec_exp), 2);
    buffer[2] = '\0';
    return buffer + 2;
  }
  // 19 is faster or equal to 12 even for 3 digits.
  constexpr int div_exp = 19, div_sig = (1 << div_exp) / 100 + 1;
  uint32_t digit = (uint32_t(dec_exp) * div_sig) >> div_exp;  // value / 100
  uint32_t digit_with_nuls = '0' + digit;
  if (is_big_endian()) digit_with_nuls <<= 24;
  memcpy(buffer, &digit_with_nuls, 4);
  buffer += dec_exp >= 100;
  memcpy(buffer, digits2(dec_exp - digit * 100), 2);
  return buffer + 2;
}

template auto write(double value, char* buffer) noexcept -> char*;
template auto write(float value, char* buffer) noexcept -> char*;

}  // namespace detail
}  // namespace zmij
