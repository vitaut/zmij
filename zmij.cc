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

#ifndef ZMIJ_USE_SIMD
#  define ZMIJ_USE_SIMD 1
#endif

#ifdef ZMIJ_USE_NEON
// Use the provided definition
#elif defined(__ARM_NEON) || defined(_M_ARM64)
#  define ZMIJ_USE_NEON ZMIJ_USE_SIMD
#else
#  define ZMIJ_USE_NEON 0
#endif
#if ZMIJ_USE_NEON
#  include <arm_neon.h>
#endif

#ifdef ZMIJ_USE_SSE
// Use the provided definition
#elif defined(__SSE2__)
#  define ZMIJ_USE_SSE ZMIJ_USE_SIMD
#elif defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86FP == 2)
#  define ZMIJ_USE_SSE ZMIJ_USE_SIMD
#else
#  define ZMIJ_USE_SSE 0
#endif
#if ZMIJ_USE_SSE
#  include <immintrin.h>
#endif

#ifdef ZMIJ_USE_SSE4_1
// Use the provided definition
static_assert(!ZMIJ_USE_SSE4_1 || ZMIJ_USE_SSE);
#elif defined(__SSE4_1__) || defined(__AVX__)
// On MSVC there's no way to check for SSE4.1 specifically so check __AVX__.
#  define ZMIJ_USE_SSE4_1 ZMIJ_USE_SSE
#else
#  define ZMIJ_USE_SSE4_1 0
#endif

#ifdef __aarch64__
#  define ZMIJ_AARCH64 1
#else
#  define ZMIJ_AARCH64 0
#endif

#ifdef _MSC_VER
#  define ZMIJ_MSC_VER _MSC_VER
#  include <intrin.h>  // __lzcnt64/_umul128/__umulh
#else
#  define ZMIJ_MSC_VER 0
#endif

#if defined(__has_builtin) && !defined(ZMIJ_NO_BUILTINS)
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

#if ZMIJ_HAS_CPP_ATTRIBUTE(maybe_unused)
#  define ZMIJ_MAYBE_UNUSED maybe_unused
#else
#  define ZMIJ_MAYBE_UNUSED
#endif

#if ZMIJ_HAS_ATTRIBUTE(always_inline)
#  define ZMIJ_INLINE __attribute__((always_inline)) inline
#elif ZMIJ_MSC_VER
#  define ZMIJ_INLINE __forceinline
#else
#  define ZMIJ_INLINE inline
#endif

#ifdef __GNUC__
#  define ZMIJ_ASM(x) asm x
#else
#  define ZMIJ_ASM(x)
#endif

namespace {

#ifdef __cpp_lib_is_constant_evaluated
using std::is_constant_evaluated;
#  define ZMIJ_CONSTEXPR constexpr
#else
auto is_constant_evaluated() -> bool { return false; }
#  define ZMIJ_CONSTEXPR
#endif

inline auto is_big_endian() noexcept -> bool {
  int n = 1;
  return *reinterpret_cast<char*>(&n) != 1;
}

inline auto bswap64(uint64_t x) noexcept -> uint64_t {
#if ZMIJ_HAS_BUILTIN(__builtin_bswap64)
  return __builtin_bswap64(x);
#elif ZMIJ_MSC_VER
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
#elif defined(_M_AMD64) && defined(__AVX2__)
  // Use lzcnt only on AVX2-capable CPUs that have this BMI instruction.
  return __lzcnt64(x);
#elif defined(_M_AMD64) || defined(_M_ARM64)
  unsigned long idx;
  _BitScanReverse64(&idx, x);  // Fallback to the BSR instruction.
  return 63 - idx;
#elif ZMIJ_MSC_VER
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

  [[ZMIJ_MAYBE_UNUSED]] explicit constexpr operator uint64_t() const noexcept {
    return lo;
  }

  [[ZMIJ_MAYBE_UNUSED]] constexpr auto operator>>(int shift) const noexcept
      -> uint128 {
    if (shift == 32) {
      uint64_t hilo = uint32_t(hi);
      return {hi >> 32, (hilo << 32) | (lo >> 32)};
    }
    assert(shift >= 64 && shift < 128);
    return {0, hi >> (shift - 64)};
  }
};

[[ZMIJ_MAYBE_UNUSED]] inline auto operator+(uint128 lhs, uint128 rhs) noexcept
    -> uint128 {
#ifdef _M_AMD64
  uint64_t lo, hi;
  _addcarry_u64(_addcarry_u64(0, lhs.lo, rhs.lo, &lo), lhs.hi, rhs.hi, &hi);
  return {hi, lo};
#else
  uint64_t lo = lhs.lo + rhs.lo;
  return {lhs.hi + rhs.hi + (lo < lhs.lo), lo};
#endif  // _M_AMD64
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

#if ZMIJ_USE_INT128 && defined(__APPLE__)
constexpr bool use_umul128_hi64 = true;  // Use umul128_hi64 for division.
#else
constexpr bool use_umul128_hi64 = false;
#endif

// Computes 128-bit result of multiplication of two 64-bit unsigned integers.
constexpr auto umul128(uint64_t x, uint64_t y) noexcept -> uint128_t {
#if ZMIJ_USE_INT128
  return uint128_t(x) * y;
#else
#  if defined(_M_AMD64)
  if (!__builtin_is_constant_evaluated()) {
    uint64_t hi = 0;
    uint64_t lo = _umul128(x, y, &hi);
    return {hi, lo};
  }
#  elif defined(_M_ARM64)
  if (!__builtin_is_constant_evaluated()) return {__umulh(x, y), x * y};
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

constexpr auto umul128_hi64(uint64_t x, uint64_t y) noexcept -> uint64_t {
  return uint64_t(umul128(x, y) >> 64);
}

inline auto umul192_hi128(uint64_t x_hi, uint64_t x_lo, uint64_t y) noexcept
    -> uint128 {
  uint128_t p = umul128(x_hi, y);
  uint64_t lo = uint64_t(p) + uint64_t(umul128(x_lo, y) >> 64);
  return {uint64_t(p >> 64) + (lo < uint64_t(p)), lo};
}

// Computes high 64 bits of multiplication of x and y, discards the least
// significant bit and rounds to odd, where x = uint128_t(x_hi << 64) | x_lo.
auto umulhi_inexact_to_odd(uint64_t x_hi, uint64_t x_lo, uint64_t y) noexcept
    -> uint64_t {
  uint128 p = umul192_hi128(x_hi, x_lo, y);
  return p.hi | ((p.lo >> 1) != 0);
}
auto umulhi_inexact_to_odd(uint64_t x_hi, uint64_t, uint32_t y) noexcept
    -> uint32_t {
  uint64_t p = uint64_t(umul128(x_hi, y) >> 32);
  return uint32_t(p >> 32) | ((uint32_t(p) >> 1) != 0);
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

  static auto is_negative(sig_type bits) noexcept -> bool {
    return bits >> (num_bits - 1);
  }
  static auto get_sig(sig_type bits) noexcept -> sig_type {
    return bits & (implicit_bit - 1);
  }
  static auto get_exp(sig_type bits) noexcept -> int64_t {
    return int64_t((bits << 1) >> (num_sig_bits + 1));
  }
};

// 128-bit significands of powers of 10 rounded down.
// Generated using 192-bit arithmetic method by Dougall Johnson.
struct pow10_significands_table {
  static constexpr bool split_tables = ZMIJ_AARCH64 != 0;
  static constexpr int num_pow10 = 617;
  uint64_t data[num_pow10 * 2] = {};

  ZMIJ_CONSTEXPR auto operator[](int dec_exp) const noexcept -> uint128 {
    constexpr int dec_exp_min = -292;
    if (!split_tables) {
      int index = (dec_exp - dec_exp_min) * 2;
      return {data[index], data[index + 1]};
    }

    const uint64_t* hi = data + num_pow10 + dec_exp_min - 1;
    const uint64_t* lo = hi + num_pow10;

    // Force indexed loads.
    if (!is_constant_evaluated()) ZMIJ_ASM(volatile("" : "+r"(hi), "+r"(lo)));
    return {hi[-dec_exp], lo[-dec_exp]};
  }

  constexpr pow10_significands_table() {
    struct uint192 {
      uint64_t w0, w1, w2;  // w0 = least significant, w2 = most significant
    };

    // First element, rounded up to cancel out rounding down in the
    // multiplication, and minimize significant bits.
    uint192 current = {0xe000000000000000, 0x25e8e89c13bb0f7a,
                       0xff77b1fcbebcdc4f};
    uint64_t ten = 0xa000000000000000;
    for (int i = 0; i < num_pow10; ++i) {
      if (split_tables) {
        data[num_pow10 - i - 1] = current.w2;
        data[num_pow10 * 2 - i - 1] = current.w1;
      } else {
        data[i * 2] = current.w2;
        data[i * 2 + 1] = current.w1;
      }

      uint64_t h0 = umul128_hi64(current.w0, ten);
      uint64_t h1 = umul128_hi64(current.w1, ten);

      uint64_t c0 = h0 + current.w1 * ten;
      uint64_t c1 = (c0 < h0) + h1 + current.w2 * ten;
      uint64_t c2 = (c1 < h1) + umul128_hi64(current.w2, ten);  // dodgy carry

      // normalise
      if (c2 >> 63)
        current = {c0, c1, c2};
      else
        current = {c0 << 1, c1 << 1 | c0 >> 63, c2 << 1 | c1 >> 63};
    }
  }
};
alignas(64) constexpr pow10_significands_table pow10_significands;

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

constexpr ZMIJ_INLINE auto do_compute_exp_shift(int bin_exp,
                                                int dec_exp) noexcept
    -> unsigned char {
  assert(dec_exp >= -350 && dec_exp <= 350);
  // log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
  constexpr int log2_pow10_sig = 217'707, log2_pow10_exp = 16;
  // pow10_bin_exp = floor(log2(10**-dec_exp))
  int pow10_bin_exp = -dec_exp * log2_pow10_sig >> log2_pow10_exp;
  // pow10 = ((pow10_hi << 64) | pow10_lo) * 2**(pow10_bin_exp - 127)
  return bin_exp + pow10_bin_exp + 1;
}

struct exp_shift_table {
  using traits = float_traits<double>;
  static constexpr bool enable = true;
  static constexpr int num_exps = traits::exp_mask + 1;
  static constexpr int offset = traits::num_sig_bits + traits::exp_bias;
  unsigned char data[enable ? num_exps : 1] = {};

  constexpr exp_shift_table() {
    if (!enable) return;
    for (int raw_exp = 0; raw_exp < num_exps; ++raw_exp) {
      int bin_exp = raw_exp - offset;
      if (raw_exp == 0) ++bin_exp;
      int dec_exp = compute_dec_exp(bin_exp, true);
      data[raw_exp] = do_compute_exp_shift(bin_exp, dec_exp);
    }
  }
};
constexpr exp_shift_table exp_shifts;

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
template <int num_bits, bool only_regular = false>
constexpr ZMIJ_INLINE auto compute_exp_shift(int bin_exp, int dec_exp) noexcept
    -> unsigned char {
  if (num_bits == 64 && exp_shift_table::enable && only_regular)
    return exp_shifts.data[bin_exp + exp_shift_table::offset];
  return do_compute_exp_shift(bin_exp, dec_exp);
}

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
  return (size_t(70) - clz((x << 1) | 1)) / 8;  // size_t for native arithmetic
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

constexpr int div10k_exp = 40;
constexpr uint32_t div10k_sig = uint32_t((1ull << div10k_exp) / 10000 + 1);
constexpr uint32_t neg10k = uint32_t((1ull << 32) - 10000);
constexpr int div100_exp = 19;
constexpr uint32_t div100_sig = (1 << div100_exp) / 100 + 1;
constexpr uint32_t neg100 = (1 << 16) - 100;
constexpr int div10_exp = 10;
constexpr uint32_t div10_sig = (1 << div10_exp) / 10 + 1;
constexpr uint32_t neg10 = (1 << 8) - 10;
// (1 << 63) / 5 == (1 << 64) / 10 without an intermediate int128.
constexpr uint64_t div10_sig64 = (1ull << 63) / 5 + 1;

constexpr uint64_t zeros = 0x0101010101010101u * '0';

auto to_bcd8(uint64_t abcdefgh) noexcept -> uint64_t {
  // An optimization from Xiang JunBo.
  // Three steps BCD. Base 10000 -> base 100 -> base 10.
  // div and mod are evaluated simultaneously as, e.g.
  //   (abcdefgh / 10000) << 32 + (abcdefgh % 10000)
  //      == abcdefgh + (2**32 - 10000) * (abcdefgh / 10000)))
  // where the division on the RHS is implemented by the usual multiply + shift
  // trick and the fractional bits are masked away.
  uint64_t abcd_efgh =
      abcdefgh + neg10k * ((abcdefgh * div10k_sig) >> div10k_exp);
  uint64_t ab_cd_ef_gh =
      abcd_efgh +
      neg100 * (((abcd_efgh * div100_sig) >> div100_exp) & 0x7f0000007f);
  uint64_t a_b_c_d_e_f_g_h =
      ab_cd_ef_gh +
      neg10 * (((ab_cd_ef_gh * div10_sig) >> div10_exp) & 0xf000f000f000f);
  return is_big_endian() ? a_b_c_d_e_f_g_h : bswap64(a_b_c_d_e_f_g_h);
}

inline auto write_if(char* buffer, uint32_t digit, bool condition) noexcept
    -> char* {
  *buffer = char('0' + digit);
  return buffer + condition;
}

inline void write8(char* buffer, uint64_t value) noexcept {
  memcpy(buffer, &value, 8);
}

// Writes a significand consisting of up to 17 decimal digits (16-17 for
// normals) and removes trailing zeros.  The significant digits start
// from buffer[1].  buffer[0] may contain '0' after this function if
// the significand has length 16.
auto write_significand17(char* buffer, uint64_t value,
                         bool has17digits) noexcept -> char* {
#if ZMIJ_USE_NEON
  // An optimized version for NEON by Dougall Johnson.
  constexpr int32_t neg10k = -10000 + 0x10000;
  using int32x4 = std::conditional_t<ZMIJ_MSC_VER != 0, int32_t[4], int32x4_t>;
  using int16x8 = std::conditional_t<ZMIJ_MSC_VER != 0, int16_t[8], int16x8_t>;
  struct mul_constants {
    uint64_t mul_const = 0xabcc77118461cefd;
    uint64_t hundred_million = 100000000;
    int32x4 multipliers32 = {div10k_sig, neg10k, div100_sig << 12, neg100};
    int16x8 multipliers16 = {0xce0, neg10};
  };
  static const mul_constants constants;
  const mul_constants* c = &constants;

  // Compiler barrier, or clang doesn't load from memory and generates 15 more
  // instructions
  ZMIJ_ASM(("" : "+r"(c)));

  uint64_t hundred_million = c->hundred_million;

  // Compiler barrier, or clang narrows the load to 32-bit and unpairs it.
  ZMIJ_ASM(("" : "+r"(hundred_million)));

  // Equivalent to abbccddee = value / 100000000, ffgghhii = value % 100000000.
  uint64_t abbccddee = uint64_t(umul128(value, c->mul_const) >> 90);
  uint64_t ffgghhii = value - abbccddee * hundred_million;

  // We could probably make this bit faster, but we're preferring to
  // reuse the constants for now.
  uint64_t a = uint64_t(umul128(abbccddee, c->mul_const) >> 90);
  uint64_t bbccddee = abbccddee - a * hundred_million;

  char* start = buffer + 1;
  buffer = write_if(start, a, has17digits);

  uint64x1_t ffgghhii_bbccddee_64 = {(uint64_t(ffgghhii) << 32) | bbccddee};
  int32x2_t bbccddee_ffgghhii = vreinterpret_s32_u64(ffgghhii_bbccddee_64);

  int32x2_t bbcc_ffgg = vreinterpret_s32_u32(
      vshr_n_u32(vreinterpret_u32_s32(
                     vqdmulh_n_s32(bbccddee_ffgghhii, c->multipliers32[0])),
                 9));
  int32x2_t ddee_bbcc_hhii_ffgg_32 =
      vmla_n_s32(bbccddee_ffgghhii, bbcc_ffgg, c->multipliers32[1]);

  int32x4_t ddee_bbcc_hhii_ffgg = vreinterpretq_s32_u32(
      vshll_n_u16(vreinterpret_u16_s32(ddee_bbcc_hhii_ffgg_32), 0));

  // Compiler barrier, or clang breaks the subsequent MLA into UADDW + MUL.
  ZMIJ_ASM(("" : "+w"(ddee_bbcc_hhii_ffgg)));

  int32x4_t dd_bb_hh_ff =
      vqdmulhq_n_s32(ddee_bbcc_hhii_ffgg, c->multipliers32[2]);
  int16x8_t ee_dd_cc_bb_ii_hh_gg_ff = vreinterpretq_s16_s32(
      vmlaq_n_s32(ddee_bbcc_hhii_ffgg, dd_bb_hh_ff, c->multipliers32[3]));
  int16x8_t high_10s =
      vqdmulhq_n_s16(ee_dd_cc_bb_ii_hh_gg_ff, c->multipliers16[0]);
  uint8x16_t digits = vrev64q_u8(vreinterpretq_u8_s16(
      vmlaq_n_s16(ee_dd_cc_bb_ii_hh_gg_ff, high_10s, c->multipliers16[1])));
  uint16x8_t str = vaddq_u16(vreinterpretq_u16_u8(digits),
                             vreinterpretq_u16_s8(vdupq_n_s8('0')));
  memcpy(buffer, &str, sizeof(str));

  uint16x8_t is_not_zero = vreinterpretq_u16_u8(vcgtzq_s8(digits));
  uint64_t zeroes =
      vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(is_not_zero, 4)), 0);

  buffer += 16 - ((zeroes != 0 ? clz(zeroes) : 64) >> 2);
  return buffer - int(buffer - start == 1);
#elif ZMIJ_USE_SSE
  uint64_t digits_16 =
      use_umul128_hi64 ? umul128_hi64(value, div10_sig64) : value / 10;
  uint32_t last_digit = value - digits_16 * 10;

  // We always write 17 digits into the buffer, but the first one can be zero.
  // buffer points to the second place in the output buffer to allow for the
  // insertion of the decimal point, so we can use the first place as scratch.
  buffer += has17digits;
  buffer[16] = char(last_digit + '0');

  uint32_t abcdefgh = digits_16 / uint64_t(1e8);
  uint32_t ijklmnop = digits_16 % uint64_t(1e8);

  alignas(64) static const struct {
    __m128i div10k = _mm_set1_epi64x(div10k_sig);
    __m128i neg10k = _mm_set1_epi64x(::neg10k);
    __m128i div100 = _mm_set1_epi32(div100_sig);
    __m128i div10 = _mm_set1_epi16((1 << 16) / 10 + 1);
#  if ZMIJ_USE_SSE4_1
    __m128i neg100 = _mm_set1_epi32(::neg100);
    __m128i neg10 = _mm_set1_epi16((1 << 8) - 10);
    __m128i bswap =
        _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#  else
    __m128i hundred = _mm_set1_epi32(100);
    __m128i moddiv10 = _mm_set1_epi16(10 * (1 << 8) - 1);
#  endif  // ZMIJ_USE_SSE4_1
    __m128i zeros = _mm_set1_epi64x(::zeros);
  } c;

  // The BCD sequences are based on ones provided by Xiang JunBo.
  __m128i x = _mm_set_epi64x(abcdefgh, ijklmnop);
  __m128i y = _mm_add_epi64(
      x, _mm_mul_epu32(c.neg10k,
                       _mm_srli_epi64(_mm_mul_epu32(x, c.div10k), div10k_exp)));
#  if ZMIJ_USE_SSE4_1
  // _mm_mullo_epi32 is SSE 4.1
  __m128i z = _mm_add_epi64(
      y, _mm_mullo_epi32(c.neg100,
                         _mm_srli_epi32(_mm_mulhi_epu16(y, c.div100), 3)));
  __m128i big_endian_bcd =
      _mm_add_epi16(z, _mm_mullo_epi16(c.neg10, _mm_mulhi_epu16(z, c.div10)));
  __m128i bcd = _mm_shuffle_epi8(big_endian_bcd, c.bswap);  // SSSE3
#  else
  __m128i y_div_100 = _mm_srli_epi16(_mm_mulhi_epu16(y, c.div100), 3);
  __m128i y_mod_100 = _mm_sub_epi16(y, _mm_mullo_epi16(y_div_100, c.hundred));
  __m128i z = _mm_or_si128(_mm_slli_epi32(y_mod_100, 16), y_div_100);
  __m128i bcd_shuffled =
      _mm_sub_epi16(_mm_slli_epi16(z, 8),
                    _mm_mullo_epi16(c.moddiv10, _mm_mulhi_epu16(z, c.div10)));
  __m128i bcd = _mm_shuffle_epi32(bcd_shuffled, _MM_SHUFFLE(0, 1, 2, 3));
#  endif  // ZMIJ_USE_SSE4_1

  auto digits = _mm_or_si128(bcd, c.zeros);

  // determine number of leading zeros
  __m128i mask128 = _mm_cmpgt_epi8(bcd, _mm_setzero_si128());
  uint16_t mask = _mm_movemask_epi8(mask128);
  // We don't need a zero-check here: if the mask were zero, either the
  // significand is zero which is handled elsewhere or the only non-zero digit
  // is the last digit which we factored off. But in that case the number would
  // be printed with a different exponent that shifts the last digit into the
  // first position.
  auto len = size_t(64) - clz(mask);  // size_t for native arithmetic

  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), digits);
  return buffer + ((last_digit != 0) ? 17 : len - (len == 1));
#else     // !ZMIJ_USE_NEON && !ZMIJ_USE_SSE
  char* start = buffer + 1;
  // Each digit is denoted by a letter so value is abbccddeeffgghhii.
  uint32_t abbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);
  buffer = write_if(start, abbccddee / 100'000'000, has17digits);
  uint64_t bcd = to_bcd8(abbccddee % 100'000'000);
  write8(buffer, bcd | zeros);
  if (ffgghhii == 0) {
    buffer += count_trailing_nonzeros(bcd);
    return buffer - int(buffer - start == 1);
  }
  bcd = to_bcd8(ffgghhii);
  write8(buffer + 8, bcd | zeros);
  return buffer + 8 + count_trailing_nonzeros(bcd);
#endif
}

// Writes a significand consisting of up to 9 decimal digits (7-9 for normals)
// and removes trailing zeros.
auto write_significand9(char* buffer, uint32_t value, bool has9digits) noexcept
    -> char* {
  char* start = buffer;
  buffer = write_if(buffer, value / 100'000'000, has9digits);
  uint64_t bcd = to_bcd8(value % 100'000'000);
  write8(buffer, bcd | zeros);
  buffer += count_trailing_nonzeros(bcd);
  return buffer - int(buffer - start == 1);
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

template <bool subnormal = false, typename UInt>
auto to_decimal_schubfach(UInt bin_sig, int64_t bin_exp, bool regular) noexcept
    -> zmij::dec_fp {
  constexpr int num_bits = std::numeric_limits<UInt>::digits;
  int dec_exp = compute_dec_exp(bin_exp, regular);
  unsigned char exp_shift = compute_exp_shift<num_bits>(bin_exp, dec_exp);
  uint128 pow10 = pow10_significands[-dec_exp];

  // Fallback to Schubfach to guarantee correctness in boundary cases.
  // This requires switching to strict overestimates of powers of 10.
  ++(num_bits == 64 ? pow10.lo : pow10.hi);

  // Shift the significand so that boundaries are integer.
  constexpr int bound_shift = 2;
  UInt bin_sig_shifted = bin_sig << bound_shift;

  // Compute the estimates of lower and upper bounds of the rounding interval
  // by multiplying them by the power of 10 and applying modified rounding.
  UInt lsb = bin_sig & 1;
  UInt lower = (bin_sig_shifted - (regular + 1)) << exp_shift;
  lower = umulhi_inexact_to_odd(pow10.hi, pow10.lo, lower) + lsb;
  UInt upper = (bin_sig_shifted + 2) << exp_shift;
  upper = umulhi_inexact_to_odd(pow10.hi, pow10.lo, upper) - lsb;

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  UInt shorter = 10 * ((upper >> bound_shift) / 10);
  if ((shorter << bound_shift) >= lower)
    return normalize<num_bits>({int64_t(shorter), dec_exp}, subnormal);

  UInt scaled_sig =
      umulhi_inexact_to_odd(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
  UInt longer_below = scaled_sig >> bound_shift;
  UInt longer_above = longer_below + 1;

  // Pick the closest of dec_sig_below and dec_sig_above and check if it's in
  // the rounding interval.
  using sint = std::make_signed_t<UInt>;
  sint cmp = sint(scaled_sig - ((longer_below + longer_above) << 1));
  bool below_closer = cmp < 0 || (cmp == 0 && (longer_below & 1) == 0);
  bool below_in = (longer_below << bound_shift) >= lower;
  UInt dec_sig = (below_closer & below_in) ? longer_below : longer_above;
  return normalize<num_bits>({int64_t(dec_sig), dec_exp}, subnormal);
}

// Here be 🐉s.
// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - num_sig_bits - exp_bias.
template <typename Float, typename UInt>
ZMIJ_INLINE auto to_decimal_normal(UInt bin_sig, int64_t raw_exp,
                                   bool regular) noexcept -> zmij::dec_fp {
  using traits = float_traits<Float>;
  int64_t bin_exp = raw_exp - traits::num_sig_bits - traits::exp_bias;
  constexpr int num_bits = std::numeric_limits<UInt>::digits;
  // An optimization from yy by Yaoyuan Guo:
  while (regular) [[ZMIJ_LIKELY]] {
    int dec_exp = use_umul128_hi64 ? umul128_hi64(bin_exp, 0x4d10500000000000)
                                   : compute_dec_exp(bin_exp, true);
    unsigned char exp_shift =
        compute_exp_shift<num_bits, true>(bin_exp, dec_exp);
    uint128 pow10 = pow10_significands[-dec_exp];

    UInt integral = 0;        // integral part of bin_sig * pow10
    uint64_t fractional = 0;  // fractional part of bin_sig * pow10
    if (num_bits == 64) {
      uint128 p = umul192_hi128(pow10.hi, pow10.lo, bin_sig << exp_shift);
      integral = p.hi;
      fractional = p.lo;
    } else {
      uint128_t p = umul128(pow10.hi, bin_sig << exp_shift);
      integral = uint64_t(p >> 64);
      fractional = uint64_t(p);
    }
    constexpr uint64_t half_ulp = uint64_t(1) << 63;

    // Exact half-ulp tie when rounding to nearest integer.
    int64_t cmp = int64_t(fractional - half_ulp);
    if (cmp == 0) [[ZMIJ_UNLIKELY]]
      break;

    uint64_t digit;
    if (ZMIJ_USE_INT128) {
      // An optimization of integral % 10 by Dougall Johnson.
      // Relies on range calculation: (max_bin_sig << max_exp_shift) * max_u128.
      digit = integral - umul128_hi64(integral, div10_sig64) * 10;
      // or it narrows to 32-bit and doesn't use madd/msub
      ZMIJ_ASM(("" : "+r"(digit)));
    } else {
      digit = integral % 10;
    }

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
    uint64_t scaled_half_ulp = pow10.hi >> (num_integral_bits - exp_shift + 1);
    uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;

    // value = 5.0507837461e-27
    // next  = 5.0507837461000010e-27
    //
    // c = integral.fractional' = 50507837461000003.153987... (value)
    //                            50507837461000010.328635... (next)
    //          scaled_half_ulp =                 3.587324...
    //
    // fractional' = fractional / 2**64, fractional = 2840565642863009226
    //
    //      50507837461000000       c               upper     50507837461000010
    //              s              l|   L             |               S
    // ───┬────┬────┼────┬────┬────┼*-──┼────┬────┬───*┬────┬────┬────┼-*--┬───
    //    8    9    0    1    2    3    4    5    6    7    8    9    0 |  1
    //            └─────────────────┼─────────────────┘                next
    //                             1ulp
    //
    // s - shorter underestimate, S - shorter overestimate
    // l - longer underestimate,  L - longer overestimate

    // Check for boundary case when rounding down to nearest 10 and
    // near-boundary case when rounding up to nearest 10.
    if (scaled_sig_mod10 == scaled_half_ulp ||
        // Case where upper == ten is insufficient: 1.342178e+08f.
        ten - upper <= 1u)  // upper == ten || upper == ten - 1
        [[ZMIJ_UNLIKELY]] {
      break;
    }

    bool round_up = upper >= ten;
    int64_t shorter = int64_t(integral - digit);
    int64_t longer = int64_t(integral + (cmp >= 0));
    if (ZMIJ_AARCH64) {  // Faster version without ccmp.
      int64_t dec_sig = scaled_sig_mod10 < scaled_half_ulp ? shorter : longer;
      return {round_up ? shorter + 10 : dec_sig, dec_exp};
    }
    shorter += round_up * 10;
    bool use_shorter = (scaled_sig_mod10 <= scaled_half_ulp) + round_up != 0;
    return {use_shorter ? shorter : longer, dec_exp};
  }
  return to_decimal_schubfach(bin_sig, bin_exp, regular);
}

}  // namespace

namespace zmij {

inline auto to_decimal(double value) noexcept -> dec_fp {
  using traits = float_traits<double>;
  auto bits = traits::to_bits(value);
  auto bin_exp = traits::get_exp(bits);  // binary exponent
  auto bin_sig = traits::get_sig(bits);  // binary significand
  dec_fp dec;
  if (bin_exp == 0 || bin_exp == traits::exp_mask) [[ZMIJ_UNLIKELY]] {
    if (bin_exp != 0) return {0, int(~0u >> 1)};
    if (bin_sig == 0) return {0, 0};
    constexpr int exp_offset = traits::num_sig_bits + traits::exp_bias;
    dec = to_decimal_schubfach<true>(bin_sig, 1 - exp_offset, true);
  } else {
    dec = to_decimal_normal<double>(bin_sig | traits::implicit_bit, bin_exp,
                                    bin_sig != 0);
  }
  return {traits::is_negative(bits) ? -dec.sig : dec.sig, dec.exp};
}

namespace detail {

// It is slightly faster to return a pointer to the end than the size.
template <typename Float>
auto write(Float value, char* buffer) noexcept -> char* {
  using traits = float_traits<Float>;
  auto bits = traits::to_bits(value);
  // It is beneficial to extract exponent and significand early.
  auto bin_exp = traits::get_exp(bits);  // binary exponent
  auto bin_sig = traits::get_sig(bits);  // binary significand

  *buffer = '-';
  buffer += traits::is_negative(bits);

  dec_fp dec;
  if (bin_exp == 0 || bin_exp == traits::exp_mask) [[ZMIJ_UNLIKELY]] {
    if (bin_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return buffer + 3;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return buffer + 1;
    }
    constexpr int exp_offset = traits::num_sig_bits + traits::exp_bias;
    dec = to_decimal_schubfach<true>(bin_sig, 1 - exp_offset, true);
  } else {
    dec = to_decimal_normal<Float>(bin_sig | traits::implicit_bit, bin_exp,
                                   bin_sig != 0);
  }
  int dec_exp = dec.exp;

  // Write significand.
  char* start = buffer;
  if (traits::num_bits == 64) {
    bool has17digits = dec.sig >= uint64_t(1e16);
    dec_exp += traits::max_digits10 - 2 + has17digits;
    buffer = write_significand17(buffer, dec.sig, has17digits);
  } else {
    if (dec.sig < uint32_t(1e7)) [[ZMIJ_UNLIKELY]] {
      dec.sig *= 10;
      --dec_exp;
    }
    bool has9digits = dec.sig >= uint32_t(1e8);
    dec_exp += traits::max_digits10 - 2 + has9digits;
    buffer = write_significand9(buffer + 1, dec.sig, has9digits);
  }
  start[0] = start[1];
  start[1] = '.';

  // Write exponent.
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian()) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  dec_exp = dec_exp >= 0 ? dec_exp : -dec_exp;
  if (traits::min_exponent10 >= -99 && traits::max_exponent10 <= 99) {
    memcpy(buffer, digits2(dec_exp), 2);
    buffer[2] = '\0';
    return buffer + 2;
  }

  // digit = dec_exp / 100
  uint32_t digit = use_umul128_hi64
                       ? umul128_hi64(dec_exp, 0x290000000000000)
                       : (uint32_t(dec_exp) * div100_sig) >> div100_exp;
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
