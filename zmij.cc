// A double-to-string conversion algorithm based on Schubfach and xjb.
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
  bool negative;
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
// Use the provided definition.
#elif defined(__ARM_NEON) || defined(_M_ARM64)
#  define ZMIJ_USE_NEON ZMIJ_USE_SIMD
#else
#  define ZMIJ_USE_NEON 0
#endif
#if ZMIJ_USE_NEON
#  include <arm_neon.h>
#endif

#ifdef ZMIJ_USE_SSE
// Use the provided definition.
#elif defined(__SSE2__)
#  define ZMIJ_USE_SSE ZMIJ_USE_SIMD
#elif defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86_FP == 2)
#  define ZMIJ_USE_SSE ZMIJ_USE_SIMD
#else
#  define ZMIJ_USE_SSE 0
#endif
#if ZMIJ_USE_SSE
#  include <immintrin.h>
#endif

#ifdef ZMIJ_USE_SSE4_1
// Use the provided definition.
static_assert(!ZMIJ_USE_SSE4_1 || ZMIJ_USE_SSE);
#elif defined(__SSE4_1__) || defined(__AVX__)
// On MSVC there's no way to check for SSE4.1 specifically so check __AVX__.
#  define ZMIJ_USE_SSE4_1 ZMIJ_USE_SSE
#else
#  define ZMIJ_USE_SSE4_1 0
#endif

#define ZMIJ_USE_SIMD_SHUFFLE \
  ((ZMIJ_USE_NEON || ZMIJ_USE_SSE4_1) && !ZMIJ_OPTIMIZE_SIZE)

#ifdef __aarch64__
#  define ZMIJ_AARCH64 1
#else
#  define ZMIJ_AARCH64 0
#endif

#ifdef __x86_64__
#  define ZMIJ_X86_64 1
#else
#  define ZMIJ_X86_64 0
#endif

#ifdef __clang__
#  define ZMIJ_CLANG 1
#else
#  define ZMIJ_CLANG 0
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

#ifdef ZMIJ_OPTIMIZE_SIZE
// Use the provided definition.
#elif defined(__OPTIMIZE_SIZE__)
#  define ZMIJ_OPTIMIZE_SIZE 1
#else
#  define ZMIJ_OPTIMIZE_SIZE 0
#endif
#ifndef ZMIJ_USE_EXP_STRING_TABLE
#  define ZMIJ_USE_EXP_STRING_TABLE ZMIJ_OPTIMIZE_SIZE == 0
#endif

#if ZMIJ_HAS_ATTRIBUTE(always_inline) && !ZMIJ_OPTIMIZE_SIZE
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
inline auto is_constant_evaluated() -> bool { return false; }
#  define ZMIJ_CONSTEXPR
#endif

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr bool is_big_endian = true;
#else
constexpr bool is_big_endian = false;
#endif

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

inline auto ctz(uint64_t x) noexcept -> int {
  assert(x != 0);
#if ZMIJ_HAS_BUILTIN(__builtin_ctzll)
  return __builtin_ctzll(x);
#elif defined(_M_AMD64) || defined(_M_ARM64)
  unsigned long idx;
  _BitScanForward64(&idx, x);
  return idx;
#elif ZMIJ_MSC_VER
  unsigned long idx;
  if (_BitScanForward(&idx, uint32_t(x))) return idx;
  _BitScanForward(&idx, uint32_t(x >> 32));
  return idx + 32;
#else
  int n = 0;
  for (; (x & 1) == 0; x >>= 1) ++n;
  return n;
#endif
}

// Returns true_value if lhs < rhs, else false_value, without branching.
ZMIJ_INLINE auto select_if_less(uint64_t lhs, uint64_t rhs, int64_t true_value,
                                int64_t false_value) -> int64_t {
  if (!ZMIJ_X86_64 || ZMIJ_CLANG) return lhs < rhs ? true_value : false_value;
  ZMIJ_ASM(
      volatile("cmp %3, %2\n\t"
               "cmovb %1, %0\n\t"  //
               : "+r"(false_value) : "r"(true_value),
               "r"(lhs), "r"(rhs) : "cc"));
  return false_value;
}

struct uint128 {
  uint64_t hi;
  uint64_t lo;

  [[ZMIJ_MAYBE_UNUSED]] explicit constexpr operator uint64_t() const noexcept {
    return lo;
  }

  [[ZMIJ_MAYBE_UNUSED]] constexpr auto operator>>(int shift) const noexcept
      -> uint128 {
    if (shift == 32) return {hi >> 32, (hi << 32) | (lo >> 32)};
    assert(shift >= 64 && shift < 128);
    return {0, hi >> (shift - 64)};
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

// Computes the decimal exponent as floor(log10(2**bin_exp)) if regular or
// floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
constexpr auto compute_dec_exp(int bin_exp, bool regular = true) noexcept
    -> int {
  assert(bin_exp >= -1334 && bin_exp <= 2620);
  // log10_3_over_4_sig = -log10(3/4) * 2**log10_2_exp rounded to a power of 2
  constexpr int log10_3_over_4_sig = 131'072;
  // log10_2_sig = round(log10(2) * 2**log10_2_exp)
  constexpr int log10_2_sig = 315'653, log10_2_exp = 20;
  return (bin_exp * log10_2_sig - !regular * log10_3_over_4_sig) >> log10_2_exp;
}

template <typename Float> struct float_traits : std::numeric_limits<Float> {
  static_assert(float_traits::is_iec559, "IEEE 754 required");

  static constexpr int num_bits = float_traits::digits == 53 ? 64 : 32;
  static constexpr int num_sig_bits = float_traits::digits - 1;
  static constexpr int num_exp_bits = num_bits - num_sig_bits - 1;
  static constexpr int exp_mask = (1 << num_exp_bits) - 1;
  static constexpr int exp_bias = (1 << (num_exp_bits - 1)) - 1;
  static constexpr int exp_offset = exp_bias + num_sig_bits;
  static constexpr int min_fixed_dec_exp = -4;
  static constexpr int max_fixed_dec_exp =
      compute_dec_exp(float_traits::digits + 1) - 1;

  using sig_type = std::conditional_t<num_bits == 64, uint64_t, uint32_t>;
  static constexpr sig_type implicit_bit = sig_type(1) << num_sig_bits;

  static auto to_bits(Float value) noexcept -> sig_type {
    sig_type bits;
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

constexpr uint64_t pow10_minor[] = {
    0x8000000000000000, 0xa000000000000000, 0xc800000000000000,
    0xfa00000000000000, 0x9c40000000000000, 0xc350000000000000,
    0xf424000000000000, 0x9896800000000000, 0xbebc200000000000,
    0xee6b280000000000, 0x9502f90000000000, 0xba43b74000000000,
    0xe8d4a51000000000, 0x9184e72a00000000, 0xb5e620f480000000,
    0xe35fa931a0000000, 0x8e1bc9bf04000000, 0xb1a2bc2ec5000000,
    0xde0b6b3a76400000, 0x8ac7230489e80000, 0xad78ebc5ac620000,
    0xd8d726b7177a8000, 0x878678326eac9000, 0xa968163f0a57b400,
    0xd3c21bcecceda100, 0x84595161401484a0, 0xa56fa5b99019a5c8,
    0xcecb8f27f4200f3a,
};
constexpr uint128 pow10_major[] = {
    {0xaf8e5410288e1b6f, 0x07ecf0ae5ee44dda},  // -303
    {0xb1442798f49ffb4a, 0x99cd11cfdf41779d},  // -275
    {0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa15},  // -247
    {0xb4bca50b065abe63, 0x0fed077a756b53aa},  // -219
    {0xb67f6455292cbf08, 0x1a3bc84c17b1d543},  // -191
    {0xb84687c269ef3bfb, 0x3d5d514f40eea742},  // -163
    {0xba121a4650e4ddeb, 0x92f34d62616ce413},  // -135
    {0xbbe226efb628afea, 0x890489f70a55368c},  // -107
    {0xbdb6b8e905cb600f, 0x5400e987bbc1c921},  //  -79
    {0xbf8fdb78849a5f96, 0xde98520472bdd034},  //  -51
    {0xc16d9a0095928a27, 0x75b7053c0f178294},  //  -23
    {0xc350000000000000, 0x0000000000000000},  //    5
    {0xc5371912364ce305, 0x6c28000000000000},  //   33
    {0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef6},  //   61
    {0xc913936dd571c84c, 0x03bc3a19cd1e38ea},  //   89
    {0xcb090c8001ab551c, 0x5cadf5bfd3072cc6},  //  117
    {0xcd036837130890a1, 0x36dba887c37a8c10},  //  145
    {0xcf02b2c21207ef2e, 0x94f967e45e03f4bc},  //  173
    {0xd106f86e69d785c7, 0xe13336d701beba52},  //  201
    {0xd31045a8341ca07c, 0x1ede48111209a051},  //  229
    {0xd51ea6fa85785631, 0x552a74227f3ea566},  //  257
    {0xd732290fbacaf133, 0xa97c177947ad4096},  //  285
    {0xd94ad8b1c7380874, 0x18375281ae7822bc},  //  313
};
constexpr uint32_t pow10_fixups[] = {
    0x0a4e363f, 0x00001840, 0x00006400, 0x24200040, 0x00000000,
    0x0c000000, 0x82c81380, 0x5e4ce01f, 0xd730f60f, 0x0000001b,
    0x00000000, 0xcdf7fffc, 0x6e8201d8, 0x40cd3fd1, 0xdb642501,
    0x00000d0d, 0x14042400, 0x53713840, 0x11781db4, 0x00000000};

// 128-bit significands of powers of 10 rounded down.
struct pow10_significand_table {
  static constexpr bool compress = ZMIJ_OPTIMIZE_SIZE != 0;
  static constexpr bool split_tables = !compress && ZMIJ_AARCH64 != 0;
  static constexpr int num_pow10s = 618;
  uint64_t data[compress ? 1 : num_pow10s * 2] = {};

  // Computes the 128-bit significand of 10**i using method by Dougall Johnson.
  static constexpr auto compute(unsigned i) noexcept -> uint128 {
    constexpr int stride = sizeof(pow10_minor) / sizeof(*pow10_minor);
    auto m = pow10_minor[(i + 10) % stride];
    auto h = pow10_major[(i + 10) / stride];

    uint64_t h1 = umul128_hi64(h.lo, m);

    uint64_t c0 = h.lo * m;
    uint64_t c1 = h1 + h.hi * m;
    uint64_t c2 = (c1 < h1) + umul128_hi64(h.hi, m);

    uint128 result = (c2 >> 63) != 0
                         ? uint128{c2, c1}
                         : uint128{c2 << 1 | c1 >> 63, c1 << 1 | c0 >> 63};
    result.lo -= (pow10_fixups[i >> 5] >> (i & 31)) & 1;
    return result;
  }

  constexpr pow10_significand_table() {
    for (int i = 0; i < num_pow10s && !compress; ++i) {
      uint128 result = compute(i);
      if (split_tables) {
        data[num_pow10s - i - 1] = result.hi;
        data[num_pow10s * 2 - i - 1] = result.lo;
      } else {
        data[i * 2] = result.hi;
        data[i * 2 + 1] = result.lo;
      }
    }
  }

  ZMIJ_CONSTEXPR auto operator[](int dec_exp) const noexcept -> uint128 {
    constexpr int dec_exp_min = -293;
    int i = dec_exp - dec_exp_min;
    if (compress) return compute(i);
    if (!split_tables) return {data[i * 2], data[i * 2 + 1]};

    const uint64_t* hi = data + num_pow10s + dec_exp_min - 1;
    const uint64_t* lo = hi + num_pow10s;

    // Force indexed loads.
    if (!is_constant_evaluated()) ZMIJ_ASM(volatile("" : "+r"(hi), "+r"(lo)));
    return {hi[-dec_exp], lo[-dec_exp]};
  }
};
alignas(64) constexpr pow10_significand_table pow10_significands;

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
  static constexpr bool enable = ZMIJ_OPTIMIZE_SIZE == 0;
  // extra_shift must be >= 3 to keep shift non-negative and <= 11 to
  // fit the significand into 64 bits after the shift.
  static constexpr int extra_shift = 6;
  unsigned char data[enable ? float_traits<double>::exp_mask + 1 : 1] = {};

  constexpr exp_shift_table() {
    for (int raw_exp = 0; raw_exp < sizeof(data) && enable; ++raw_exp) {
      int bin_exp = raw_exp - float_traits<double>::exp_offset;
      if (raw_exp == 0) ++bin_exp;
      int dec_exp = compute_dec_exp(bin_exp);
      data[raw_exp] = compute_exp_shift(bin_exp, dec_exp + 1) + extra_shift;
    }
  }
};
constexpr exp_shift_table exp_shifts;

// An optional table of precomputed exponent strings for scientific notation.
// Each entry packs "e+dd" or "e+ddd" into a uint64_t with the length in byte 7.
struct exp_string_table {
  static constexpr bool enable = ZMIJ_USE_EXP_STRING_TABLE;
  using traits = float_traits<double>;
  static constexpr int min_dec_exp =
      traits::min_exponent10 - traits::max_digits10;
  static constexpr int offset = -min_dec_exp;
  uint64_t data[enable ? traits::max_exponent10 - min_dec_exp + 1 : 1] = {};

  constexpr exp_string_table() {
    for (int e = min_dec_exp; e <= traits::max_exponent10 && enable; ++e) {
      uint64_t abs_e = e >= 0 ? e : -e;
      uint64_t bc = abs_e % 100;
      uint64_t val = ((bc % 10 + '0') << 8) | (bc / 10 + '0');
      if (uint64_t a = abs_e / 100) val = (val << 8) | (a + '0');
      uint64_t len =
          e >= -4 && e < traits::max_fixed_dec_exp ? 0 : 4 + (abs_e >= 100);
      data[e + offset] =
          (len << 48) | (val << 16) | (uint64_t(e >= 0 ? '+' : '-') << 8) | 'e';
    }
  }
};
constexpr exp_string_table exp_strings;

// Per-decimal-exponent formatting positions for branchless output.
// Each entry holds positions for the decimal point, leading zeros, and the
// exponent, indexed by the decimal exponent (dec_exp).
struct dec_exp_format_table {
  using traits = float_traits<double>;
  static constexpr int num_entries =
      traits::max_fixed_dec_exp - traits::min_fixed_dec_exp + 2;  // +1 sentinel

  struct entry {
    // Byte offset past leading "0.00..." before first significant digit.
    unsigned char start_pos;
    unsigned char point_pos;
    // Start position for shifting digits right by one to insert the point.
    unsigned char shift_pos;
    // Position where exponent notation starts, indexed by sig length - 1.
    // For fixed-notation entries this points past all output digits.
    unsigned char exp_pos[traits::max_digits10];
  };

  entry data[num_entries] = {};

  constexpr dec_exp_format_table() {
    for (int dec_exp = traits::min_fixed_dec_exp;
         dec_exp <= traits::max_fixed_dec_exp + 1; ++dec_exp) {
      auto& e = data[dec_exp - traits::min_fixed_dec_exp];
      bool neg_fixed = dec_exp >= traits::min_fixed_dec_exp && dec_exp <= -1;
      bool pos_fixed = dec_exp >= 0 && dec_exp <= traits::max_fixed_dec_exp;

      e.start_pos = neg_fixed ? 1 - dec_exp : 0;
      e.point_pos = pos_fixed ? 1 + dec_exp : 1;
      e.shift_pos =
          e.point_pos + (dec_exp >= 0 || dec_exp < traits::min_fixed_dec_exp);

      for (int s = 1; s <= traits::max_digits10; ++s) {
        if (neg_fixed)
          e.exp_pos[s - 1] = s;
        else if (pos_fixed)
          e.exp_pos[s - 1] = s > dec_exp + 1 ? s + 1 : dec_exp + 1;
        else
          e.exp_pos[s - 1] = s + 1 - (s == 1);
      }
    }
  }

  template <typename Traits>
  constexpr auto get(int dec_exp) const noexcept -> const entry& {
    constexpr auto min = traits::min_fixed_dec_exp,
                   max = Traits::max_fixed_dec_exp;
    unsigned i = unsigned(dec_exp - min);
    return data[i <= unsigned(max - min) ? i : num_entries - 1];
  }
};
constexpr dec_exp_format_table dec_exp_formats;

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
  if (is_big_endian) x = bswap64(x);
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
  return is_big_endian ? a_b_c_d_e_f_g_h : bswap64(a_b_c_d_e_f_g_h);
}

inline auto write_if(char* buffer, uint32_t digit, bool condition) noexcept
    -> char* {
  *buffer = char('0' + digit);
  return buffer + condition;
}

#if ZMIJ_USE_SSE
alignas(64) constexpr struct sse_constants {
  static constexpr auto splat64(uint64_t x) -> uint128 { return {x, x}; }
  static constexpr auto splat32(uint32_t x) -> uint128 {
    return splat64(uint64_t(x) << 32 | x);
  }
  static constexpr auto splat16(uint16_t x) -> uint128 {
    return splat32(uint32_t(x) << 16 | x);
  }
  static constexpr auto pack8(uint8_t a, uint8_t b, uint8_t c, uint8_t d,  //
                              uint8_t e, uint8_t f, uint8_t g, uint8_t h)
      -> uint64_t {
    using u64 = uint64_t;
    return u64(h) << 56 | u64(g) << 48 | u64(f) << 40 | u64(e) << 32 |
           u64(d) << 24 | u64(c) << 16 | u64(b) << +8 | u64(a);
  }

  uint128 div10k = splat64(div10k_sig);
  uint128 neg10k = splat64(::neg10k);
  uint128 div100 = splat32(div100_sig);
  uint128 div10 = splat16((1 << 16) / 10 + 1);
#  if ZMIJ_USE_SSE4_1
  uint128 neg100 = splat32(::neg100);
  uint128 neg10 = splat16((1 << 8) - 10);
  uint128 bswap = uint128{pack8(15, 14, 13, 12, 11, 10, 9, 8),
                          pack8(7, 6, 5, 4, 3, 2, 1, 0)};
#  else
  uint128 hundred = splat32(100);
  uint128 moddiv10 = splat16(10 * (1 << 8) - 1);
#  endif  // ZMIJ_USE_SSE4_1
  uint128 zeros = splat64(::zeros);
} sse_consts;

using m128ptr = const __m128i*;

// SSE parallel version of to_bcd8: converts bbccddee and ffgghhii into
// individual BCD digits in SIMD lane order (caller must shuffle).
ZMIJ_INLINE auto to_unshuffled_digits(uint32_t bbccddee, uint32_t ffgghhii,
                                      const sse_constants& c) noexcept
    -> __m128i {
  const __m128i div10k = _mm_load_si128(m128ptr(&c.div10k));
  const __m128i neg10k = _mm_load_si128(m128ptr(&c.neg10k));
  const __m128i div100 = _mm_load_si128(m128ptr(&c.div100));
  const __m128i div10 = _mm_load_si128(m128ptr(&c.div10));
#  if ZMIJ_USE_SSE4_1
  const __m128i neg100 = _mm_load_si128(m128ptr(&c.neg100));
  const __m128i neg10 = _mm_load_si128(m128ptr(&c.neg10));
#  else
  const __m128i hundred = _mm_load_si128(m128ptr(&c.hundred));
  const __m128i moddiv10 = _mm_load_si128(m128ptr(&c.moddiv10));
#  endif

  __m128i x = _mm_set_epi64x(bbccddee, ffgghhii);
  __m128i y = _mm_add_epi64(
      x, _mm_mul_epu32(neg10k,
                       _mm_srli_epi64(_mm_mul_epu32(x, div10k), div10k_exp)));
#  if ZMIJ_USE_SSE4_1
  // _mm_mullo_epi32 is SSE 4.1
  __m128i z = _mm_add_epi64(
      y,
      _mm_mullo_epi32(neg100, _mm_srli_epi32(_mm_mulhi_epu16(y, div100), 3)));
  return _mm_add_epi16(z, _mm_mullo_epi16(neg10, _mm_mulhi_epu16(z, div10)));
#  else
  __m128i y_div_100 = _mm_srli_epi16(_mm_mulhi_epu16(y, div100), 3);
  __m128i y_mod_100 = _mm_sub_epi16(y, _mm_mullo_epi16(y_div_100, hundred));
  __m128i z = _mm_or_si128(_mm_slli_epi32(y_mod_100, 16), y_div_100);
  return _mm_sub_epi16(_mm_slli_epi16(z, 8),
                       _mm_mullo_epi16(moddiv10, _mm_mulhi_epu16(z, div10)));
#  endif  // ZMIJ_USE_SSE4_1
}
#endif  // ZMIJ_USE_SSE

#if ZMIJ_USE_NEON
// When reverse_hi_lo is true, the two 8-digit halves are returned in reversed
// order so that trailing zeros become leading zero bytes, letting ctz count
// them while the shuffle runs in parallel.
template <bool reverse_hi_lo = false>
ZMIJ_INLINE auto to_unshuffled_digits(uint64_t value) -> uint8x16_t {
  // An optimized version for NEON by Dougall Johnson.
  using int32x4 = std::conditional_t<ZMIJ_MSC_VER != 0, int32_t[4], int32x4_t>;
  using int16x8 = std::conditional_t<ZMIJ_MSC_VER != 0, int16_t[8], int16x8_t>;
  constexpr int32_t neg10k = -10000 + 0x10000;
  alignas(64) static constexpr struct {
    uint64_t mul_const = 0xabcc77118461cefd;
    uint64_t hundred_million = 100000000;
    int32x4 multipliers32 = {div10k_sig, neg10k, div100_sig << 12, neg100};
    int16x8 multipliers16 = {0xce0, neg10};
  } consts;
  const auto* c = &consts;

  // Compiler barrier, or clang doesn't load from memory and generates 15 more
  // instructions.
  ZMIJ_ASM(("" : "+r"(c)));

  uint64_t hundred_million = c->hundred_million;

  // Compiler barrier, or clang narrows the load to 32-bit and unpairs it.
  ZMIJ_ASM(("" : "+r"(hundred_million)));

  // Equivalent to abbccddee = value / 100000000, ffgghhii = value % 100000000.
  uint64_t abbccddee = uint64_t(umul128(value, c->mul_const) >> 90);
  uint64_t ffgghhii = value - abbccddee * hundred_million;

  // When reverse_hi_lo, actual order is abbccddee|ffgghhii; the 17th digit
  // ends up at byte index 0, so it can be extracted for the trailing write.
  uint64x1_t ffgghhii_bbccddee_64 = {
      reverse_hi_lo ? (uint64_t(abbccddee) << 32) | ffgghhii
                    : (uint64_t(ffgghhii) << 32) | abbccddee};
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
  return vreinterpretq_u8_s16(
      vmlaq_n_s16(ee_dd_cc_bb_ii_hh_gg_ff, high_10s, c->multipliers16[1]));
}
#endif

template <int num_bits> struct dec_digits {
  uint64_t digits;
  int num_digits;
};

template <> struct dec_digits<64> {
#if ZMIJ_USE_NEON
  uint16x8_t digits;
#elif ZMIJ_USE_SSE
  __m128i digits;
#else
  uint128 digits;
#endif
  int num_digits;
};

// Converts a significand to decimal digits, removing trailing zeros. value has
// up to 17 decimal digits (16-17 for normals) for double (num_bits == 64) and
// up to 9 digits (8-9 for normals) for float.
template <int num_bits>
ZMIJ_INLINE auto to_digits(char* buffer, uint64_t value,
                           bool extra_digit) noexcept -> dec_digits<num_bits> {
#if !ZMIJ_USE_SIMD
  // Digits/pairs of digits are denoted by letters: value = bbccddeeffgghhii.
  uint32_t bbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);
  uint64_t hi = to_bcd8(bbccddee);
  if (ffgghhii == 0) return {{hi + zeros, zeros}, count_trailing_nonzeros(hi)};
  uint64_t lo = to_bcd8(ffgghhii);
  return {{hi + zeros, lo + zeros}, 8 + count_trailing_nonzeros(lo)};
#elif ZMIJ_USE_NEON
  auto unshuffled_digits = to_unshuffled_digits(value);
  uint8x16_t digits = vrev64q_u8(unshuffled_digits);
  uint16x8_t str = vaddq_u16(vreinterpretq_u16_u8(digits),
                             vreinterpretq_u16_s8(vdupq_n_s8('0')));

  uint16x8_t is_not_zero =
      vreinterpretq_u16_u8(vcgtzq_s8(vreinterpretq_s8_u8(digits)));
  uint64_t zeroes =
      vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(is_not_zero, 4)), 0);
  return {str, 16 - (clz(zeroes) >> 2)};
#else  // ZMIJ_USE_SSE
  uint32_t abbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);

  const auto* c = &sse_consts;
  ZMIJ_ASM(("" : "+r"(c)));  // Load constants from memory.

  const __m128i zeros = _mm_load_si128(m128ptr(&c->zeros));
  auto unshuffled_bcd = to_unshuffled_digits(abbccddee, ffgghhii, *c);
#  if ZMIJ_USE_SSE4_1
  const __m128i bswap = _mm_load_si128(m128ptr(&c->bswap));
  auto bcd = _mm_shuffle_epi8(unshuffled_bcd, bswap);  // SSSE3
#  else
  auto bcd = _mm_shuffle_epi32(unshuffled_bcd, _MM_SHUFFLE(0, 1, 2, 3));
#  endif

  // Count leading zeros.
  __m128i mask128 = _mm_cmpgt_epi8(bcd, _mm_setzero_si128());
  uint64_t mask = _mm_movemask_epi8(mask128);
#  if defined(__LZCNT__) && !defined(ZMIJ_NO_BUILTINS)
  int len = 32 - _lzcnt_u32(mask);
#  else
  int len = 63 - clz((mask << 1) | 1);
#  endif
  return {_mm_or_si128(bcd, zeros), len};
#endif  // ZMIJ_USE_SSE
}

template <>
ZMIJ_INLINE auto to_digits<32>(char* buffer, uint64_t value,
                               bool extra_digit) noexcept -> dec_digits<32> {
  write_if(buffer, value / 100'000'000, extra_digit);
  uint64_t bcd = to_bcd8(value % 100'000'000);
  return {bcd + zeros, count_trailing_nonzeros(bcd)};
}

#if ZMIJ_USE_SIMD_SHUFFLE
struct shuffle_table {
  constexpr static bool merge_tables = ZMIJ_USE_SSE4_1;
  constexpr static size_t table_size =
      (1 + merge_tables) * (float_traits<double>::max_fixed_dec_exp + 2);
  alignas(16 * (1 + merge_tables)) uint8_t data[table_size][16] = {};
  constexpr shuffle_table() {
    for (int i = 0; i < float_traits<double>::max_fixed_dec_exp + 2; ++i) {
      uint8_t v = 0xf;
      for (int j = 0; j < 16; ++j) {
        data[(1 + merge_tables) * i][j] = i == j ? 0x80 : v--;
        if (merge_tables) data[2 * i + 1][j] = i == j ? '.' : '0';
      }
    }
  }

  ZMIJ_INLINE auto get_shuffler(int index) const noexcept -> const uint8_t* {
    assert((1 + merge_tables) * index < table_size);
    // GCC combines the address calculation with the one in get_point_and_zeros
    // below.
    return merge_tables ? data[0] + 32 * index : &data[index][0];
  }

  ZMIJ_INLINE auto get_point_and_zeros(int index) const noexcept
      -> const uint8_t* {
    assert(merge_tables && 2 * index + 1 < table_size);
    // GCC combines the address calculation with the one in get_shuffler above.
    return data[0] + 32 * index + 16;
  }
};
constexpr shuffle_table shuffles;
#endif  // ZMIJ_USE_SIMD_SHUFFLE

auto write_fixed_double_simd(char* buffer, uint64_t dec_sig, int dec_exp,
                             bool extra_digit) noexcept -> char* {
  int point_index = dec_exp + !extra_digit, len = 0;

#if ZMIJ_USE_NEON && !ZMIJ_OPTIMIZE_SIZE
  uint64_t a = dec_sig / 10000000000000000ull;
  uint64_t remaining = dec_sig - a * 10000000000000000ull;
  buffer = write_if(buffer, uint32_t(a), extra_digit);
  auto unshuffled_digits = to_unshuffled_digits<true>(remaining);
  auto unshuffled_str = vaddq_u16(vreinterpretq_u16_u8(unshuffled_digits),
                                  vreinterpretq_u16_s8(vdupq_n_s8('0')));
  auto str = vqtbl1q_u8(vreinterpretq_u8_u16(unshuffled_str),
                        vld1q_u8(shuffles.get_shuffler(point_index)));

  // Count trailing zeros.
  uint8x16_t is_not_zero = vcgtzq_s8(vreinterpretq_s8_u8(unshuffled_digits));
  uint64_t zeroes = vget_lane_u64(
      vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(is_not_zero), 4)),
      0);
  len = 16 - ((zeroes ? ctz(zeroes) : 64) >> 2);

  // Write 20 bytes non-overlappingly.
  uint32_t trailing_digit = vreinterpretq_u32_u16(unshuffled_str)[0];
  memcpy(buffer, &str, sizeof(str));
  memcpy(buffer + 16, &trailing_digit, 4);  // only need the lowest byte

  char* point = buffer + point_index;
  *point = '.';
#elif ZMIJ_USE_SSE4_1 && !ZMIJ_OPTIMIZE_SIZE
  uint32_t abbccddee = uint32_t(dec_sig / 100'000'000);
  uint32_t ffgghhii = uint32_t(dec_sig % 100'000'000);
  uint32_t a = abbccddee / 100'000'000;
  uint32_t bbccddee = abbccddee % 100'000'000;

  buffer = write_if(buffer, a, extra_digit);

  const auto* c = &sse_consts;
  ZMIJ_ASM(("" : "+r"(c)));  // Load constants from memory.
  __m128i zeros = _mm_load_si128(m128ptr(&c->zeros));

  auto reversed_bcd = to_unshuffled_digits(bbccddee, ffgghhii, *c);

  // Count trailing zeros.
  __m128i mask128 = _mm_cmpgt_epi8(reversed_bcd, _mm_setzero_si128());
  uint32_t mask = _mm_movemask_epi8(mask128) | (1u << 16);
#  if defined(__BMI1__) && !defined(ZMIJ_NO_BUILTINS)
  len = 16 - _tzcnt_u32(mask);
#  else
  len = 16 - ctz(mask);
#  endif

  __m128i shuffler =
      _mm_load_si128(m128ptr(shuffles.get_shuffler(point_index)));
  __m128i bcd = _mm_shuffle_epi8(reversed_bcd, shuffler);  // SSSE3
  __m128i point_mask =
      _mm_load_si128(m128ptr(shuffles.get_point_and_zeros(point_index)));
  __m128i digits_with_point = _mm_or_si128(bcd, point_mask);

  // Write 20 bytes non-overlappingly.
  uint32_t trailing_digit = _mm_cvtsi128_si32(reversed_bcd) + '0';
  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), digits_with_point);
  memcpy(buffer + 16, &trailing_digit, 4);  // only need the lowest byte
#endif  // ZMIJ_USE_SSE4_1 && !ZMIJ_OPTIMIZE_SIZE
  return buffer + (len > point_index ? len + 1 : point_index);
}

struct to_decimal_result {
  long long sig;
  int exp;
  int last_digit = 0;
};

template <typename UInt>
ZMIJ_INLINE auto to_decimal_schubfach(UInt bin_sig, int64_t bin_exp,
                                      bool regular) noexcept
    -> to_decimal_result {
  constexpr int num_bits = std::numeric_limits<UInt>::digits;
  int dec_exp = compute_dec_exp(bin_exp, regular);
  unsigned char exp_shift = compute_exp_shift(bin_exp, dec_exp);
  uint128 pow10 = pow10_significands[-dec_exp];

  // Shubfach requires strict overestimates of powers of 10.
  ++(num_bits == 64 ? pow10.lo : pow10.hi);

  // Shift the significand so that boundaries are integer.
  // The two extra bits act as guard and sticky for correct rounding.
  UInt bin_sig_shifted = bin_sig << 2;
  UInt odd = bin_sig & 1;

  // Compute the lower and upper bounds of the rounding interval by
  // multiplying them by the power of 10 and applying modified rounding.
  UInt lower = (bin_sig_shifted - (regular + 1)) << exp_shift;
  lower = umulhi_inexact_to_odd(pow10.hi, pow10.lo, lower) + odd;
  lower = (lower + 3) >> 2;  // ceil
  UInt upper = (bin_sig_shifted + 2) << exp_shift;
  upper = umulhi_inexact_to_odd(pow10.hi, pow10.lo, upper) - odd;
  upper = upper >> 2;  // floor

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  UInt shorter = (upper / 10) * 10;
  if (shorter >= lower) return {int64_t(shorter), dec_exp};

  // The simplified longer candidate selection is by Russ Cox.
  UInt dec_sig =
      umulhi_inexact_to_odd(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
  dec_sig = (dec_sig + 1 + ((dec_sig >> 2) & 1)) >> 2;  // round
  return {int64_t(lower == upper ? lower : dec_sig), dec_exp};
}

// Returns x / 10 for x <= 2**62.
ZMIJ_INLINE auto div10(uint64_t x) noexcept -> uint64_t {
  assert(x <= (1ull << 62));
  // ceil(2**64 / 10) computed as (1 << 63) / 5 + 1 to avoid int128.
  constexpr uint64_t div10_sig64 = (1ull << 63) / 5 + 1;
  return ZMIJ_USE_INT128 ? umul128_hi64(x, div10_sig64) : x / 10;
}

// Here be 🐉s.
// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - exp_offset.
template <typename Float, typename UInt>
ZMIJ_INLINE auto to_decimal(UInt bin_sig, int64_t raw_exp,
                            bool regular) noexcept -> to_decimal_result {
  using traits = float_traits<Float>;
  int64_t bin_exp = raw_exp - traits::exp_offset;
  constexpr int num_bits = std::numeric_limits<UInt>::digits;

  constexpr uint64_t log10_2_sig = 78'913;
  constexpr int log10_2_exp = 18;
  int dec_exp = use_umul128_hi64
                    ? umul128_hi64(bin_exp, log10_2_sig << (64 - log10_2_exp))
                    : compute_dec_exp(bin_exp);
  uint64_t even = 1 - (bin_sig & 1);
  constexpr uint64_t half = uint64_t(1) << 63;
  constexpr int extra_shift = exp_shift_table::extra_shift;

  if (!regular) [[ZMIJ_UNLIKELY]] {
    int dec_exp = compute_dec_exp(bin_exp, false);
    unsigned char shift = compute_exp_shift(bin_exp, dec_exp + 1) + extra_shift;
    uint128 pow10 = pow10_significands[-dec_exp - 1];
    uint128 p = umul192_hi128(pow10.hi, pow10.lo, bin_sig << shift);

    long long integral = p.hi >> extra_shift;
    uint64_t fractional = p.hi << (64 - extra_shift) | p.lo >> extra_shift;

    uint64_t half_ulp = pow10.hi >> (extra_shift + 1 - shift);
    uint64_t down_half_ulp = half_ulp >> 1;
    bool round_up = half_ulp > ~uint64_t(0) - fractional;
    bool round_down = down_half_ulp > fractional;
    integral += round_up;

    uint64_t rem = fractional * 10;
    int digit = int(umul128_hi64(fractional, 10));
    // Lower midpoint of the asymmetric interval in digit space.
    uint64_t lo_frac = fractional - down_half_ulp;
    uint64_t lo_rem = lo_frac * 10;
    int lo = int(umul128_hi64(lo_frac, 10) + (lo_rem != 0));
    digit += (rem == half) ? (digit & 1) : int(rem + half < rem);
    if (digit < lo) digit = lo;
    digit = (round_up + round_down) != 0 ? 0 : digit;
    if (num_bits == 64) return {integral, dec_exp, digit};
    return {integral * 10 + digit, dec_exp};
  }

  if (num_bits == 32) {
    constexpr int extra_shift = 34;
    unsigned char shift = compute_exp_shift(bin_exp, dec_exp + 1) + extra_shift;
    uint64_t pow10_hi = pow10_significands[-dec_exp - 1].hi;
    uint64_t p = umul128_hi64(pow10_hi + 1, uint64_t(bin_sig) << shift);

    long long integral = p >> extra_shift;
    uint64_t fractional = p & ((1ull << extra_shift) - 1);

    uint64_t half_ulp = (pow10_hi >> (65 - shift)) + even;
    bool round_up = (fractional + half_ulp) >> extra_shift;
    bool round_down = half_ulp > fractional;
    integral += round_up;

    uint64_t prod = fractional * 10;
    int digit = int(prod >> extra_shift);
    uint64_t rem = prod & ((1ull << extra_shift) - 1);
    digit += rem > (1ull << (extra_shift - 1)) ||
             (rem == (1ull << (extra_shift - 1)) && (digit & 1));
    digit = (round_up + round_down) != 0 ? 0 : digit;
    return {integral * 10 + digit, dec_exp};
  }

  // An optimization by Xiang JunBo:
  // Scale by 10**(-dec_exp-1) to directly produce the shorter candidate
  // (15-16 digits), deriving the extra digit from the fractional part.
  // This eliminates div10 from the critical path.
  //
  // value = 5.0507837461e-27
  // next  = 5.0507837461000010e-27
  //
  // c = integral.fractional' = 5050783746100000.3153987... (value)
  //                            5050783746100001.0328635... (next)
  //                 half_ulp =                0.3587324...
  //
  // fractional = fractional' * 2**64 = 5818079786399166407
  //
  //    5050783746100000.0       c               upper    5050783746100001.0
  //             s              l|   L             |               S
  // ──┬────┬────┼────┬────┬────┼*-──┼────┬────┬───*┬────┬────┬────┼-*--┬───
  //  .8   .9   .0   .1   .2   .3   .4   .5   .6   .7   .8   .9   .0 | .1
  //           └─────────────────┼─────────────────┘                next
  //                            1ulp
  //
  // s - shorter underestimate, S - shorter overestimate
  // l - longer underestimate,  L - longer overestimate
  unsigned char shift =
      exp_shift_table::enable
          ? exp_shifts.data[bin_exp + traits::exp_offset]
          : compute_exp_shift(bin_exp, dec_exp + 1) + extra_shift;
  ZMIJ_ASM(("" : "+r"(dec_exp)));  // Force 32-bit reg for sxtw addressing.
  uint128 pow10 = pow10_significands[-dec_exp - 1];
  uint128 p = umul192_hi128(pow10.hi, pow10.lo, bin_sig << shift);

  long long integral = p.hi >> extra_shift;
  uint64_t fractional = p.hi << (64 - extra_shift) | p.lo >> extra_shift;

  uint64_t half_ulp = (pow10.hi >> (extra_shift + 1 - shift)) + even;
  bool round_up = fractional + half_ulp < fractional;
  bool round_down = half_ulp > fractional;
  integral += round_up;  // Compute integral before digit.

  // Derive the extra digit from the fractional part (parallel with
  // rounding). +6 is needed for boundary cases found by verify.py.
  uint64_t rem = fractional * 10;
  int digit = int(umul128_hi64(fractional, 10) + (rem + half + 6 < rem));
  if (fractional == (1ull << 62)) [[ZMIJ_UNLIKELY]]
    digit = 2;  // Round 2.5 to 2.
  return {integral, dec_exp, (round_up + round_down) != 0 ? 0 : digit};
}

}  // namespace

namespace zmij {

inline auto to_decimal(double value) noexcept -> dec_fp {
  using traits = float_traits<double>;
  auto bits = traits::to_bits(value);
  auto bin_exp = traits::get_exp(bits);  // binary exponent
  auto bin_sig = traits::get_sig(bits);  // binary significand
  auto negative = traits::is_negative(bits);
  if (bin_exp == 0 || bin_exp == traits::exp_mask) [[ZMIJ_UNLIKELY]] {
    if (bin_exp != 0) return {int64_t(bin_sig), int(~0u >> 1), negative};
    if (bin_sig == 0) return {0, 0, negative};
    bin_exp = 1;
    bin_sig |= traits::implicit_bit;
  }
  auto dec = ::to_decimal<double>(bin_sig ^ traits::implicit_bit, bin_exp,
                                  bin_sig != 0);
  return {dec.sig * 10 + dec.last_digit, dec.exp, negative};
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

  to_decimal_result dec;
  constexpr uint64_t threshold = uint64_t(traits::num_bits == 64 ? 1e15 : 1e8);
  if (bin_exp == 0 || bin_exp == traits::exp_mask) [[ZMIJ_UNLIKELY]] {
    if (bin_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return buffer + 3;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return buffer + 1;
    }
    dec = to_decimal_schubfach(bin_sig, 1 - traits::exp_offset, true);
    while (dec.sig < threshold) {
      dec.sig *= 10;
      --dec.exp;
    }
    if (traits::num_bits == 64) {
      long long div10 = ::div10(dec.sig);
      dec.last_digit = dec.sig - div10 * 10;
      dec.sig = div10;
    }
  } else {
    dec = ::to_decimal<Float>(bin_sig | traits::implicit_bit, bin_exp,
                              bin_sig != 0);
  }
  int dec_exp = dec.exp;
  bool extra_digit = dec.sig >= threshold;
  dec_exp += traits::max_digits10 - 2 + extra_digit;
  if (traits::num_bits == 32 && dec.sig < uint32_t(1e7)) [[ZMIJ_UNLIKELY]] {
    dec.sig *= 10;
    --dec_exp;
  }

  char* start = buffer;
  constexpr bool split_last_digit = traits::num_bits == 64;
  if (dec_exp >= traits::min_fixed_dec_exp &&
      dec_exp <= traits::max_fixed_dec_exp) {
    if (traits::num_bits == 64 && dec_exp >= 0 && ZMIJ_USE_SIMD_SHUFFLE) {
      dec.sig = dec.sig * 10 + dec.last_digit;
      return write_fixed_double_simd(buffer, dec.sig, dec_exp, extra_digit);
    }
    memcpy(buffer, &zeros, 8);  // For dec_exp < 0.
    const auto& fmt = dec_exp_formats.get<traits>(dec_exp);
    char* sig_start = buffer + fmt.start_pos;
    auto dig = to_digits<traits::num_bits>(sig_start, dec.sig, extra_digit);
    if (split_last_digit) {
      memcpy(sig_start, &dig.digits, 16);
      if (!extra_digit) memmove(sig_start, sig_start + 1, 15);
      sig_start[15 + extra_digit] = '0' + dec.last_digit;
    } else {
      memcpy(sig_start + extra_digit, &dig.digits, sizeof(dig.digits));
    }
    memmove(start + fmt.shift_pos, start + fmt.point_pos, sizeof(dig.digits));
    start[fmt.point_pos] = '.';
    int total = split_last_digit
                    ? (dec.last_digit ? 16 + extra_digit
                                      : dig.num_digits + extra_digit - 1)
                    : dig.num_digits + extra_digit;
    return sig_start + fmt.exp_pos[total - 1];
  }

  auto dig = to_digits<traits::num_bits>(buffer + 1, dec.sig, extra_digit);
  buffer += extra_digit + (split_last_digit ? 0 : 1);
  memcpy(buffer, &dig.digits, sizeof(dig.digits));
  if (split_last_digit) {
    buffer[16] = '0' + dec.last_digit;
    buffer += dec.last_digit ? 17 : dig.num_digits;
  } else {
    buffer += dig.num_digits;
  }
  start[0] = start[1];
  start[1] = '.';
  buffer -= (buffer - 1 == start + 1);  // Remove trailing point.

  // Write exponent.
  if (exp_string_table::enable && traits::num_bits == 64) {
    uint64_t exp_data = exp_strings.data[dec_exp + exp_string_table::offset];
    int len = int(exp_data >> 48);
    if (is_big_endian) exp_data = bswap64(exp_data);
    memcpy(buffer, &exp_data, traits::max_exponent10 >= 100 ? 8 : 4);
    return buffer + len;
  }
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  dec_exp = dec_exp >= 0 ? dec_exp : -dec_exp;
  if (traits::max_exponent10 >= 100) {
    // digit = dec_exp / 100
    uint32_t digit = use_umul128_hi64
                         ? umul128_hi64(dec_exp, 0x290000000000000)
                         : (uint32_t(dec_exp) * div100_sig) >> div100_exp;
    *buffer = '0' + digit;
    buffer += dec_exp >= 100;
    dec_exp -= digit * 100;
  }
  memcpy(buffer, digits2(dec_exp), 2);
  return buffer + 2;
}

template auto write(float value, char* buffer) noexcept -> char*;
template auto write(double value, char* buffer) noexcept -> char*;

}  // namespace detail
}  // namespace zmij
