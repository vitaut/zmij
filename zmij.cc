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
// Use the provided definition
#elif defined(__OPTIMIZE_SIZE__)
#  define ZMIJ_OPTIMIZE_SIZE 1
#else
#  define ZMIJ_OPTIMIZE_SIZE 0
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
  static constexpr int exp_offset = exp_bias + num_sig_bits;

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

constexpr auto floor_log2_pow10(int e) noexcept -> int {
  return e * 1741647 >> 19;
}

constexpr uint64_t pow10s[] = {
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
constexpr uint128 high_parts[] = {
    {0xaf8e5410288e1b6f, 0x07ecf0ae5ee44dda},
    {0xb1442798f49ffb4a, 0x99cd11cfdf41779d},
    {0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa15},
    {0xb4bca50b065abe63, 0x0fed077a756b53aa},
    {0xb67f6455292cbf08, 0x1a3bc84c17b1d543},
    {0xb84687c269ef3bfb, 0x3d5d514f40eea742},
    {0xba121a4650e4ddeb, 0x92f34d62616ce413},
    {0xbbe226efb628afea, 0x890489f70a55368c},
    {0xbdb6b8e905cb600f, 0x5400e987bbc1c921},
    {0xbf8fdb78849a5f96, 0xde98520472bdd034},
    {0xc16d9a0095928a27, 0x75b7053c0f178294},
    {0xc350000000000000, 0x0000000000000000},
    {0xc5371912364ce305, 0x6c28000000000000},
    {0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef6},
    {0xc913936dd571c84c, 0x03bc3a19cd1e38ea},
    {0xcb090c8001ab551c, 0x5cadf5bfd3072cc6},
    {0xcd036837130890a1, 0x36dba887c37a8c10},
    {0xcf02b2c21207ef2e, 0x94f967e45e03f4bc},
    {0xd106f86e69d785c7, 0xe13336d701beba52},
    {0xd31045a8341ca07c, 0x1ede48111209a051},
    {0xd51ea6fa85785631, 0x552a74227f3ea566},
    {0xd732290fbacaf133, 0xa97c177947ad4096},
    {0xd94ad8b1c7380874, 0x18375281ae7822bc},
};
constexpr uint32_t fixups[] = {0x05271b1f, 0x00000c20, 0x00003200, 0x12100020,
                               0x00000000, 0x06000000, 0xc16409c0, 0xaf26700f,
                               0xeb987b07, 0x0000000d, 0x00000000, 0x66fbfffe,
                               0xb74100ec, 0xa0669fe8, 0xedb21280, 0x00000686,
                               0x0a021200, 0x29b89c20, 0x08bc0eda, 0x00000000};

// 128-bit significands of powers of 10 rounded down.
struct pow10_significands_table {
  static constexpr bool compress = ZMIJ_OPTIMIZE_SIZE != 0;
  static constexpr bool split_tables = !compress && ZMIJ_AARCH64 != 0;
  static constexpr int num_pow10 = 617;
  uint64_t data[compress ? 1 : num_pow10 * 2] = {};

  // Computes the 128-bit significand of 10**i using method by Dougall Johnson.
  static constexpr auto compute(unsigned i) noexcept -> uint128 {
    uint64_t m = pow10s[(i + 11) % 28];
    uint128 h = high_parts[(i + 11) / 28];

    uint64_t h1 = umul128_hi64(h.lo, m);

    uint64_t c0 = h.lo * m;
    uint64_t c1 = h1 + h.hi * m;
    uint64_t c2 = (c1 < h1) + umul128_hi64(h.hi, m);

    uint128 result = (c2 >> 63) != 0
                         ? uint128{c2, c1}
                         : uint128{c2 << 1 | c1 >> 63, c1 << 1 | c0 >> 63};
    result.lo -= (fixups[i >> 5] >> (i & 31)) & 1;
    return result;
  }

  constexpr pow10_significands_table() {
    for (int i = 0; i < num_pow10 && !compress; ++i) {
      uint128 result = compute(i);
      if (split_tables) {
        data[num_pow10 - i - 1] = result.hi;
        data[num_pow10 * 2 - i - 1] = result.lo;
      } else {
        data[i * 2] = result.hi;
        data[i * 2 + 1] = result.lo;
      }
    }
  }

  ZMIJ_CONSTEXPR auto operator[](int dec_exp) const noexcept -> uint128 {
    constexpr int dec_exp_min = -292;
    if (compress) return compute(dec_exp - dec_exp_min);
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
};
alignas(64) constexpr pow10_significands_table pow10_significands;

// Computes the decimal exponent as floor(log10(2**bin_exp)) if regular or
// floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
constexpr auto compute_dec_exp(int bin_exp, bool regular = true) noexcept
    -> int {
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
  static constexpr bool enable = ZMIJ_OPTIMIZE_SIZE == 0;
  unsigned char data[enable ? float_traits<double>::exp_mask + 1 : 1] = {};

  constexpr exp_shift_table() {
    for (int raw_exp = 0; raw_exp < sizeof(data) && enable; ++raw_exp) {
      int bin_exp = raw_exp - float_traits<double>::exp_offset;
      if (raw_exp == 0) ++bin_exp;
      int dec_exp = compute_dec_exp(bin_exp);
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
    return exp_shifts.data[bin_exp + float_traits<double>::exp_offset];
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

inline auto read8(char* buffer) noexcept -> uint64_t {
  uint64_t r;
  memcpy(&r, buffer, 8);
  return r;
}

// Writes a significand and removes trailing zeros. value has up to 17 decimal
// digits (16-17 for normals) for double (num_bits == 64) and up to 9 digits
// (8-9 for normals) for float. The significant digits start from buffer[1].
// buffer[0] may contain '0' after this function if the leading digit is zero.
template <int num_bits, bool use_sse = ZMIJ_USE_SSE != 0 && num_bits == 64>
ZMIJ_INLINE auto write_significand(char* buffer, uint64_t value,
                                   bool extra_digit) noexcept -> char* {
  if (num_bits == 32) {
    buffer = write_if(buffer, value / 100'000'000, extra_digit);
    uint64_t bcd = to_bcd8(value % 100'000'000);
    write8(buffer, bcd + zeros);
    return buffer + count_trailing_nonzeros(bcd);
  }
  if (!ZMIJ_USE_NEON && !use_sse) {
    // Digits/pairs of digits are denoted by letters: value = abbccddeeffgghhii.
    uint32_t abbccddee = uint32_t(value / 100'000'000);
    uint32_t ffgghhii = uint32_t(value % 100'000'000);
    buffer = write_if(buffer, abbccddee / 100'000'000, extra_digit);
    uint64_t bcd = to_bcd8(abbccddee % 100'000'000);
    write8(buffer, bcd + zeros);
    if (ffgghhii == 0) {
      write8(buffer + 8, zeros);
      return buffer + count_trailing_nonzeros(bcd);
    }
    bcd = to_bcd8(ffgghhii);
    write8(buffer + 8, bcd + zeros);
    return buffer + 8 + count_trailing_nonzeros(bcd);
  }
#if ZMIJ_USE_NEON
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

  // We could probably make this bit faster, but we're preferring to
  // reuse the constants for now.
  uint64_t a = uint64_t(umul128(abbccddee, c->mul_const) >> 90);
  uint64_t bbccddee = abbccddee - a * hundred_million;

  buffer = write_if(buffer, a, extra_digit);

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

  uint16x8_t is_not_zero =
      vreinterpretq_u16_u8(vcgtzq_s8(vreinterpretq_s8_u8(digits)));
  uint64_t zeroes =
      vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(is_not_zero, 4)), 0);
  return buffer + (16 - ((zeroes != 0 ? clz(zeroes) : 64) >> 2));
#elif ZMIJ_USE_SSE
  uint32_t abbccddee = uint32_t(value / 100'000'000);
  uint32_t ffgghhii = uint32_t(value % 100'000'000);
  uint32_t a = abbccddee / 100'000'000;
  uint32_t bbccddee = abbccddee % 100'000'000;

  buffer = write_if(buffer, a, extra_digit);

  alignas(64) static constexpr struct {
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
#  endif
    uint128 zeros = splat64(::zeros);
  } consts;
  const auto* c = &consts;
  ZMIJ_ASM(("" : "+r"(c)));  // Load constants from memory.

  using ptr = const __m128i*;
  const __m128i div10k = _mm_load_si128(ptr(&c->div10k));
  const __m128i neg10k = _mm_load_si128(ptr(&c->neg10k));
  const __m128i div100 = _mm_load_si128(ptr(&c->div100));
  const __m128i div10 = _mm_load_si128(ptr(&c->div10));
#  if ZMIJ_USE_SSE4_1
  const __m128i neg100 = _mm_load_si128(ptr(&c->neg100));
  const __m128i neg10 = _mm_load_si128(ptr(&c->neg10));
  const __m128i bswap = _mm_load_si128(ptr(&c->bswap));
#  else
  const __m128i hundred = _mm_load_si128(ptr(&c->hundred));
  const __m128i moddiv10 = _mm_load_si128(ptr(&c->moddiv10));
#  endif
  const __m128i zeros = _mm_load_si128(ptr(&c->zeros));

  // The BCD sequences are based on ones provided by Xiang JunBo.
  __m128i x = _mm_set_epi64x(bbccddee, ffgghhii);
  __m128i y = _mm_add_epi64(
      x, _mm_mul_epu32(neg10k,
                       _mm_srli_epi64(_mm_mul_epu32(x, div10k), div10k_exp)));
#  if ZMIJ_USE_SSE4_1
  // _mm_mullo_epi32 is SSE 4.1
  __m128i z = _mm_add_epi64(
      y,
      _mm_mullo_epi32(neg100, _mm_srli_epi32(_mm_mulhi_epu16(y, div100), 3)));
  __m128i big_endian_bcd =
      _mm_add_epi16(z, _mm_mullo_epi16(neg10, _mm_mulhi_epu16(z, div10)));
  __m128i bcd = _mm_shuffle_epi8(big_endian_bcd, bswap);  // SSSE3
#  else   // !ZMIJ_USE_SSE4_1
  __m128i y_div_100 = _mm_srli_epi16(_mm_mulhi_epu16(y, div100), 3);
  __m128i y_mod_100 = _mm_sub_epi16(y, _mm_mullo_epi16(y_div_100, hundred));
  __m128i z = _mm_or_si128(_mm_slli_epi32(y_mod_100, 16), y_div_100);
  __m128i bcd_shuffled =
      _mm_sub_epi16(_mm_slli_epi16(z, 8),
                    _mm_mullo_epi16(moddiv10, _mm_mulhi_epu16(z, div10)));
  __m128i bcd = _mm_shuffle_epi32(bcd_shuffled, _MM_SHUFFLE(0, 1, 2, 3));
#  endif  // ZMIJ_USE_SSE4_1

  auto digits = _mm_or_si128(bcd, zeros);

  // Count leading zeros.
  __m128i mask128 = _mm_cmpgt_epi8(bcd, _mm_setzero_si128());
  uint64_t mask = _mm_movemask_epi8(mask128);
#  if defined(__LZCNT__) && !defined(ZMIJ_NO_BUILTINS)
  auto len = 32 - _lzcnt_u32(mask);
#  else
  auto len = 63 - clz((mask << 1) | 1);
#  endif

  _mm_storeu_si128(reinterpret_cast<__m128i*>(buffer), digits);
  return buffer + len;
#endif  // ZMIJ_USE_SSE
}

struct to_decimal_result {
  long long sig;
  int exp;
};

template <typename UInt>
ZMIJ_INLINE auto to_decimal_schubfach(UInt bin_sig, int64_t bin_exp,
                                      bool regular) noexcept
    -> to_decimal_result {
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
  UInt shorter = ((upper >> bound_shift) / 10) * 10;
  if ((shorter << bound_shift) >= lower) return {int64_t(shorter), dec_exp};

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
  return {int64_t(dec_sig), dec_exp};
}

// Here be 🐉s.
// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - exp_offset.
template <typename Float, typename UInt>
ZMIJ_INLINE auto to_decimal_fast(UInt bin_sig, int64_t raw_exp,
                                 bool regular) noexcept -> to_decimal_result {
  using traits = float_traits<Float>;
  int64_t bin_exp = raw_exp - traits::exp_offset;
  constexpr int num_bits = std::numeric_limits<UInt>::digits;
  // An optimization from yy by Yaoyuan Guo:
  while (regular) [[ZMIJ_LIKELY]] {
    int dec_exp = use_umul128_hi64 ? umul128_hi64(bin_exp, 0x4d10500000000000)
                                   : compute_dec_exp(bin_exp);
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

    // An optimization of integral % 10 by Dougall Johnson.
    // Relies on range calculation: (max_bin_sig << max_exp_shift) * max_u128.
    // (1 << 63) / 5 == (1 << 64) / 10 without an intermediate int128.
    constexpr uint64_t div10_sig64 = (1ull << 63) / 5 + 1;
    long long div10 =
        ZMIJ_USE_INT128 ? umul128_hi64(integral, div10_sig64) : integral / 10;
    uint64_t digit = integral - div10 * 10;
    // or it narrows to 32-bit and doesn't use madd/msub
    ZMIJ_ASM(("" : "+r"(digit)));

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
    // Case where upper == ten is insufficient: 1.342178e+08f.
    if (ten - upper <= 1u ||  // upper == ten || upper == ten - 1
        scaled_sig_mod10 == scaled_half_ulp) [[ZMIJ_UNLIKELY]] {
      break;
    }

    int64_t shorter = int64_t(integral - digit);
    int64_t longer = int64_t(integral + (cmp >= 0));
    int64_t dec_sig =
        select_if_less(scaled_sig_mod10, scaled_half_ulp, shorter, longer);
    return {select_if_less(ten, upper, shorter + 10, dec_sig), dec_exp};
  }
  return to_decimal_schubfach(bin_sig, bin_exp, regular);
}

template <int num_bits>
auto write_fixed(char* buffer, uint64_t dec_sig, int dec_exp,
                 bool extra_digit) noexcept -> char* {
  if (dec_exp < 0) {
    memcpy(buffer, "0.000000", 8);
    return
        write_significand<num_bits>(buffer + 1 - dec_exp, dec_sig, extra_digit);
  }

  // Avoid reading uninitialized memory (would be unnecessary in asm).
  write8(buffer + (num_bits == 64 ? 16 : 7), 0);

  char* start = buffer;
  buffer = write_significand<num_bits>(buffer, dec_sig, extra_digit);

  // Branchless move to make space for the '.' without OOB accesses.
  char* part1 = start + dec_exp + (dec_exp < 2);
  char* part2 = part1 + (dec_exp < 2) + (dec_exp < 9 ? 7 : 0);
  if (num_bits == 64) {
    uint64_t value1 = read8(part1);
    uint64_t value2 = read8(part2);
    write8(part1 + 1, value1);
    write8(part2 + 1, value2);
  } else {
    write8(part1 + 1, read8(part1));
  }

  char* point = start + dec_exp + 1;
  *point = '.';
  return buffer > point ? buffer + 1 : point;
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
  auto dec = to_decimal_fast<double>(bin_sig ^ traits::implicit_bit, bin_exp,
                                     bin_sig != 0);
  return {dec.sig, dec.exp, negative};
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
  constexpr uint64_t threshold = uint64_t(traits::num_bits == 64 ? 1e16 : 1e8);
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
  } else {
    dec = to_decimal_fast<Float>(bin_sig | traits::implicit_bit, bin_exp,
                                 bin_sig != 0);
  }
  int dec_exp = dec.exp;
  bool extra_digit = dec.sig >= threshold;
  dec_exp += traits::max_digits10 - 2 + extra_digit;
  if (traits::num_bits == 32 && dec.sig < uint32_t(1e7)) [[ZMIJ_UNLIKELY]] {
    dec.sig *= 10;
    --dec_exp;
  }

  // Write significand.
  if (dec_exp >= -4 && dec_exp < compute_dec_exp(traits::digits + 1))
    return write_fixed<traits::num_bits>(buffer, dec.sig, dec_exp, extra_digit);
  char* start = buffer;
  buffer =
      write_significand<traits::num_bits>(buffer + 1, dec.sig, extra_digit);
  start[0] = start[1];
  start[1] = '.';
  buffer -= (buffer - 1 == start + 1);  // Remove trailing point.

  // Write exponent.
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian()) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  dec_exp = dec_exp >= 0 ? dec_exp : -dec_exp;
  if (traits::max_exponent10 < 100) {
    memcpy(buffer, digits2(dec_exp), 2);
    return buffer + 2;
  }
  // digit = dec_exp / 100
  uint32_t digit = use_umul128_hi64
                       ? umul128_hi64(dec_exp, 0x290000000000000)
                       : (uint32_t(dec_exp) * div100_sig) >> div100_exp;
  *buffer = '0' + digit;
  buffer += dec_exp >= 100;
  memcpy(buffer, digits2(dec_exp - digit * 100), 2);
  return buffer + 2;
}

template auto write(float value, char* buffer) noexcept -> char*;
template auto write(double value, char* buffer) noexcept -> char*;

}  // namespace detail
}  // namespace zmij
