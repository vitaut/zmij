// Integer formatting for zmij: https://github.com/vitaut/zmij/
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.
//
// Self-contained: shares only zmij.h and zmij-shared.h with the floating-point
// implementation.
// Kernels are selected by ISA: NEON, SSE4.1 (+ AVX2 wide paths and, on Zen 5,
// the float-reciprocal u64 split), plain SSE2, and a scalar SWAR fallback.

#include "zmij.h"
#include "zmij-shared.h"

#include <assert.h>  // assert
#include <stddef.h>  // offsetof
#include <stdint.h>  // uint64_t
#include <string.h>  // memcpy

#include <type_traits>  // std::make_unsigned, std::conditional_t

// The u64 kernel (see itoa): the 20-digit grid as five 4-digit groups from
// two 128-bit products, two group chains running side by side in the two
// 64-bit lanes of an XMM register. It needs SSE4.1 and a 64x64->128
// multiply; other configurations, NEON included, keep the 16 + 4 peel. The top group comes
// from a 7380-byte string table, or under ZMIJ_OPTIMIZE_SIZE from one product
// and two small tables.
#define ZMIJ_USE_U64_LANE_CHAIN (ZMIJ_USE_SSE4_1 && ZMIJ_USE_INT128)

#if ZMIJ_HAS_ATTRIBUTE(noinline)
#  define ZMIJ_NOINLINE __attribute__((noinline))
#elif ZMIJ_MSC_VER
#  define ZMIJ_NOINLINE __declspec(noinline)
#else
#  define ZMIJ_NOINLINE
#endif

// Whether check_room can see the destination's size (see itoa).
#if ZMIJ_HAS_BUILTIN(__builtin_dynamic_object_size)
#  define ZMIJ_CHECK_ROOM 1
#else
#  define ZMIJ_CHECK_ROOM 0
#endif

namespace zmij {
namespace details_int {
namespace {

using detail::uint128;
using detail::uint128_t;
using detail::umul128;

// Traps unless `out` has room for `need` bytes. Skipped when the size isn't
// known, exactly as _FORTIFY_SOURCE does.
ZMIJ_INLINE void check_room(void* out, size_t need) noexcept {
  if (!ZMIJ_CHECK_ROOM) return;
#if ZMIJ_CHECK_ROOM
  size_t room = __builtin_dynamic_object_size(out, 0);
  if (room != size_t(-1) && room < need) __builtin_trap();
#else
  (void)out, (void)need;
#endif
}

// Copy for destinations already covered by a check_room call. Bypasses the
// fortified memcpy, whose per-copy check is a runtime computation whenever
// the destination offset isn't constant -- as in itoa, where the digit count
// decides where the last digits land, costing a call plus clamp branches on
// every conversion.
ZMIJ_INLINE void copy_bytes(void* dst, const void* src, size_t n) noexcept {
#if ZMIJ_HAS_BUILTIN(__builtin_memcpy)
  __builtin_memcpy(dst, src, n);
#else
  memcpy(dst, src, n);  // MSVC: memcpy is an intrinsic and isn't fortified.
#endif
}

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

// Used by the SSE4.1 to_bcd8 length count only.
[[ZMIJ_MAYBE_UNUSED]] inline auto ctz(uint64_t x) noexcept -> int {
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

constexpr auto umul128_hi64(uint64_t x, uint64_t y) noexcept -> uint64_t {
  return uint64_t(umul128(x, y) >> 64);
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
  if (is_big_endian) x = bswap64(x);
  return (size_t(70) - clz((x << 1) | 1)) / 8;  // size_t for native arithmetic
}

// Converts value in the range [0, 100) to a string. GCC generates a bit better
// code when value is pointer-size (https://www.godbolt.org/z/5fEPMT1cc).
// Unused on the AVX2 tiers, whose kernels never leave SIMD registers.
[[ZMIJ_MAYBE_UNUSED]] inline auto digits2(size_t value) noexcept -> const char* {
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

// The u64 kernels' 4-digit group -- the 16 + 4 peel's tail, the 4 + 16
// split's head and the scalar kernel's top group -- as two pairs from one
// product instead of a divide, a multiply and a subtract: n * (5243 << 13),
// the /100 reciprocal 5243 / 2^19 scaled to a 32-bit fraction, has n / 100
// above bit 32 and the remainder's pair under the fraction's top 7 bits. With
// n < 10000 the fraction for remainder r lies in [r/100, r/100 + 0.0023), so
// 128 buckets of width 1/128 never mix two remainders (checked exhaustively);
// the two lookups no longer depend on each other. For the head, mul4[c] =
// 10^(20-c) * (5243 << 13) folds the left-alignment to four digits into the
// same multiply: the aligned head's quotient, 10..99, indexes a pair table,
// and the four bytes store at out with the body's 16-byte store at out + hlen
// covering whatever is surplus.
constexpr uint32_t frac_tail_mul = 5243u << 13;
// The digits of n mod 100 for n < 10000, indexed by bits 25..31 of the
// 32-bit fraction of n * frac_tail_mul. 256 bytes.
struct frac100_table {
  char e[128][2];
};
constexpr auto make_frac100() -> frac100_table {
  frac100_table t{};
  for (uint32_t n = 0; n < 10000; ++n) {
    uint32_t idx = uint32_t(uint64_t(n) * frac_tail_mul) >> 25;
    t.e[idx][0] = char('0' + n % 100 / 10);
    t.e[idx][1] = char('0' + n % 10);
  }
  return t;
}
#if ZMIJ_USE_U64_LANE_CHAIN && !ZMIJ_OPTIMIZE_SIZE
// The u64 head as a table: for g in [0, 1845), its digits without leading
// zeros, left-aligned in four bytes. What follows the digits is dead, the
// body store covers it. 7380 bytes.
struct top4_table {
  char e[1845][4];
};
constexpr top4_table make_top4() {
  top4_table t{};
  for (int v = 0; v < 1845; ++v) {
    char buf[4] = {'0', '0', '0', '0'};
    int n = v, len = 0;
    do { buf[3 - len] = char('0' + n % 10); n /= 10; ++len; } while (n != 0);
    for (int i = 0; i < 4; ++i) t.e[v][i] = i < len ? buf[4 - len + i] : '0';
  }
  return t;
}
#endif
#if ZMIJ_USE_U64_LANE_CHAIN && ZMIJ_OPTIMIZE_SIZE
struct head4_tables {
  // Per count c, 10^(20-c) -- the scale that left-aligns the head to four
  // digits: head * scale < 10000 for every count, and the head is zero below
  // 17 -- times frac_tail_mul, so one multiply both aligns the head and
  // splits it (see write_head4). The aligned head's quotient runs
  // 10..99, so the pair table covers all of it.
  uint64_t mul4[21] = {};
  uint16_t pairs[100] = {};
  constexpr head4_tables() {
    for (int i = 0; i < 100; ++i)
      pairs[i] = uint16_t(('0' + i / 10) | ('0' + i % 10) << 8);
    for (int c = 0; c <= 20; ++c) {
      uint64_t p = 1;
      for (int k = c; k < 20 && c > 16; ++k) p *= 10;
      mul4[c] = p * frac_tail_mul;
    }
  }
};
#endif
#if ZMIJ_USE_U64_LANE_CHAIN
struct u64_lane_tables {
  uint128 k1e4 = {10000, 10000};  // 10^4 in each qword lane's low dword
  // ceil(2^90 / 1e8), a memory operand for both splits: x * it has x / 1e8
  // above bit 90 and (x mod 1e8) / 1e8 below, exact to x * 0.0076 / 2^90.
  uint64_t mul90 = 12379400392853802749ull;
  // Window shuffles by digit count. to_ascii_4x4 leaves the vector as
  // [G2 G4 G3 G5] with each group's digits reversed within its dword; entry
  // c lays digits 5..20 out in print order dropping the first 16 - c bytes
  // (none for c >= 16), zeros after.
  alignas(16) uint8_t window[21][16] = {};
  // By digit count: the body's offset, the head digits max(c - 16, 0).
  uint8_t body_off[21] = {};
  constexpr u64_lane_tables() {
    for (int c = 0; c <= 20; ++c) body_off[c] = uint8_t(c > 16 ? c - 16 : 0);
    constexpr uint8_t print_order[16] = {3, 2, 1, 0,  11, 10, 9,  8,
                                         7, 6, 5, 4,  15, 14, 13, 12};
    for (int c = 0; c <= 20; ++c) {
      int w = c < 16 ? 16 - c : 0;
      for (int j = 0; j < 16; ++j)
        window[c][j] = j + w < 16 ? print_order[j + w] : uint8_t(0x80);
    }
  }
};
#endif

// count_digits tables. The index is the RAW output of the leading-zero
// instruction so no fixup is ever emitted: with LZCNT (or ARM's clz) that is
// clz(n | 1) itself, while on x64 without it clz lowers to bsr ^ 63, so
// indexing by clz(n | 1) ^ 63 cancels back into the plain bsr result. The
// entries are computed rather than spelled out: for MSB position b the estimate
// is the digit count of 2^(b+1) - 1 and the correction threshold is
// 10^(estimate-1) (0 for the one-digit rows, so the correction never fires).
#if defined(__LZCNT__) || (ZMIJ_MSC_VER && defined(__AVX2__)) || !ZMIJ_X86_64
#  define ZMIJ_COUNT_DIGITS_BSR 0  // hardware returns the leading-zero count as-is
#else
#  define ZMIJ_COUNT_DIGITS_BSR 1  // clz evaluated via bsr ^ 63
#endif

struct count_digits_tables {
#if !ZMIJ_USE_NEON
  // Fused form valid only for n < 1e16 (< 2^54): each entry is
  // (estimate << 54) - threshold, so a single 64-bit add + `>> 54` yields the
  // digit count, the power-of-10 compare folded into the add's carry. First
  // member; on NEON the equivalent rows live at the front of `data` instead
  // (see inc_lt1e16_rows).
  uint64_t inc_lt1e16[65] = {};
#endif
  // Digit-count estimate of a 64-bit value by MSB position, plus the
  // power-of-ten thresholds (indexed by estimate) deciding the -1 correction.
  // Row 64 serves n == 0 on the defined-at-zero clz form (never indexed by
  // the bsr form).
  uint8_t estimate[65] = {};
#if ZMIJ_OPTIMIZE_SIZE
  // Correction thresholds indexed by the digit estimate.
  uint64_t pow10[21] = {};
#else
  // Correction threshold indexed by the leading-zero index rather than by the
  // digit estimate.
  uint64_t threshold[65] = {};
#endif

private:
  // Table index of an entry by MSB position.
  static constexpr auto index64(int msb) noexcept -> int {
    return ZMIJ_COUNT_DIGITS_BSR ? msb : 63 - msb;
  }

public:
  // Table index for a value.
  static auto index_of(uint64_t n) noexcept -> uint64_t {
#if ZMIJ_COUNT_DIGITS_BSR
    // bsr is undefined at zero; the | 1 makes it legal and ^ 63 cancels the
    // clz lowering back into the plain bsr result.
    return clz(n | 1) ^ 63;
#elif ZMIJ_HAS_BUILTIN(__builtin_clzg)
    // The hardware count is defined at zero (lzcnt / ARM clz return 64) and
    // row 64 covers it, so no | 1 is needed.
    return unsigned(__builtin_clzg(n, 64));
#elif ZMIJ_MSC_VER && ZMIJ_X86_64
    return __lzcnt64(n);  // BSR == 0 on MSVC x64 implies AVX2, so lzcnt exists
#else
    return clz(n | 1);
#endif
  }

  constexpr count_digits_tables() {
    uint64_t p10[20] = {1};  // 10^i, i in [0, 19]
    for (int i = 1; i < 20; ++i) p10[i] = p10[i - 1] * 10;
#if ZMIJ_OPTIMIZE_SIZE
    for (int t = 2; t <= 20; ++t) pow10[t] = p10[t - 1];
#endif
    for (int b = 0; b < 64; ++b) {
      uint64_t max_val = (uint64_t(2) << b) - 1;  // b == 63 wraps to ~0
      int t = 1;  // digit count of max_val
      while (t < 20 && max_val >= p10[t]) ++t;
      estimate[index64(b)] = uint8_t(t);
#if !ZMIJ_USE_NEON
      if (b < 54)  // n < 1e16 => MSB <= 53
        inc_lt1e16[index64(b)] = (uint64_t(t) << 54) - (t > 1 ? p10[t - 1] : 0);
#endif
#if !ZMIJ_OPTIMIZE_SIZE
      threshold[index64(b)] = t > 1 ? p10[t - 1] : 0;
#endif
    }
    // Row 64: n == 0 under the defined-at-zero clz form. One digit, no
    // correction (threshold[64] stays 0 from the initializer).
    estimate[64] = 1;
#if !ZMIJ_USE_NEON
    inc_lt1e16[64] = uint64_t(1) << 54;
#endif
  }
};

#if ZMIJ_USE_NEON
// The NEON layout stores only rows 10..64: the index -- a leading-zero count
// of a value below 2^54 (or 64 for zero) -- never goes below 10, and the
// entry load wants byte offset index * 8 straight off the pinned data
// pointer (one scaled register-offset ldr, no address add), so rows 0..9 are
// dead space that data's scalar-constant head occupies instead.
struct inc_lt1e16_rows {
  uint64_t rows[55] = {};

  constexpr inc_lt1e16_rows() {
    uint64_t p10[20] = {1};  // 10^i, i in [0, 19]
    for (int i = 1; i < 20; ++i) p10[i] = p10[i - 1] * 10;
    for (int b = 0; b < 54; ++b) {  // n < 1e16 => MSB <= 53, index 63 - b >= 10
      uint64_t max_val = (uint64_t(2) << b) - 1;
      int t = 1;  // digit count of max_val
      while (t < 20 && max_val >= p10[t]) ++t;
      rows[63 - b - 10] = (uint64_t(t) << 54) - (t > 1 ? p10[t - 1] : 0);
    }
    // Row 64: n == 0 under the defined-at-zero clz form (one digit).
    rows[64 - 10] = uint64_t(1) << 54;
  }
};
#endif  // ZMIJ_USE_NEON

struct data {
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

#if ZMIJ_USE_NEON
  static constexpr int32_t neg10k = 0x10000 - 10000;

  using int32x4 = std::conditional_t<ZMIJ_MSC_VER != 0, int32_t[4], int32x4_t>;
  using int16x8 = std::conditional_t<ZMIJ_MSC_VER != 0, int16_t[8], int16x8_t>;

  // Scalar-constant head, exactly 80 bytes: it occupies the dead rows 0..9
  // of the fused count_digits table that follows (see inc_lt1e16_rows), and
  // staying under byte 80 keeps every pair inside ldp immediate range.
  // mul_const and neg1e8 are adjacent for the itoa head's single ldp.
  uint64_t mul_const = 0xabcc77118461cefd;
  // (1 << 32) - 1e8: one madd packs a value's base-1e8 divmod as
  // remainder | quotient << 32 (see to_unshuffled_digits_itoa).
  uint64_t neg1e8 = (uint64_t(1) << 32) - 100000000;
  // u64toa head constants, paired for one ldp: the full-range /1e4 umulh
  // reciprocal (post-shift 11, the compiler's own magic) and the 16-digit
  // threshold 1e16 - 1.
  uint64_t u64toa_consts[2] = {0x346dc5d63886594b, 9999999999999999};
  int32x4 multipliers32 = {div10k_sig, neg10k, div100_sig << 12, neg100};
  int16x8 multipliers16 = {0xce0, neg10};
  uint64_t pad = 0;  // keeps the head at 80 bytes
  uint64_t hundred_million = 100000000;
  // Rows 10..64 of the fused count table, at byte offset 80 == 10 * 8.
  inc_lt1e16_rows inc_rows;
#endif
#if ZMIJ_USE_SSE
  // Ordered so the SIMD kernel constants fit in a single cache line.
  uint128 div100 = splat32(div100_sig);
  uint128 div10 = splat16((1 << 16) / 10 + 1);
#  if ZMIJ_USE_SSE4_1
  uint128 neg100 = splat32(details_int::neg100);
  uint128 neg10 = splat16((1 << 8) - 10);
  // The full 16-byte reversal, as raw memory bytes 15, 14, ..., 0. Written
  // against detail::uint128's lo-first layout: the (hi, lo) constructor takes
  // the *high* half first, so the byte run 15..8 goes in the lo member.
  // (The itoa-benchmark fork spells this with swapped arguments because its
  // local uint128 stores hi first -- the bytes, not the code, are the spec.)
  uint128 bswap = uint128{pack8(7, 6, 5, 4, 3, 2, 1, 0),
                          pack8(15, 14, 13, 12, 11, 10, 9, 8)};
#  else
  uint128 hundred = splat32(100);
  uint128 moddiv10 = splat16(10 * (1 << 8) - 1);
#  endif  // ZMIJ_USE_SSE4_1
  uint128 div10k = splat64(div10k_sig);
  uint128 neg10k = splat64(details_int::neg10k);
  uint128 zeros = splat64(details_int::zeros);
#  if !ZMIJ_USE_SSE4_1
  // 10^lz for lz = 16 - len in [0, 15]: scaling a len-digit value by it moves
  // the MSD into the top digit of the 16-digit field.
  uint64_t scale10[16] = {1,
                          10,
                          100,
                          1000,
                          10000,
                          100000,
                          1000000,
                          10000000,
                          100000000,
                          1000000000,
                          10000000000,
                          100000000000,
                          1000000000000,
                          10000000000000,
                          100000000000000,
                          1000000000000000};
#  endif  // !ZMIJ_USE_SSE4_1
#endif    // ZMIJ_USE_SSE

  // Reverse-and-left-align shuffle for integer output. Indexing at offset `lz`
  // (the leading-zero count) yields a window {15-lz, 14-lz, ..., 0, <zero>...}
  // that reverses an MSB-first BCD vector while dropping `lz` leading zeros in a
  // single pshufb. Indices >= 0x80 emit a zero byte (positions past the last
  // significant digit, which the caller does not write out). Only itoa_body's
  // SSE4.1 pshufb uses it (SSE2 itoa left-aligns by the scale10 pre-scale;
  // the padded bodies use bswap), so it's absent from non-SSE4.1 builds.
#if ZMIJ_USE_SSE4_1
  // Read 16 bytes at offset 16 - len, len in [1, 16], so the reversal drops
  // the 16 - len leading zeros and the 0x80 run zeroes the bytes past them.
  alignas(32) unsigned char revalign_shuffle[32] = {
      15,   14,   13,   12,   11,   10,   9,    8,    7,    6,
      5,    4,    3,    2,    1,    0,    0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80};
  // Like revalign_shuffle, but for itoa_body10's lane layout (lane0 = mid
  // 4 digits, lane1 = top 4, lane2 = bottom 2): indexing at 10 - len drops
  // the leading zeros and emits the ten digits MSB-first in one pshufb.
  alignas(32) unsigned char revalign_shuffle10[26] = {
      7,    6,    5,    4,    3,    2,    1,    0,    9,    8,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
#  if ZMIJ_USE_AVX2
  // Sliding reversal window for itoa_body_head16_pad's 256-bit head+tail pass.
  // The base [7..0, 15..8] is the per-8-byte-half reversal to_ascii16x2_256
  // emits. Loading 16 bytes at offset lz = 16 - hlen gives the head lane
  // (reverse + drop lz leading zeros, 0x80 tail); offset 0 gives the fixed tail
  // lane -- one array serves both, merged with loadu2_m128i. lz is in [0, 13]
  // (hlen in [3, 16]; the [2^63, 2^64) corner has a 3-digit head), so the max
  // read at offset 13 + 16 stays inside the 32 bytes.
  alignas(32) unsigned char mixed_align_shuffle[32] = {
      7,    6,    5,    4,    3,    2,    1,    0,
      15,   14,   13,   12,   11,   10,   9,    8,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
#  endif  // ZMIJ_USE_AVX2
#  if ZMIJ_USE_FMA
  // Sliding gather+trim for itoa_top8. The base picks the 8 top-block
  // digits MSD-first from the four /10 lanes (each 32-bit lane holds
  // [units, tens] in its low 16 bits). Loading 16 bytes at offset (8 - len)
  // fuses the leading-zero trim into the same pshufb; 0x80 lanes past the
  // significant digits emit zero (not stored). Offset <= 7, so the 16-byte read
  // at offset 7 stays inside the 24 bytes.
  alignas(32) unsigned char top8_shuffle[24] = {
      13,   12,   9,    8,    5,    4,    1,    0,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
  // Float constants for the reciprocal digit kernels, kept in the table so they
  // load as disp(base) off the shared data pointer like the integer constants.
  alignas(16) float top8_recip[4] = {1e0f, 1e-2f, 1e-4f, 1e-6f};  // itoa_top8
  alignas(16) float hundred_ps[4] = {100.0f, 100.0f, 100.0f, 100.0f};
#  endif  // ZMIJ_USE_FMA
#endif    // ZMIJ_USE_SSE4_1

#if ZMIJ_USE_NEON
  // Reverse-and-left-align shuffle for integer output. The BCD from
  // to_unshuffled_digits_itoa is LSD-first across the whole vector (byte b =
  // digit 15 - b), so the first 16 entries are the plain descending run;
  // indexing at offset `lz` (leading-zero count) reverses while dropping `lz`
  // leading zeros, and offset 0 is a pure reversal (used by itoa_body16_pad).
  // Indices >= 16 (0x80) emit a zero byte past the last significant digit.
  alignas(32) unsigned char revalign_shuffle[31] = {
      15,   14,   13,   12,   11,   10,   9,    8,    7,    6,   5,
      4,    3,    2,    1,    0,    0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
  // Post-shuffle bias for itoa_i32, row-selected by neg: negatives get '-'
  // on byte 0 (which holds a padded BCD leading zero after the widened
  // shuffle window) and '0' on the digits. Two 16-byte rows rather than a
  // 17-byte sliding pair so the row address is neg << 4 off the base and the
  // struct offset folds into the load's scaled immediate.
  alignas(32) unsigned char sign_bias[2][16] = {
      {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
       '0', '0'},
      {'-', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
       '0', '0'}};
  // u128 wide-path constants: the 128-bit /1e16 reciprocal (hi, lo), the
  // second peel's 5^16 reciprocal (ceil(2^96 / 5^16)), and 1e16 itself --
  // loaded instead of mov/movk-materialized (~15 instructions per wide
  // call). Plain ldrs: the head's ldp range is fully occupied.
  uint64_t u128_consts[4] = {0x39a5652fb1137856, 0xd30baf9a1e626a6d,
                             0x734aca5f6226f0b, 10000000000000000};
#endif  // ZMIJ_USE_NEON

  count_digits_tables cd_tables;
  frac100_table frac100 = make_frac100();
  uint32_t frac_mul = frac_tail_mul;  // loaded, not materialized, on NEON
#if ZMIJ_USE_U64_LANE_CHAIN
  u64_lane_tables u64_lane;
#  if ZMIJ_OPTIMIZE_SIZE
  head4_tables head4;
#  else
  top4_table top4 = make_top4();
#  endif
#endif
};
alignas(64) constexpr data static_data;

#if ZMIJ_USE_U64_LANE_CHAIN
// Writes the u64 value's top four digits at out, left-aligned: the caller's
// body store covers whatever of the four is surplus, all four when the head
// is empty. `head` is the digits above the low 16, at most 1844, and `c` the
// value's total digit count.
#  if ZMIJ_OPTIMIZE_SIZE
// One product: head * mul4[c] has the aligned head's /100 quotient above bit
// 32 -- indexing the pair table -- and the remainder's pair under the
// fraction's top 7 bits (frac100). Storing the two pairs separately beat
// merging them into a word by 0.05..0.17 ns on x86.
ZMIJ_INLINE void write_head4(char* out, uint64_t head, uint64_t c,
                             const data& d) noexcept {
  uint64_t t = head * d.head4.mul4[c];
  copy_bytes(out, &d.head4.pairs[t >> 32], 2);
  copy_bytes(out + 2, d.frac100.e[uint32_t(t) >> 25], 2);
}
#  else
// One load from the 7380-byte table of left-aligned strings, one store.
ZMIJ_INLINE void write_head4(char* out, uint64_t head, uint64_t,
                             const data& d) noexcept {
  copy_bytes(out, d.top4.e[head], 4);
}
#  endif
#endif

#if ZMIJ_USE_NEON
// count_digits_lt_1e16 loads the fused count entry from byte offset
// index * 8 of `data` itself; the index (a leading-zero count of n < 2^54,
// or 64 for zero) is never below 10, and the scalar head must exactly fill
// those dead rows so stored row 0 lands in unreachable row 10's slot.
static_assert(offsetof(data, inc_rows) == 10 * sizeof(uint64_t),
              "scalar head must end exactly at the count table's row 10");
static_assert(offsetof(data, inc_rows) + sizeof(inc_lt1e16_rows::rows) ==
                  65 * sizeof(uint64_t),
              "stored rows must cover count indices 10..64");
#endif

#if ZMIJ_USE_NEON  // An optimized version for NEON by Dougall Johnson.

// Converts four numbers < 10000, one in each 32-bit lane, to BCD digits.
ZMIJ_INLINE auto to_bcd_4x4(int32x4_t efgh_abcd_mnop_ijkl,
                            const data& d) noexcept -> uint8x16_t {
  // Compiler barrier, or clang breaks the subsequent MLA into UADDW + MUL.
  ZMIJ_ASM(("" : "+w"(efgh_abcd_mnop_ijkl)));

  int32x4_t ef_ab_mn_ij =
      vqdmulhq_n_s32(efgh_abcd_mnop_ijkl, d.multipliers32[2]);
  int16x8_t gh_ef_cd_ab_op_mn_kl_ij = vreinterpretq_s16_s32(
      vmlaq_n_s32(efgh_abcd_mnop_ijkl, ef_ab_mn_ij, d.multipliers32[3]));
  int16x8_t high_10s =
      vqdmulhq_n_s16(gh_ef_cd_ab_op_mn_kl_ij, d.multipliers16[0]);
  return vreinterpretq_u8_s16(
      vmlaq_n_s16(gh_ef_cd_ab_op_mn_kl_ij, high_10s, d.multipliers16[1]));
}

#elif ZMIJ_USE_SSE

using m128ptr = const __m128i*;

// Converts four numbers < 10000, one in each 32-bit lane, to BCD digits.
// Digits in each 32-bit lane will be in order for SSE2, reversed for SSE4.1.
ZMIJ_INLINE auto to_bcd_4x4(__m128i y, const data& d) noexcept -> __m128i {
  const __m128i div100 = _mm_load_si128(m128ptr(&d.div100));
  const __m128i div10 = _mm_load_si128(m128ptr(&d.div10));
#  if ZMIJ_USE_SSE4_1
  const __m128i neg100 = _mm_load_si128(m128ptr(&d.neg100));
  const __m128i neg10 = _mm_load_si128(m128ptr(&d.neg10));

  // _mm_mullo_epi32 is SSE 4.1
  __m128i z = _mm_add_epi64(
      y,
      _mm_mullo_epi32(neg100, _mm_srli_epi32(_mm_mulhi_epu16(y, div100), 3)));
  return _mm_add_epi16(z, _mm_mullo_epi16(neg10, _mm_mulhi_epu16(z, div10)));
#  else
  const __m128i hundred = _mm_load_si128(m128ptr(&d.hundred));
  const __m128i moddiv10 = _mm_load_si128(m128ptr(&d.moddiv10));

  __m128i y_div_100 = _mm_srli_epi16(_mm_mulhi_epu16(y, div100), 3);
  __m128i y_mod_100 = _mm_sub_epi16(y, _mm_mullo_epi16(y_div_100, hundred));
  __m128i z = _mm_or_si128(_mm_slli_epi32(y_mod_100, 16), y_div_100);
  return _mm_sub_epi16(_mm_slli_epi16(z, 8),
                       _mm_mullo_epi16(moddiv10, _mm_mulhi_epu16(z, div10)));
#  endif  // ZMIJ_USE_SSE4_1
}

#  if ZMIJ_USE_SSE4_1
// Converts four numbers < 10000, one in each 32-bit lane, to ASCII digits,
// reversed within each 32-bit lane like to_bcd_4x4. The '0' bias is added to
// z in parallel with the 10s mulhi/mullo chain.
ZMIJ_INLINE auto to_ascii_4x4(__m128i y, const data& d) noexcept -> __m128i {
  const __m128i div100 = _mm_load_si128(m128ptr(&d.div100));
  const __m128i div10 = _mm_load_si128(m128ptr(&d.div10));
  const __m128i neg100 = _mm_load_si128(m128ptr(&d.neg100));
  const __m128i neg10 = _mm_load_si128(m128ptr(&d.neg10));
  const __m128i zeros = _mm_load_si128(m128ptr(&d.zeros));

  __m128i z = _mm_add_epi64(
      y,
      _mm_mullo_epi32(neg100, _mm_srli_epi32(_mm_mulhi_epu16(y, div100), 3)));
  __m128i biased = _mm_add_epi16(z, zeros);
  // Compiler barrier to prevent gcc and clang from reassociating and adding
  // zeros to the mullo result, and thus lengthening the critical path.
  ZMIJ_ASM(("" : "+x"(biased)));
  return _mm_add_epi16(biased,
                       _mm_mullo_epi16(neg10, _mm_mulhi_epu16(z, div10)));
}
#  endif  // ZMIJ_USE_SSE4_1

#endif  // ZMIJ_USE_SSE

struct bcd_result {
  uint64_t bcd;
  int len;
};

// to_bcd8 with the base-10000 quotient abcd = abcdefgh / 10000 supplied by the
// caller, so callers that can compute it off the critical path (e.g. the u32
// fallback, where it equals value / 1e6, available in parallel with
// value / 100) skip the chained divide.
ZMIJ_INLINE auto to_bcd8_split(uint32_t abcdefgh, uint32_t abcd) noexcept
    -> bcd_result {
  if (!ZMIJ_USE_SSE && !ZMIJ_USE_NEON) {
    // Three steps BCD. Base 10000 -> base 100 -> base 10, each step adding
    // neg * quotient so quotient and remainder land side by side.
    uint64_t abcd_efgh = abcdefgh + neg10k * uint64_t(abcd);
    uint64_t ab_cd_ef_gh =
        abcd_efgh +
        neg100 * (((abcd_efgh * div100_sig) >> div100_exp) & 0x7f0000007f);
    uint64_t a_b_c_d_e_f_g_h =
        ab_cd_ef_gh +
        neg10 * (((ab_cd_ef_gh * div10_sig) >> div10_exp) & 0xf000f000f000f);
    uint64_t bcd = is_big_endian ? a_b_c_d_e_f_g_h : bswap64(a_b_c_d_e_f_g_h);
    return {bcd, count_trailing_nonzeros(bcd)};
  }

  const auto* d = &static_data;
  ZMIJ_ASM(("" : "+r"(d)));  // Load constants from memory.

#if ZMIJ_USE_NEON
  uint64_t abcd_efgh_64 = abcdefgh + neg10k * uint64_t(abcd);
  int32x4_t abcd_efgh = vcombine_s32(
      vreinterpret_s32_u64(vcreate_u64(abcd_efgh_64)), vdup_n_s32(0));
  uint8x16_t digits_128 = to_bcd_4x4(abcd_efgh, *d);
  uint8x8_t digits = vget_low_u8(digits_128);
  uint64_t bcd = vget_lane_u64(vreinterpret_u64_u8(vrev64_u8(digits)), 0);
  return {bcd, count_trailing_nonzeros(bcd)};
#elif ZMIJ_USE_SSE4_1
  uint64_t abcd_efgh = abcdefgh + neg10k * uint64_t(abcd);
  uint64_t unshuffled_bcd =
      _mm_cvtsi128_si64(to_bcd_4x4(_mm_set_epi64x(0, abcd_efgh), *d));
  int len = unshuffled_bcd ? 8 - ctz(unshuffled_bcd) / 8 : 0;
  return {bswap64(unshuffled_bcd), len};
#elif ZMIJ_USE_SSE
  // Evaluate the 4-digit limbs and arrange them such that we get a result which
  // is in the correct order.
  uint64_t abcd_efgh =
      (uint64_t(abcdefgh) << 32) - uint64_t((10000ull << 32) - 1) * abcd;
  __m128i v = to_bcd_4x4(_mm_set_epi64x(0, abcd_efgh), *d);
#  if defined(__x86_64__) || defined(_M_X64)
  uint64_t bcd = _mm_cvtsi128_si64(v);
#  else
  uint64_t bcd = uint64_t(_mm_cvtsi128_si32(_mm_srli_si128(v, 4))) << 32 |
                 uint32_t(_mm_cvtsi128_si32(v));
#  endif
  return {bcd, count_trailing_nonzeros(bcd)};
#endif  // ZMIJ_USE_SSE
}

auto to_bcd8(uint32_t abcdefgh) noexcept -> bcd_result {
  return to_bcd8_split(
      abcdefgh, uint32_t((uint64_t(abcdefgh) * div10k_sig) >> div10k_exp));
}

// Number of decimal digits in n (1 for n == 0). Branchless: the MSB position
// gives a log10 estimate via a small table, corrected by a single power-of-10
// compare (the Kendall Willets technique, as used by fmt). The tables live in
// `data`, so they load as disp(base) off the same pinned pointer as the other
// constants rather than via their own RIP-relative address.
ZMIJ_INLINE auto count_digits(uint64_t n, const data& d) noexcept -> uint64_t {
  uint64_t z = count_digits_tables::index_of(n);
#if ZMIJ_OPTIMIZE_SIZE
  uint64_t t = d.cd_tables.estimate[z];
  return t - (n < d.cd_tables.pow10[t]);
#else
  return d.cd_tables.estimate[z] - (n < d.cd_tables.threshold[z]);
#endif
}

// Number of decimal digits in n, valid only for n < 1e16. Fused single-load
// form (see inc_lt1e16): one add + shift, no second dependent load or compare.
ZMIJ_INLINE auto count_digits_lt_1e16(uint64_t n, const data& d) noexcept
    -> uint64_t {
  assert(n < uint64_t(1e16));
  uint64_t i = count_digits_tables::index_of(n);
#if ZMIJ_USE_NEON
  // Entry at byte offset i * 8 off the data pointer itself: i >= 10 always
  // (n < 2^54), and data's scalar head occupies rows 0..9's space, so the
  // load is one scaled register-offset ldr with no address add.
  uint64_t inc;
  memcpy(&inc, reinterpret_cast<const char*>(&d) + i * 8, sizeof(inc));
  return (n + inc) >> 54;
#else
  return (n + d.cd_tables.inc_lt1e16[i]) >> 54;
#endif
}

// Number of decimal digits in a 32-bit n (1 for n == 0). Every uint32_t is
// below 1e16, so the fused form covers it and no 32-bit table is needed.
ZMIJ_INLINE auto count_digits(uint32_t n, const data& d) noexcept -> uint64_t {
  return count_digits_lt_1e16(n, d);
}

#if ZMIJ_USE_INT128
// High 128 bits of the 256-bit product a*b.
ZMIJ_INLINE auto mulhi128(uint128_t a, uint128_t b) noexcept -> uint128_t {
  uint64_t a0 = uint64_t(a), a1 = uint64_t(a >> 64);
  uint64_t b0 = uint64_t(b), b1 = uint64_t(b >> 64);
  uint128_t t0 = umul128(a0, b0);
  uint128_t t1 = umul128(a1, b0) + uint64_t(t0 >> 64);
  uint128_t t2 = umul128(a0, b1) + uint64_t(t1);
  return umul128(a1, b1) + uint64_t(t1 >> 64) + uint64_t(t2 >> 64);
}

// Division of uint128_t by 1e16, explicit implementation avoids libcall.
struct divmod_1e16_result {
  uint128_t quot;
  uint64_t rem;
};
ZMIJ_INLINE auto divmod_1e16(uint128_t n,
                             [[ZMIJ_MAYBE_UNUSED]] const data& d) noexcept
    -> divmod_1e16_result {
#if ZMIJ_USE_NEON
  // The constants come from static_data (see u128_consts); materializing
  // them costs mov + 3 movk apiece.
  const uint128_t magic =
      (uint128_t(d.u128_consts[0]) << 64) | d.u128_consts[1];
  uint128_t q = mulhi128(n, magic) >> 51;
  return {q, uint64_t(n - q * d.u128_consts[3])};
#else
  const uint128_t magic =
      (uint128_t(0x39a5652fb1137856ull) << 64) | 0xd30baf9a1e626a6dull;
  uint128_t q = mulhi128(n, magic) >> 51;
  return {q, uint64_t(n - q * uint64_t(1e16))};
#endif
}

// Divmod by 1e16 for the second peel: n = value / 1e16 < 2**75, so with
// 1e16 = 2**16 * 5**16 the quotient needs only one 64-bit reciprocal --
// (n >> 16) < 2**59 divided by 5**16 (magic exact to 2**62) -- and the
// remainder, < 1e16, comes from the low 64 bits alone.
struct divmod_1e16_narrow_result {
  uint32_t quot;
  uint64_t rem;
};

ZMIJ_INLINE auto divmod_1e16_narrow(
    uint128_t n, [[ZMIJ_MAYBE_UNUSED]] const data& d) noexcept
    -> divmod_1e16_narrow_result {
#if ZMIJ_USE_NEON
  uint32_t q =
      uint32_t(umul128_hi64(uint64_t(n >> 16), d.u128_consts[2]) >> 32);
  // Remainder is evaluated mod 2**64; quot is <= 7 digits, so uint32_t holds
  // it and every consumer gets the cheaper 32-bit count_digits.
  return {q, uint64_t(n) - q * d.u128_consts[3]};
#else
  constexpr uint64_t div5p16_sig = 0x734aca5f6226f0b;  // ceil(2**96 / 5**16)
  uint32_t q = uint32_t(umul128_hi64(uint64_t(n >> 16), div5p16_sig) >> 32);
  // Remainder is evaluated mod 2**64; quot is <= 7 digits, so uint32_t holds
  // it and every consumer gets the cheaper 32-bit count_digits.
  return {q, uint64_t(n) - q * uint64_t(1e16)};
#endif
}
#endif  // ZMIJ_USE_INT128

#if ZMIJ_USE_NEON
// to_unshuffled_digits with the base-1e8 split packed by one madd -- value +
// ((1 << 32) - 1e8) * (value / 1e8) = remainder | quotient << 32 -- placing
// the low 8-digit group in lane 0. The BCD bytes then come out LSD-first
// across the whole vector (byte b = digit 15 - b), matching
// revalign_shuffle's descending windows.
ZMIJ_INLINE auto to_unshuffled_digits_itoa(uint64_t value, const data& d)
    -> uint8x16_t {
  uint64_t abcdefgh = uint64_t(umul128(value, d.mul_const) >> 90);
  uint64_t packed = value + d.neg1e8 * abcdefgh;
  // Compiler barrier, or clang unpairs the 64-bit pack into vector inserts.
  ZMIJ_ASM(("" : "+r"(packed)));
  int32x2_t ijklmnop_abcdefgh = vreinterpret_s32_u64(vcreate_u64(packed));

  int32x2_t ijkl_abcd = vreinterpret_s32_u32(
      vshr_n_u32(vreinterpret_u32_s32(
                     vqdmulh_n_s32(ijklmnop_abcdefgh, d.multipliers32[0])),
                 9));
  int32x2_t mnop_ijkl_efgh_abcd_32 =
      vmla_n_s32(ijklmnop_abcdefgh, ijkl_abcd, d.multipliers32[1]);

  int32x4_t mnop_ijkl_efgh_abcd = vreinterpretq_s32_u32(
      vshll_n_u16(vreinterpret_u16_s32(mnop_ijkl_efgh_abcd_32), 0));
  return to_bcd_4x4(mnop_ijkl_efgh_abcd, d);
}

// Build the 16-wide BCD of value in [0, 1e16), convert to ASCII, and
// apply shuffle.
ZMIJ_INLINE auto to_ascii16_and_shuffle(uint64_t value, uint8x16_t shuffle,
                                        const data& d) noexcept -> uint8x16_t {
  uint8x16_t ascii =
      vaddq_u8(to_unshuffled_digits_itoa(value, d), vdupq_n_u8('0'));
  return vqtbl1q_u8(ascii, shuffle);
}

// Convert value in [0, 1e16) to ASCII digits, write left-aligned at out.
// Returns the past-the-end pointer, out + len. `len` is the digit count of
// `value`, which the caller supplies (it depends only on `value`, so it
// computes in parallel with the BCD work; u32-range callers use the cheaper
// fused 32-bit counter).
// Mirrors the SSE4.1 itoa_body, folding the BCD reversal and leading-zero drop
// into one vqtbl1q_u8.
ZMIJ_INLINE char* itoa_body(char* out, uint64_t value, uint64_t len,
                            const data& d) noexcept {
  uint64_t leading_zeroes = 16 - len;
  uint8x16_t shuffle = vld1q_u8(d.revalign_shuffle + leading_zeroes);
  vst1q_u8(reinterpret_cast<uint8_t*>(out),
           to_ascii16_and_shuffle(value, shuffle, d));
  return out + len;
}

// No narrow 32-bit kernel on NEON (the SSE4.1 SWAR-pack diet measured
// exact instruction-count and time parity with this route on the M5); the
// 32-bit entry forwards to the 16-digit body.
ZMIJ_INLINE char* itoa_body10(char* out, uint32_t value, uint64_t len,
                              const data& d) noexcept {
  return itoa_body(out, uint64_t(value), len, d);
}

// Signed 32-bit body folding the '-' into the single 16-byte store: the
// shuffle window is widened by one for negatives (byte 0 then holds a padded
// BCD leading zero), and the post-shuffle bias -- sign_bias at offset
// 1 - neg -- turns that zero into '-' while biasing the digits with '0'.
ZMIJ_INLINE char* itoa_i32(char* out, int32_t value, const data& d) noexcept {
  uint32_t mag = value >= 0 ? uint32_t(value) : -uint32_t(value);
  uint64_t neg = value < 0;
  uint64_t chars = count_digits(mag, d) + neg;
  uint8x16_t shuffle = vld1q_u8(d.revalign_shuffle + (16 - chars));
  uint8x16_t bias = vld1q_u8(d.sign_bias[neg]);
  vst1q_u8(
      reinterpret_cast<uint8_t*>(out),
      vaddq_u8(vqtbl1q_u8(to_unshuffled_digits_itoa(mag, d), shuffle), bias));
  return out + chars;
}

// Writes exactly 16 ASCII digits of `value` in [0, 1e16) at `out`, zero-padded,
// no length trim -- a mid/low 16-digit chunk of a u128. Offset 0 into
// revalign_shuffle is a pure reversal (the analogue of the SSE4.1 bswap).
ZMIJ_INLINE void itoa_body16_pad(char* out, uint64_t value,
                                 const data& d) noexcept {
  uint8x16_t shuffle = vld1q_u8(d.revalign_shuffle);
  vst1q_u8(reinterpret_cast<uint8_t*>(out),
           to_ascii16_and_shuffle(value, shuffle, d));
}

// Writes 32 ASCII digits: `mid` at out[0,16) then `low` at out[16,32), both
// zero-padded. Mirrors the SSE4.1 (non-AVX2) two-chunk itoa_body32_pad.
ZMIJ_NOINLINE static void itoa_body32_pad(char* out, uint64_t mid,
                                          uint64_t low,
                                          const data& d) noexcept {
  itoa_body16_pad(out, mid, d);
  itoa_body16_pad(out + 16, low, d);
}

#elif ZMIJ_USE_SSE4_1

// Mirrors the NEON implementation

// Builds the 16 ASCII digits from the two 8-digit lanes (hi = value / 1e8,
// lo = value % 1e8) and applies shuffle.
ZMIJ_INLINE auto to_ascii16_lanes_and_shuffle(uint32_t hi, uint32_t lo,
                                              const __m128i& shuffle,
                                              const data& d) noexcept
    -> __m128i {
  const __m128i div10k = _mm_load_si128(m128ptr(&d.div10k));
  const __m128i neg10k = _mm_load_si128(m128ptr(&d.neg10k));
  __m128i x = _mm_set_epi64x(hi, lo);
  __m128i y = _mm_add_epi64(
      x, _mm_mul_epu32(neg10k,
                       _mm_srli_epi64(_mm_mul_epu32(x, div10k), div10k_exp)));
  return _mm_shuffle_epi8(to_ascii_4x4(y, d), shuffle);
}

ZMIJ_INLINE auto to_ascii16_and_shuffle(uint64_t value, const __m128i& shuffle,
                                        const data& d) noexcept -> __m128i {
  return to_ascii16_lanes_and_shuffle(uint32_t(value / 100'000'000),
                                      uint32_t(value % 100'000'000), shuffle,
                                      d);
}

// Unused where the u64 path is the lane chain and the u128 paths use
// itoa_body_head16_pad / itoa_top8.
[[ZMIJ_MAYBE_UNUSED]] ZMIJ_INLINE char* itoa_body(char* out, uint64_t value,
                                                  uint64_t len,
                                                  const data& d) noexcept {
  uint64_t leading_zeroes = 16 - len;
  __m128i shuffle = _mm_loadu_si128(
      reinterpret_cast<const __m128i*>(d.revalign_shuffle + leading_zeroes));

  __m128i ascii = to_ascii16_and_shuffle(value, shuffle, d);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), ascii);
  return out + len;
}

ZMIJ_INLINE char* itoa_body10(char* out, uint32_t value, uint64_t len,
                              const data& d) noexcept {
  __m128i shuffle =
      _mm_loadu_si128(m128ptr(d.revalign_shuffle10 + (10 - len)));
  uint32_t high8 = value / 100;
  uint32_t top4 = value / 1'000'000;
  uint32_t low2 = value - high8 * 100;
  uint64_t ae = high8 + neg10k * uint64_t(top4);
  __m128i x = _mm_set_epi64x(low2, ae);
  __m128i ascii = to_ascii_4x4(x, d);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                   _mm_shuffle_epi8(ascii, shuffle));
  return out + len;
}

// Write value in [0, 1e16) as ASCII at out, zero-padded.
// This is a mid/low 16-digit chunk of a u128. Unused under AVX2, which fuses
// the padded chunks into 256-bit passes.
[[ZMIJ_MAYBE_UNUSED]] ZMIJ_INLINE void itoa_body16_pad(
    char* out, uint64_t value, const data& d) noexcept {
  __m128i shuffle = _mm_load_si128(m128ptr(&d.bswap));

  __m128i ascii = to_ascii16_and_shuffle(value, shuffle, d);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), ascii);
}

#  if ZMIJ_USE_AVX2
// Broadcasts the 64-bit splat pattern of a data constant straight from
// memory.  Reading eight bytes is enough for all, doing it this way
// measured faster than expanding the constants to 32 bytes.
ZMIJ_INLINE auto bcastq256(const void* p) noexcept -> __m256i {
  int64_t v;
  copy_bytes(&v, p, sizeof v);
  return _mm256_set1_epi64x(v);
}

// 32 ASCII digits of two values in [0, 1e16): lane0 = a, lane1 = b. Same
// sequence as the SSE4.1 to_ascii_4x4, but widened to two 128-bit lanes; the
// '0' bias is added to z in parallel with the 10s mulhi/mullo chain. The lane
// setup is picked to minimize setup time (remainders become available after
// quotients). The caller shuffles accordingly.
ZMIJ_INLINE auto to_ascii16x2_256(uint64_t a, uint64_t b,
                                  const data& d) noexcept -> __m256i {
  uint32_t ah = uint32_t(a / 100'000'000), al = uint32_t(a % 100'000'000);
  uint32_t bh = uint32_t(b / 100'000'000), bl = uint32_t(b % 100'000'000);
  __m256i x = _mm256_set_epi64x(bl, bh, al, ah);  // lane0 = a, lane1 = b
  const __m256i div10k = bcastq256(&d.div10k);
  const __m256i neg10k = bcastq256(&d.neg10k);
  const __m256i div100 = bcastq256(&d.div100);
  const __m256i div10 = bcastq256(&d.div10);
  const __m256i neg100 = bcastq256(&d.neg100);
  const __m256i neg10 = bcastq256(&d.neg10);
  const __m256i zeros = bcastq256(&d.zeros);
  __m256i y = _mm256_add_epi64(
      x,
      _mm256_mul_epu32(
          neg10k, _mm256_srli_epi64(_mm256_mul_epu32(x, div10k), div10k_exp)));
  __m256i z = _mm256_add_epi64(
      y, _mm256_mullo_epi32(
             neg100, _mm256_srli_epi32(_mm256_mulhi_epu16(y, div100), 3)));
  // Compiler barrier, or gcc reassociates the final sum to
  // (product + zeros) + z, moving the bias back onto the critical path.
  __m256i biased = _mm256_add_epi16(z, zeros);
  ZMIJ_ASM(("" : "+x"(biased)));
  return _mm256_add_epi16(
      biased, _mm256_mullo_epi16(neg10, _mm256_mulhi_epu16(z, div10)));
}

// Writes 32 ASCII digits: mid (16 digits) at out[0,16) then low at out[16,32),
// both zero-padded. Left inlinable under AVX2: the itoa_top8 u128 tail
// benefits (gcc gains ~1-2.5%, clang unchanged). The other tiers keep the
// noinline barrier, which otherwise pessimizes their other cases via extra
// stack adjustment.
static void itoa_body32_pad(char* out, uint64_t mid, uint64_t low,
                            const data& d) noexcept {
  __m256i ascii_bcd = to_ascii16x2_256(mid, low, d);  // lane0 = mid, lane1 = low
  __m256i shuffle = _mm256_broadcastsi128_si256(
      _mm_load_si128(m128ptr(d.mixed_align_shuffle)));
  __m256i ascii = _mm256_shuffle_epi8(ascii_bcd, shuffle);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), ascii);
}

// Writes a trimmed head chunk (`head` < 1e16, `hlen` significant digits) left-
// aligned at `out`, immediately followed by the fixed 16-digit `tail` chunk at
// `out + hlen`. Both chunks are converted in one 256-bit pass; lane0 gets the
// reverse+trim revalign shuffle, lane1 the plain reversal. The two 16-byte
// lanes are stored to separate addresses (offset by hlen), so the digits
// concatenate without any lane-crossing shuffle. Returns out + hlen + 16.
ZMIJ_INLINE char* itoa_body_head16_pad(char* out, uint64_t head, uint64_t hlen,
                                       uint64_t tail, const data& d) noexcept {
  __m256i ascii_bcd = to_ascii16x2_256(head, tail, d);  // lane0=head, lane1=tail
  // lane0 (head): sliding window at offset lz = 16 - hlen; lane1 (tail): the
  // fixed reversal at offset 0. Both come from the one mixed_align_shuffle
  // array.
  __m256i mask =
      _mm256_loadu2_m128i(m128ptr(d.mixed_align_shuffle),
                          m128ptr(d.mixed_align_shuffle + (16 - hlen)));
  __m256i ascii = _mm256_shuffle_epi8(ascii_bcd, mask);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out),
                   _mm256_castsi256_si128(ascii));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out + hlen),
                   _mm256_extracti128_si256(ascii, 1));
  return out + hlen + 16;
}
#  else
ZMIJ_INLINE void itoa_body32_pad(char* out, uint64_t mid, uint64_t low,
                                 const data& d) noexcept {
  itoa_body16_pad(out, mid, d);
  itoa_body16_pad(out + 16, low, d);
}
#  endif  // ZMIJ_USE_AVX2

#elif ZMIJ_USE_SSE

// Builds the 16 first ASCII digits of `value` in [0, 1e16), right-aligned
// and zero-padded.
ZMIJ_INLINE auto to_ascii16(uint64_t value, const data& d) noexcept
    -> __m128i {
  uint32_t hi = uint32_t(value / 100'000'000);
  uint32_t lo = uint32_t(value % 100'000'000);
  const __m128i div10k = _mm_load_si128(m128ptr(&d.div10k));
  const __m128i neg10k = _mm_load_si128(m128ptr(&d.neg10k));
  __m128i x = _mm_set_epi64x(hi, lo);
  __m128i y = _mm_add_epi64(
      x, _mm_mul_epu32(neg10k,
                       _mm_srli_epi64(_mm_mul_epu32(x, div10k), div10k_exp)));
  y = _mm_shuffle_epi32(y, _MM_SHUFFLE(0, 1, 2, 3));
  return _mm_or_si128(to_bcd_4x4(y, d), _mm_load_si128(m128ptr(&d.zeros)));
}

// Writes value in [0, 1e16) as ASCII at dst, zero-padded.
ZMIJ_INLINE void itoa_body16_pad(char* dst, uint64_t value,
                                 const data& d) noexcept {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), to_ascii16(value, d));
}

ZMIJ_INLINE void itoa_body32_pad(char* dst, uint64_t mid, uint64_t low,
                                 const data& d) noexcept {
  itoa_body16_pad(dst, mid, d);
  itoa_body16_pad(dst + 16, low, d);
}

// Convert value in [0, 1e16) to ASCII digits, write left-aligned at out.
// Returns the past-the-end pointer, out + len. `len` is the digit count of
// `value`, which the caller supplies.
ZMIJ_INLINE char* itoa_body(char* out, uint64_t value, uint64_t len,
                            const data& d) noexcept {
  // Scale by 10^(16 - len) so the MSD lands in the top digit of the field:
  // the kernel's output is left-aligned in the register and the XMM stores
  // directly -- no GPR extraction and no 128-bit shift. The kernel now waits
  // on the digit count and the scale multiply, but the returned length does
  // not.
  uint64_t scaled = value * d.scale10[16 - len];
  _mm_storeu_si128(reinterpret_cast<__m128i*>(out), to_ascii16(scaled, d));
  return out + len;
}

// Left-aligned head for a value < 1e8 (the 33-39-digit u128 top group): one
// 8-digit BCD group, right-aligned in a u64, shifted down by the leading
// zeros and stored as 8 bytes -- the following padded chunks overwrite the
// bytes past len. Cheaper than the 16-digit kernel, whose scaled form would
// wait on the digit count at the end of the two-divmod u128 chain. Mirrors
// the scalar u32 path's high-group store.
ZMIJ_INLINE char* itoa_head7(char* out, uint32_t top, uint64_t len) noexcept {
  uint64_t bcd = to_bcd8(top).bcd + zeros;
  uint64_t sh = 8 * (8 - len);
  uint64_t aligned = is_big_endian ? bcd << sh : bcd >> sh;
  copy_bytes(out, &aligned, 8);
  return out + len;
}

#endif  // ZMIJ_USE_SSE

#if ZMIJ_USE_FMA
// The u128 highest block: top < 1e7 (<= 7 digits, so < 2^24 and exact in f32).
// Four base-100 blocks via the FP reciprocals -- no scalar divide -- then each
// block split to two digits by one SWAR /10 (the same mul/shift the scalar
// paths use), left-trimmed to `len` significant digits and stored. No LUT, no
// memcpy, one store.
ZMIJ_INLINE char* itoa_top8(char* out, uint32_t top, uint64_t len,
                            const data& d) noexcept {
  // top < 1e7 < 2^24, so top and every floor(top / 10^k) are exact in f32 and
  // the naive reciprocals truncate correctly -- no shift, no fold needed here.
  __m128 x = _mm_cvtepi32_ps(_mm_set1_epi32(int(top)));
  __m128 q = _mm_round_ps(
      _mm_mul_ps(x, _mm_load_ps(d.top8_recip)),
      _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);  // [top, /100, /1e4, /1e6]
  __m128 prev = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(q), 4));
  // 4 base-100 blocks in [0,99], each in the low 16 bits of a 32-bit lane.
  __m128i blocks =
      _mm_cvttps_epi32(_mm_fnmadd_ps(_mm_load_ps(d.hundred_ps), prev, q));
  const __m128i div10 = _mm_load_si128(m128ptr(&d.div10));
  const __m128i neg10 = _mm_load_si128(m128ptr(&d.neg10));
  const __m128i zeros = _mm_load_si128(m128ptr(&d.zeros));
  __m128i biased = _mm_add_epi16(blocks, zeros);
  __m128i digs = _mm_add_epi16(
      biased, _mm_mullo_epi16(neg10, _mm_mulhi_epu16(blocks, div10)));
  // One pshufb both gathers the 8 digits MSD-first and drops the (8 - len)
  // leading zeros, via the sliding top8_shuffle window.
  __m128i sh = _mm_loadu_si128(m128ptr(d.top8_shuffle + (8 - len)));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(out),
                   _mm_shuffle_epi8(digs, sh));
  return out + len;
}
#endif  // ZMIJ_USE_FMA

#if ZMIJ_USE_INT128
ZMIJ_INLINE auto itoa_u128(char* out, uint128_t value) noexcept -> char*;
#endif

}  // namespace

// Writes the decimal representation of unsigned value to out. Minimum buffer
// sizes: u32 -> 16, u64 -> 20, u128 -> 48 bytes (+1 for the sign in the signed
// wrappers). Returns one past the last digit.
template <typename UInt>
ZMIJ_INLINE auto itoa(char* out, UInt value) noexcept -> char* {
  // One check up front covers every copy below, which then need none of their
  // own: the kernels write within this bound regardless of the digit count.
  check_room(out, sizeof(UInt) <= 4   ? uint32_buffer_size
                  : sizeof(UInt) <= 8 ? uint64_buffer_size
                                      : uint128_buffer_size);
#if ZMIJ_USE_INT128
  // Guarded like the sizeof(Int) > 8 block in itoa_signed: MSVC at /Od keeps
  // the reference to itoa_u128 in the statically-false branch, so the
  // preprocessor has to remove it where no 128-bit support exists.
  if (sizeof(UInt) > 8) {
    return itoa_u128(out, value);
  } else
#endif
  {
    uint64_t v = value;
    const auto* d = &static_data;
    ZMIJ_ASM(("" : "+r"(d)));  // Load constants from memory.
    if (sizeof(UInt) <= 4) {
#if ZMIJ_USE_SSE4_1 || ZMIJ_USE_NEON
      return itoa_body10(out, uint32_t(value),
                         count_digits(uint32_t(value), *d), *d);
#else
      uint32_t high8 = uint32_t(value) / 100;
      uint32_t top4 = uint32_t(value) / 1'000'000;
      uint64_t hi = to_bcd8_split(high8, top4).bcd + zeros;
      uint32_t low2 = uint32_t(value) - high8 * 100;
      uint64_t len = count_digits(uint32_t(value), *d);
      uint64_t len_hi = len < 2 ? 0 : len - 2;  // 0..8 significant high chars
      uint64_t sh = (8 * (8 - len_hi)) & 63;  // &63: hsig==0 => shift is dead anyway
      uint64_t hi_aligned = is_big_endian ? hi << sh : hi >> sh;
      copy_bytes(out, &hi_aligned, 8);  // high sig at out[0..hsig)
      const char* d2 = digits2(low2);
      out[len_hi] = d2[0];
      out[len - 1] = d2[1];  // if len == 1 overwrites d2[0]
      return out + len;
#endif
    } else {
#if ZMIJ_USE_U64_LANE_CHAIN && ZMIJ_USE_SSE4_1
      // Five 4-digit groups G1..G5 on the 20-digit grid. Two products by
      // ceil(2^90 / 1e8): v's gives q = v / 1e8 above bit 90 and G4 G5 as
      // the fraction below; q's gives G1 above bit 90 and G2 G3 below. The
      // top 32 fraction bits, rounded up, feed one lane each: eight digits
      // from a 32-bit fraction allow 42.9 units of 2^-32 of error and get
      // under 2. Two pmuludq by 10^4 then expose a group per lane per step
      // in the high dword, the low dword being the next fraction already in
      // place. The count is consumed only at the end, by the window shuffle
      // and the store offset, so the first product starts from v at once.
      uint64_t c = count_digits(v, *d);
      uint128_t p1 = uint128_t(v) * d->u64_lane.mul90;
      uint64_t q = uint64_t(p1 >> 90);  // v / 1e8, < 1.85e11
      uint32_t f_b = uint32_t(p1 >> 58) + 1;  // digits 13..20
      uint128_t p2 = uint128_t(q) * d->u64_lane.mul90;
      uint64_t g1 = uint64_t(p2 >> 90);  // digits 1..4, <= 1844
      uint32_t f_a = uint32_t(p2 >> 58) + 1;  // digits 5..12

      const __m128i k1e4 = _mm_load_si128(m128ptr(&d->u64_lane.k1e4));
      __m128i x = _mm_set_epi64x(int64_t(f_b), int64_t(f_a));  // lane 0 a
      __m128i s1 = _mm_mul_epu32(x, k1e4);   // G2 | G4 in the high dwords
      __m128i s2 = _mm_mul_epu32(s1, k1e4);  // G3 | G5
      __m128i g = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(s1), _mm_castsi128_ps(s2), _MM_SHUFFLE(3, 1, 3, 1)));
      __m128i ascii = to_ascii_4x4(g, *d);  // [G2 G4 G3 G5], reversed within
      // The head is G1's four digits, left-aligned at out; the window
      // shuffle puts digits 5..20 in print order, left-aligned for c <= 16,
      // stored over the head's unused bytes at out + (c - 16 or 0), the
      // offset from a byte table by count.
      __m128i window = _mm_loadu_si128(m128ptr(d->u64_lane.window[c]));
      write_head4(out, g1, c, *d);
      _mm_storeu_si128(
          reinterpret_cast<__m128i*>(out + d->u64_lane.body_off[c]),
          _mm_shuffle_epi8(ascii, window));
      return out + c;
#elif ZMIJ_USE_SSE || ZMIJ_USE_NEON
      // We peel off the last four digits and always write them at the end,
      // but if the number is < 10000 we don't move them around but instead
      // fill them into the SIMD kernel.  This benchmarked fastest out of
      // the variations that I tried.
#  if ZMIJ_USE_NEON
      // The big constants (the /1e4 umulh reciprocal, the 1e16 - 1 threshold
      // and the tail's /100 pair) load from static_data ldp pairs; letting
      // the compiler materialize them costs seven mov/movk and, measured on
      // the M5, ~a cycle per call.
      uint64_t high = umul128_hi64(v, d->u64toa_consts[0]) >> 11;
      uint64_t big = v > d->u64toa_consts[1];
      uint32_t low4 = uint32_t(v - high * 10000);
      uint64_t body = big ? high : v;  // body < 1e16 either way
      char* p = itoa_body(out, body, count_digits_lt_1e16(body, *d), *d);
      uint64_t t = uint64_t(low4) * d->frac_mul;
      copy_bytes(p, digits2(t >> 32), 2);
      copy_bytes(p + 2, d->frac100.e[uint32_t(t) >> 25], 2);
      // The trailing digits only count if they aren't redundant.
      return p + 4 * big;
#  else
      uint64_t high = v / 10000;
      uint32_t low4 = uint32_t(v - high * 10000);
      uint64_t big = v >= uint64_t(1e16);
      uint64_t body = big ? high : v;  // body < 1e16 either way
      char* p = itoa_body(out, body, count_digits_lt_1e16(body, *d), *d);
      uint64_t t = uint64_t(low4) * frac_tail_mul;
      copy_bytes(p, digits2(t >> 32), 2);
      copy_bytes(p + 2, d->frac100.e[uint32_t(t) >> 25], 2);
      // The trailing digits only count if they aren't redundant.
      return p + 4 * big;
#  endif  // ZMIJ_USE_NEON
#else
      // u64: at most 20 digits -> three groups (top, mid 8, low 8). The top
      // group is v / 1e16 in [0, 1844], at most 4 digits, so divmod100 + two
      // digits2 lookups beat a full to_bcd8. Right-aligned, those 4 bytes land
      // at buf[4..8) -- where the len==20 read window (buf + 24 - len) begins.
      char buf[48] = {};
      uint64_t q = v / 100'000'000ull;
      uint32_t top = uint32_t(v / uint64_t(1e16));  // <= 1844
      uint64_t lo = to_bcd8(uint32_t(v - q * 100'000'000ull)).bcd + zeros;
      uint64_t mid = to_bcd8(uint32_t(q - top * 100'000'000ull)).bcd + zeros;
      // One product: top / 100 (at most 18) above bit 32, the remainder's
      // pair under the fraction's top 7 bits.
      uint64_t t = uint64_t(top) * frac_tail_mul;
      copy_bytes(buf + 4, digits2(t >> 32), 2);
      copy_bytes(buf + 6, d->frac100.e[uint32_t(t) >> 25], 2);
      copy_bytes(buf + 8, &mid, 8);
      copy_bytes(buf + 16, &lo, 8);
      uint64_t len = count_digits(v, *d);
      copy_bytes(out, buf + 24 - len, 20);
      return out + len;
#endif  // ZMIJ_USE_SSE4_1
    }
  }
}

namespace {

#if ZMIJ_USE_INT128
// gcc and clang need some handholding.  The combination of ZMIJ_NOINLINE here,
// ZMIJ_UNLIKELY in itoa_signed and ZMIJ_NOINLINE on itoa_body32_pad turned out
// to be the best compromise with neither compiler regressing >10% on some
// benchmarks.
#if ZMIJ_USE_SSE && !ZMIJ_USE_SSE4_1
ZMIJ_NOINLINE
#else
ZMIJ_INLINE
#endif
auto itoa_u128_wide(char* out, uint128_t value) noexcept -> char* {
  const auto* d = &static_data;
  ZMIJ_ASM(("" : "+r"(d)));  // Load constants from memory.
  if (!ZMIJ_USE_SSE && !ZMIJ_USE_NEON) {
    // Mirrors the SIMD paths, but we have to move in 8-digit blocks.
    bool big = value >= uint128_t(uint64_t(1e16)) * uint64_t(1e16);  // >= 1e32
    divmod_1e16_result lo = divmod_1e16(value, *d);  // lo.rem = low 16 digits
    uint64_t low_hi = to_bcd8(uint32_t(lo.rem / 100'000'000ull)).bcd + zeros;
    uint64_t low_lo = to_bcd8(uint32_t(lo.rem % 100'000'000ull)).bcd + zeros;
    // No zero-init: unwritten bytes are only ever copied into the scratch
    // region past out + len that the buffer contract already permits.
    // Digit groups end at buf + 40, so the read window buf + 40 - len starts
    // at buf + 8 at the lowest (len <= 32) and buf + 1 in the 33-39 case.
    char buf[64];
    copy_bytes(buf + 24, &low_hi, 8);
    copy_bytes(buf + 32, &low_lo, 8);
    if (!big) {  // 19-32 digits: top (<=16) + low 16
      uint64_t top = uint64_t(lo.quot);
      uint64_t top_hi = to_bcd8(uint32_t(top / 100'000'000ull)).bcd + zeros;
      uint64_t top_lo = to_bcd8(uint32_t(top % 100'000'000ull)).bcd + zeros;
      copy_bytes(buf + 8, &top_hi, 8);
      copy_bytes(buf + 16, &top_lo, 8);
      uint64_t len = 16 + count_digits(top, *d);
      copy_bytes(out, buf + 40 - len, 32);
      return out + len;
    }
    // 33-39 digits: top (<=7) + mid 16 + low 16.
    divmod_1e16_narrow_result hi = divmod_1e16_narrow(lo.quot, *d);
    uint32_t top = hi.quot;
    uint64_t mid = hi.rem;
    // top is <= 7 digits, so one to_bcd8 covers it.
    uint64_t top_8 = to_bcd8(top).bcd + zeros;
    uint64_t mid_hi = to_bcd8(uint32_t(mid / 100'000'000ull)).bcd + zeros;
    uint64_t mid_lo = to_bcd8(uint32_t(mid % 100'000'000ull)).bcd + zeros;
    copy_bytes(buf, &top_8, 8);
    copy_bytes(buf + 8, &mid_hi, 8);
    copy_bytes(buf + 16, &mid_lo, 8);
    uint64_t len = 32 + count_digits(top, *d);
    copy_bytes(out, buf + 40 - len, 40);
    return out + len;
  }
#if ZMIJ_USE_SSE || ZMIJ_USE_NEON
  divmod_1e16_result lo = divmod_1e16(value, *d);  // lo.rem = digits [0, 16)
#if ZMIJ_USE_NEON
  // Same 1e16 the divmod just loaded, so the compare reuses the register.
  uint64_t ten16 = d->u128_consts[3];
#else
  uint64_t ten16 = uint64_t(1e16);
#endif
  if (lo.quot < uint128_t(ten16)) {  // 19-32 digits: top (<=16) + low
    uint64_t q = uint64_t(lo.quot);
#if ZMIJ_USE_AVX2
    // Fuse the trimmed head (q) and the fixed low chunk into one 256-bit pass,
    // then store the two lanes at out and out + len(q) (offset concatenation,
    // no lane-crossing shuffle).
    return itoa_body_head16_pad(out, q, count_digits_lt_1e16(q, *d), lo.rem,
                                *d);
#else
    char* p = itoa_body(out, q, count_digits_lt_1e16(q, *d), *d);  // q < 1e16
    itoa_body16_pad(p, lo.rem, *d);
    return p + 16;
#endif
  }
  // hi.rem = digits [16, 32); hi.quot = top (<= 7 digits)
  divmod_1e16_narrow_result hi = divmod_1e16_narrow(lo.quot, *d);
#if ZMIJ_USE_FMA
  char* p = itoa_top8(out, hi.quot, count_digits(hi.quot, *d), *d);
#elif ZMIJ_USE_SSE && !ZMIJ_USE_SSE4_1
  char* p = itoa_head7(out, hi.quot, count_digits(hi.quot, *d));
#else
  char* p = itoa_body(out, hi.quot, count_digits(hi.quot, *d), *d);
#endif
  itoa_body32_pad(p, hi.rem, lo.rem, *d);
  return p + 32;
#endif
}
#endif  // ZMIJ_USE_INT128

// std::make_unsigned is ill-formed for __int128 in strict-conformance mode
// (it's not a standard integer type), so map it here. The specialization sits
// behind ZMIJ_USE_INT128, which is 0 on toolchains without __int128 (e.g.
// MSVC), so the primary template handles 32/64-bit there.
template <typename Int>
struct itoa_make_unsigned {
  using type = typename std::make_unsigned<Int>::type;
};
#if ZMIJ_USE_INT128
template <>
struct itoa_make_unsigned<__int128> {
  using type = unsigned __int128;
};
#endif

}  // namespace

// Write the decimal representation of signed value.
template <typename Int>
ZMIJ_INLINE auto itoa_signed(char* out, Int value) noexcept -> char* {
  check_room(out, sizeof(Int) <= 4   ? int32_buffer_size
                  : sizeof(Int) <= 8 ? int64_buffer_size
                                     : int128_buffer_size);
#if ZMIJ_USE_INT128
  // If possible, use the 64bit path.  The 128x128 multiplications required for
  // the full width are more expensive than branching.
  if (sizeof(Int) > 8) {
    if (value == Int(int64_t(value))) [[ZMIJ_LIKELY]] {
      return itoa_signed(out, int64_t(value));
    } else [[ZMIJ_UNLIKELY]] {
      using UInt = typename itoa_make_unsigned<Int>::type;
      UInt mag = value < 0 ? -UInt(value) : UInt(value);
      *out = '-';
      out += value < 0;
      return itoa_u128_wide(out, mag);
    }
  }
#endif
  using UInt = typename itoa_make_unsigned<Int>::type;
#if ZMIJ_USE_NEON
  if (sizeof(Int) <= 4) {
    const auto* d = &static_data;
    ZMIJ_ASM(("" : "+r"(d)));  // Load constants from memory.
    return itoa_i32(out, int32_t(value), *d);
  }
#endif
  UInt mag = value >= 0 ? UInt(value) : -UInt(value);
  *out = '-';
  out += value < 0;
  return itoa(out, mag);
}

namespace {

#if ZMIJ_USE_INT128
// A u128 holds 20-39 digits. Values <= u64 delegate to the u64 path (realistic
// data is overwhelmingly small, so this branch is well predicted). Otherwise
// peel 16 decimal digits at a time with a 128-bit reciprocal: one peel leaves a
// 19-32 digit value (top <=16 + low 16); two peels leave 33-39 (top <=7 +
// mid 16 + low 16). The two interior chunks are fixed-width; only the top trims
// leading zeros.
ZMIJ_INLINE auto itoa_u128(char* out, uint128_t value) noexcept -> char* {
  if (value <= UINT64_MAX) [[ZMIJ_LIKELY]] return itoa(out, uint64_t(value));
  return itoa_u128_wide(out, value);
}
#endif

}  // namespace

template auto itoa(char* out, uint32_t value) noexcept -> char*;
template auto itoa(char* out, uint64_t value) noexcept -> char*;
template auto itoa_signed(char* out, int32_t value) noexcept -> char*;
template auto itoa_signed(char* out, int64_t value) noexcept -> char*;
#if ZMIJ_USE_INT128
template auto itoa(char* out, uint128_t value) noexcept -> char*;
template auto itoa_signed(char* out, __int128 value) noexcept -> char*;
#endif

}  // namespace details_int
}  // namespace zmij
