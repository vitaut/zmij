// A double-to-string conversion algorithm based on Schubfach and yy.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.
// https://github.com/vitaut/zmij/

#if __has_include("zmij-c.h")
#  include "zmij-c.h"
#endif

#include <assert.h>   // assert
#include <float.h>    // DBL_MANT_DIG
#include <stdbool.h>  // bool
#include <stddef.h>   // size_t
#include <stdint.h>   // uint64_t
#include <string.h>   // memcpy

#ifndef ZMIJ_USE_SIMD
#  if defined(__cplusplus)
#    define ZMIJ_USE_SIMD 1
#  else
#    define ZMIJ_USE_SIMD 0
#  endif
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
#  define ZMIJ_MAYBE_UNUSED [[maybe_unused]]
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

#if defined(ZMIJ_ALIGNAS)
// Use the provided definition
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define ZMIJ_ALIGNAS(x) _Alignas(x)
#elif ZMIJ_MSC_VER
#  define ZMIJ_ALIGNAS(x) __declspec(align(x))
#elif __GNUC__
#  define ZMIJ_ALIGNAS(x) __attribute__((aligned(x)))
#else
#  define ZMIJ_ALIGNAS(x)
#endif

static inline bool is_big_endian() {
  int n = 1;
  return *(char*)(&n) != 1;
}

static inline uint64_t bswap64(uint64_t x) {
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

static inline int clz(uint64_t x) {
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

typedef struct {
  uint64_t hi;
  uint64_t lo;
} uint128;

ZMIJ_MAYBE_UNUSED static inline uint64_t uint128_to_uint64(uint128 u) {
  return u.lo;
}

ZMIJ_MAYBE_UNUSED static inline uint128 uint128_add(uint128 lhs, uint128 rhs) {
#ifdef _M_AMD64
  uint64_t lo, hi;
  _addcarry_u64(_addcarry_u64(0, lhs.lo, rhs.lo, &lo), lhs.hi, rhs.hi, &hi);
  uint128 result = {hi, lo};
  return result;
#else
  uint64_t lo = lhs.lo + rhs.lo;
  uint128 result = {lhs.hi + rhs.hi + (lo < lhs.lo), lo};
  return result;
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
typedef unsigned __int128 uint128_t;
#else
typedef uint128 uint128_t;
#endif  // ZMIJ_USE_INT128

ZMIJ_MAYBE_UNUSED static inline uint128_t uint128_rshift(uint128_t u,
                                                         int shift) {
#if ZMIJ_USE_INT128
  return u >> shift;
#else
  if (shift == 32) {
    uint64_t hilo = (uint32_t)u.hi;
    uint128 result = {u.hi >> 32, (hilo << 32) | (u.lo >> 32)};
    return result;
  }
  assert(shift >= 64 && shift < 128);
  uint128 result = {0, u.hi >> (shift - 64)};
  return result;
#endif
}
#if ZMIJ_USE_INT128 && defined(__APPLE__)
static const bool use_umul128_hi64 = true;  // Use umul128_hi64 for division.
#else
static const bool use_umul128_hi64 = false;
#endif

// Computes 128-bit result of multiplication of two 64-bit unsigned integers.
static uint128_t umul128(uint64_t x, uint64_t y) {
#if ZMIJ_USE_INT128
  return (uint128_t)x * y;
#else
#  if defined(_M_AMD64)
  {
    uint64_t hi = 0;
    uint64_t lo = _umul128(x, y, &hi);
    uint128 result = {hi, lo};
    return result;
  }
#  elif defined(_M_ARM64)
  {
    uint128 result = {__umulh(x, y), x * y};
    return result;
  }
#  endif
  uint64_t a = x >> 32;
  uint64_t b = (uint32_t)x;
  uint64_t c = y >> 32;
  uint64_t d = (uint32_t)y;

  uint64_t ac = a * c;
  uint64_t bc = b * c;
  uint64_t ad = a * d;
  uint64_t bd = b * d;

  uint64_t cs = (bd >> 32) + (uint32_t)ad + (uint32_t)bc;  // cross sum
  uint128 result = {ac + (ad >> 32) + (bc >> 32) + (cs >> 32),
                    (cs << 32) + (uint32_t)bd};
  return result;
#endif  // ZMIJ_USE_INT128
}

static uint64_t umul128_hi64(uint64_t x, uint64_t y) {
#if ZMIJ_USE_INT128
  return (uint64_t)(umul128(x, y) >> 64);
#else
  return umul128(x, y).hi;
#endif
}

static uint64_t lo64(uint128_t v) {
#if ZMIJ_USE_INT128
  return (uint64_t)v;
#else
  return v.lo;
#endif
}
static uint64_t hi64(uint128_t v) {
#if ZMIJ_USE_INT128
  return v >> 64;
#else
  return v.hi;
#endif
}

static inline uint128 umul192_hi128(uint64_t x_hi, uint64_t x_lo, uint64_t y) {
  uint128_t p = umul128(x_hi, y);
  uint64_t lo = lo64(p) + umul128_hi64(x_lo, y);
  uint128 result = {hi64(p) + (lo < lo64(p)), lo};
  return result;
}

// Computes high 64 bits of multiplication of x and y, discards the least
// significant bit and rounds to odd, where x = uint128_t(x_hi << 64) | x_lo.
uint64_t umulhi_inexact_to_odd64(uint64_t x_hi, uint64_t x_lo, uint64_t y) {
  uint128 p = umul192_hi128(x_hi, x_lo, y);
  return p.hi | ((p.lo >> 1) != 0);
}
uint32_t umulhi_inexact_to_odd32(uint64_t x_hi, uint64_t _, uint32_t y) {
  uint64_t p = lo64(uint128_rshift(umul128(x_hi, y), 32));
  return (uint32_t)(p >> 32) | (((uint32_t)p >> 1) != 0);
}

enum {
  double_num_bits = 64,
  double_num_sig_bits = DBL_MANT_DIG - 1,
  double_num_exp_bits = double_num_bits - double_num_sig_bits - 1,
  double_exp_mask = (1 << double_num_exp_bits) - 1,
  double_exp_bias = (1 << (double_num_exp_bits - 1)) - 1,
  double_exp_offset = double_exp_bias + double_num_sig_bits,
};

typedef uint64_t double_sig_type;
static const double_sig_type double_implicit_bit = (double_sig_type)1
                                                   << double_num_sig_bits;

static inline double_sig_type double_to_bits(double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof(value));
  return bits;
}

static inline bool double_is_negative(double_sig_type bits) {
  return bits >> (double_num_bits - 1);
}
static inline double_sig_type double_get_sig(double_sig_type bits) {
  return bits & (double_implicit_bit - 1);
}
static int64_t double_get_exp(double_sig_type bits) {
  return (int64_t)((bits << 1) >> (double_num_sig_bits + 1));
}

enum {
  float_num_bits = 32,
  float_num_sig_bits = FLT_MANT_DIG - 1,
  float_num_exp_bits = float_num_bits - float_num_sig_bits - 1,
  float_exp_mask = (1 << float_num_exp_bits) - 1,
  float_exp_bias = (1 << (float_num_exp_bits - 1)) - 1,
  float_exp_offset = float_exp_bias + float_num_sig_bits,
};

typedef uint32_t float_sig_type;
static const float_sig_type float_implicit_bit = (float_sig_type)1
                                                 << float_num_sig_bits;

static inline float_sig_type float_to_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(value));
  return bits;
}

static inline bool float_is_negative(float_sig_type bits) {
  return bits >> (float_num_bits - 1);
}
static inline float_sig_type float_get_sig(float_sig_type bits) {
  return bits & (float_implicit_bit - 1);
}
static int64_t float_get_exp(float_sig_type bits) {
  return (int64_t)((bits << 1) >> (float_num_sig_bits + 1));
}

// 128-bit significands of powers of 10 rounded down.
ZMIJ_ALIGNAS(64)
const uint128 pow10_significands_data[] = {
    {0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7a},  // -292
    {0x9faacf3df73609b1, 0x77b191618c54e9ac},  // -291
    {0xc795830d75038c1d, 0xd59df5b9ef6a2417},  // -290
    {0xf97ae3d0d2446f25, 0x4b0573286b44ad1d},  // -289
    {0x9becce62836ac577, 0x4ee367f9430aec32},  // -288
    {0xc2e801fb244576d5, 0x229c41f793cda73f},  // -287
    {0xf3a20279ed56d48a, 0x6b43527578c1110f},  // -286
    {0x9845418c345644d6, 0x830a13896b78aaa9},  // -285
    {0xbe5691ef416bd60c, 0x23cc986bc656d553},  // -284
    {0xedec366b11c6cb8f, 0x2cbfbe86b7ec8aa8},  // -283
    {0x94b3a202eb1c3f39, 0x7bf7d71432f3d6a9},  // -282
    {0xb9e08a83a5e34f07, 0xdaf5ccd93fb0cc53},  // -281
    {0xe858ad248f5c22c9, 0xd1b3400f8f9cff68},  // -280
    {0x91376c36d99995be, 0x23100809b9c21fa1},  // -279
    {0xb58547448ffffb2d, 0xabd40a0c2832a78a},  // -278
    {0xe2e69915b3fff9f9, 0x16c90c8f323f516c},  // -277
    {0x8dd01fad907ffc3b, 0xae3da7d97f6792e3},  // -276
    {0xb1442798f49ffb4a, 0x99cd11cfdf41779c},  // -275
    {0xdd95317f31c7fa1d, 0x40405643d711d583},  // -274
    {0x8a7d3eef7f1cfc52, 0x482835ea666b2572},  // -273
    {0xad1c8eab5ee43b66, 0xda3243650005eecf},  // -272
    {0xd863b256369d4a40, 0x90bed43e40076a82},  // -271
    {0x873e4f75e2224e68, 0x5a7744a6e804a291},  // -270
    {0xa90de3535aaae202, 0x711515d0a205cb36},  // -269
    {0xd3515c2831559a83, 0x0d5a5b44ca873e03},  // -268
    {0x8412d9991ed58091, 0xe858790afe9486c2},  // -267
    {0xa5178fff668ae0b6, 0x626e974dbe39a872},  // -266
    {0xce5d73ff402d98e3, 0xfb0a3d212dc8128f},  // -265
    {0x80fa687f881c7f8e, 0x7ce66634bc9d0b99},  // -264
    {0xa139029f6a239f72, 0x1c1fffc1ebc44e80},  // -263
    {0xc987434744ac874e, 0xa327ffb266b56220},  // -262
    {0xfbe9141915d7a922, 0x4bf1ff9f0062baa8},  // -261
    {0x9d71ac8fada6c9b5, 0x6f773fc3603db4a9},  // -260
    {0xc4ce17b399107c22, 0xcb550fb4384d21d3},  // -259
    {0xf6019da07f549b2b, 0x7e2a53a146606a48},  // -258
    {0x99c102844f94e0fb, 0x2eda7444cbfc426d},  // -257
    {0xc0314325637a1939, 0xfa911155fefb5308},  // -256
    {0xf03d93eebc589f88, 0x793555ab7eba27ca},  // -255
    {0x96267c7535b763b5, 0x4bc1558b2f3458de},  // -254
    {0xbbb01b9283253ca2, 0x9eb1aaedfb016f16},  // -253
    {0xea9c227723ee8bcb, 0x465e15a979c1cadc},  // -252
    {0x92a1958a7675175f, 0x0bfacd89ec191ec9},  // -251
    {0xb749faed14125d36, 0xcef980ec671f667b},  // -250
    {0xe51c79a85916f484, 0x82b7e12780e7401a},  // -249
    {0x8f31cc0937ae58d2, 0xd1b2ecb8b0908810},  // -248
    {0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa15},  // -247
    {0xdfbdcece67006ac9, 0x67a791e093e1d49a},  // -246
    {0x8bd6a141006042bd, 0xe0c8bb2c5c6d24e0},  // -245
    {0xaecc49914078536d, 0x58fae9f773886e18},  // -244
    {0xda7f5bf590966848, 0xaf39a475506a899e},  // -243
    {0x888f99797a5e012d, 0x6d8406c952429603},  // -242
    {0xaab37fd7d8f58178, 0xc8e5087ba6d33b83},  // -241
    {0xd5605fcdcf32e1d6, 0xfb1e4a9a90880a64},  // -240
    {0x855c3be0a17fcd26, 0x5cf2eea09a55067f},  // -239
    {0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481e},  // -238
    {0xd0601d8efc57b08b, 0xf13b94daf124da26},  // -237
    {0x823c12795db6ce57, 0x76c53d08d6b70858},  // -236
    {0xa2cb1717b52481ed, 0x54768c4b0c64ca6e},  // -235
    {0xcb7ddcdda26da268, 0xa9942f5dcf7dfd09},  // -234
    {0xfe5d54150b090b02, 0xd3f93b35435d7c4c},  // -233
    {0x9efa548d26e5a6e1, 0xc47bc5014a1a6daf},  // -232
    {0xc6b8e9b0709f109a, 0x359ab6419ca1091b},  // -231
    {0xf867241c8cc6d4c0, 0xc30163d203c94b62},  // -230
    {0x9b407691d7fc44f8, 0x79e0de63425dcf1d},  // -229
    {0xc21094364dfb5636, 0x985915fc12f542e4},  // -228
    {0xf294b943e17a2bc4, 0x3e6f5b7b17b2939d},  // -227
    {0x979cf3ca6cec5b5a, 0xa705992ceecf9c42},  // -226
    {0xbd8430bd08277231, 0x50c6ff782a838353},  // -225
    {0xece53cec4a314ebd, 0xa4f8bf5635246428},  // -224
    {0x940f4613ae5ed136, 0x871b7795e136be99},  // -223
    {0xb913179899f68584, 0x28e2557b59846e3f},  // -222
    {0xe757dd7ec07426e5, 0x331aeada2fe589cf},  // -221
    {0x9096ea6f3848984f, 0x3ff0d2c85def7621},  // -220
    {0xb4bca50b065abe63, 0x0fed077a756b53a9},  // -219
    {0xe1ebce4dc7f16dfb, 0xd3e8495912c62894},  // -218
    {0x8d3360f09cf6e4bd, 0x64712dd7abbbd95c},  // -217
    {0xb080392cc4349dec, 0xbd8d794d96aacfb3},  // -216
    {0xdca04777f541c567, 0xecf0d7a0fc5583a0},  // -215
    {0x89e42caaf9491b60, 0xf41686c49db57244},  // -214
    {0xac5d37d5b79b6239, 0x311c2875c522ced5},  // -213
    {0xd77485cb25823ac7, 0x7d633293366b828b},  // -212
    {0x86a8d39ef77164bc, 0xae5dff9c02033197},  // -211
    {0xa8530886b54dbdeb, 0xd9f57f830283fdfc},  // -210
    {0xd267caa862a12d66, 0xd072df63c324fd7b},  // -209
    {0x8380dea93da4bc60, 0x4247cb9e59f71e6d},  // -208
    {0xa46116538d0deb78, 0x52d9be85f074e608},  // -207
    {0xcd795be870516656, 0x67902e276c921f8b},  // -206
    {0x806bd9714632dff6, 0x00ba1cd8a3db53b6},  // -205
    {0xa086cfcd97bf97f3, 0x80e8a40eccd228a4},  // -204
    {0xc8a883c0fdaf7df0, 0x6122cd128006b2cd},  // -203
    {0xfad2a4b13d1b5d6c, 0x796b805720085f81},  // -202
    {0x9cc3a6eec6311a63, 0xcbe3303674053bb0},  // -201
    {0xc3f490aa77bd60fc, 0xbedbfc4411068a9c},  // -200
    {0xf4f1b4d515acb93b, 0xee92fb5515482d44},  // -199
    {0x991711052d8bf3c5, 0x751bdd152d4d1c4a},  // -198
    {0xbf5cd54678eef0b6, 0xd262d45a78a0635d},  // -197
    {0xef340a98172aace4, 0x86fb897116c87c34},  // -196
    {0x9580869f0e7aac0e, 0xd45d35e6ae3d4da0},  // -195
    {0xbae0a846d2195712, 0x8974836059cca109},  // -194
    {0xe998d258869facd7, 0x2bd1a438703fc94b},  // -193
    {0x91ff83775423cc06, 0x7b6306a34627ddcf},  // -192
    {0xb67f6455292cbf08, 0x1a3bc84c17b1d542},  // -191
    {0xe41f3d6a7377eeca, 0x20caba5f1d9e4a93},  // -190
    {0x8e938662882af53e, 0x547eb47b7282ee9c},  // -189
    {0xb23867fb2a35b28d, 0xe99e619a4f23aa43},  // -188
    {0xdec681f9f4c31f31, 0x6405fa00e2ec94d4},  // -187
    {0x8b3c113c38f9f37e, 0xde83bc408dd3dd04},  // -186
    {0xae0b158b4738705e, 0x9624ab50b148d445},  // -185
    {0xd98ddaee19068c76, 0x3badd624dd9b0957},  // -184
    {0x87f8a8d4cfa417c9, 0xe54ca5d70a80e5d6},  // -183
    {0xa9f6d30a038d1dbc, 0x5e9fcf4ccd211f4c},  // -182
    {0xd47487cc8470652b, 0x7647c3200069671f},  // -181
    {0x84c8d4dfd2c63f3b, 0x29ecd9f40041e073},  // -180
    {0xa5fb0a17c777cf09, 0xf468107100525890},  // -179
    {0xcf79cc9db955c2cc, 0x7182148d4066eeb4},  // -178
    {0x81ac1fe293d599bf, 0xc6f14cd848405530},  // -177
    {0xa21727db38cb002f, 0xb8ada00e5a506a7c},  // -176
    {0xca9cf1d206fdc03b, 0xa6d90811f0e4851c},  // -175
    {0xfd442e4688bd304a, 0x908f4a166d1da663},  // -174
    {0x9e4a9cec15763e2e, 0x9a598e4e043287fe},  // -173
    {0xc5dd44271ad3cdba, 0x40eff1e1853f29fd},  // -172
    {0xf7549530e188c128, 0xd12bee59e68ef47c},  // -171
    {0x9a94dd3e8cf578b9, 0x82bb74f8301958ce},  // -170
    {0xc13a148e3032d6e7, 0xe36a52363c1faf01},  // -169
    {0xf18899b1bc3f8ca1, 0xdc44e6c3cb279ac1},  // -168
    {0x96f5600f15a7b7e5, 0x29ab103a5ef8c0b9},  // -167
    {0xbcb2b812db11a5de, 0x7415d448f6b6f0e7},  // -166
    {0xebdf661791d60f56, 0x111b495b3464ad21},  // -165
    {0x936b9fcebb25c995, 0xcab10dd900beec34},  // -164
    {0xb84687c269ef3bfb, 0x3d5d514f40eea742},  // -163
    {0xe65829b3046b0afa, 0x0cb4a5a3112a5112},  // -162
    {0x8ff71a0fe2c2e6dc, 0x47f0e785eaba72ab},  // -161
    {0xb3f4e093db73a093, 0x59ed216765690f56},  // -160
    {0xe0f218b8d25088b8, 0x306869c13ec3532c},  // -159
    {0x8c974f7383725573, 0x1e414218c73a13fb},  // -158
    {0xafbd2350644eeacf, 0xe5d1929ef90898fa},  // -157
    {0xdbac6c247d62a583, 0xdf45f746b74abf39},  // -156
    {0x894bc396ce5da772, 0x6b8bba8c328eb783},  // -155
    {0xab9eb47c81f5114f, 0x066ea92f3f326564},  // -154
    {0xd686619ba27255a2, 0xc80a537b0efefebd},  // -153
    {0x8613fd0145877585, 0xbd06742ce95f5f36},  // -152
    {0xa798fc4196e952e7, 0x2c48113823b73704},  // -151
    {0xd17f3b51fca3a7a0, 0xf75a15862ca504c5},  // -150
    {0x82ef85133de648c4, 0x9a984d73dbe722fb},  // -149
    {0xa3ab66580d5fdaf5, 0xc13e60d0d2e0ebba},  // -148
    {0xcc963fee10b7d1b3, 0x318df905079926a8},  // -147
    {0xffbbcfe994e5c61f, 0xfdf17746497f7052},  // -146
    {0x9fd561f1fd0f9bd3, 0xfeb6ea8bedefa633},  // -145
    {0xc7caba6e7c5382c8, 0xfe64a52ee96b8fc0},  // -144
    {0xf9bd690a1b68637b, 0x3dfdce7aa3c673b0},  // -143
    {0x9c1661a651213e2d, 0x06bea10ca65c084e},  // -142
    {0xc31bfa0fe5698db8, 0x486e494fcff30a62},  // -141
    {0xf3e2f893dec3f126, 0x5a89dba3c3efccfa},  // -140
    {0x986ddb5c6b3a76b7, 0xf89629465a75e01c},  // -139
    {0xbe89523386091465, 0xf6bbb397f1135823},  // -138
    {0xee2ba6c0678b597f, 0x746aa07ded582e2c},  // -137
    {0x94db483840b717ef, 0xa8c2a44eb4571cdc},  // -136
    {0xba121a4650e4ddeb, 0x92f34d62616ce413},  // -135
    {0xe896a0d7e51e1566, 0x77b020baf9c81d17},  // -134
    {0x915e2486ef32cd60, 0x0ace1474dc1d122e},  // -133
    {0xb5b5ada8aaff80b8, 0x0d819992132456ba},  // -132
    {0xe3231912d5bf60e6, 0x10e1fff697ed6c69},  // -131
    {0x8df5efabc5979c8f, 0xca8d3ffa1ef463c1},  // -130
    {0xb1736b96b6fd83b3, 0xbd308ff8a6b17cb2},  // -129
    {0xddd0467c64bce4a0, 0xac7cb3f6d05ddbde},  // -128
    {0x8aa22c0dbef60ee4, 0x6bcdf07a423aa96b},  // -127
    {0xad4ab7112eb3929d, 0x86c16c98d2c953c6},  // -126
    {0xd89d64d57a607744, 0xe871c7bf077ba8b7},  // -125
    {0x87625f056c7c4a8b, 0x11471cd764ad4972},  // -124
    {0xa93af6c6c79b5d2d, 0xd598e40d3dd89bcf},  // -123
    {0xd389b47879823479, 0x4aff1d108d4ec2c3},  // -122
    {0x843610cb4bf160cb, 0xcedf722a585139ba},  // -121
    {0xa54394fe1eedb8fe, 0xc2974eb4ee658828},  // -120
    {0xce947a3da6a9273e, 0x733d226229feea32},  // -119
    {0x811ccc668829b887, 0x0806357d5a3f525f},  // -118
    {0xa163ff802a3426a8, 0xca07c2dcb0cf26f7},  // -117
    {0xc9bcff6034c13052, 0xfc89b393dd02f0b5},  // -116
    {0xfc2c3f3841f17c67, 0xbbac2078d443ace2},  // -115
    {0x9d9ba7832936edc0, 0xd54b944b84aa4c0d},  // -114
    {0xc5029163f384a931, 0x0a9e795e65d4df11},  // -113
    {0xf64335bcf065d37d, 0x4d4617b5ff4a16d5},  // -112
    {0x99ea0196163fa42e, 0x504bced1bf8e4e45},  // -111
    {0xc06481fb9bcf8d39, 0xe45ec2862f71e1d6},  // -110
    {0xf07da27a82c37088, 0x5d767327bb4e5a4c},  // -109
    {0x964e858c91ba2655, 0x3a6a07f8d510f86f},  // -108
    {0xbbe226efb628afea, 0x890489f70a55368b},  // -107
    {0xeadab0aba3b2dbe5, 0x2b45ac74ccea842e},  // -106
    {0x92c8ae6b464fc96f, 0x3b0b8bc90012929d},  // -105
    {0xb77ada0617e3bbcb, 0x09ce6ebb40173744},  // -104
    {0xe55990879ddcaabd, 0xcc420a6a101d0515},  // -103
    {0x8f57fa54c2a9eab6, 0x9fa946824a12232d},  // -102
    {0xb32df8e9f3546564, 0x47939822dc96abf9},  // -101
    {0xdff9772470297ebd, 0x59787e2b93bc56f7},  // -100
    {0x8bfbea76c619ef36, 0x57eb4edb3c55b65a},  //  -99
    {0xaefae51477a06b03, 0xede622920b6b23f1},  //  -98
    {0xdab99e59958885c4, 0xe95fab368e45eced},  //  -97
    {0x88b402f7fd75539b, 0x11dbcb0218ebb414},  //  -96
    {0xaae103b5fcd2a881, 0xd652bdc29f26a119},  //  -95
    {0xd59944a37c0752a2, 0x4be76d3346f0495f},  //  -94
    {0x857fcae62d8493a5, 0x6f70a4400c562ddb},  //  -93
    {0xa6dfbd9fb8e5b88e, 0xcb4ccd500f6bb952},  //  -92
    {0xd097ad07a71f26b2, 0x7e2000a41346a7a7},  //  -91
    {0x825ecc24c873782f, 0x8ed400668c0c28c8},  //  -90
    {0xa2f67f2dfa90563b, 0x728900802f0f32fa},  //  -89
    {0xcbb41ef979346bca, 0x4f2b40a03ad2ffb9},  //  -88
    {0xfea126b7d78186bc, 0xe2f610c84987bfa8},  //  -87
    {0x9f24b832e6b0f436, 0x0dd9ca7d2df4d7c9},  //  -86
    {0xc6ede63fa05d3143, 0x91503d1c79720dbb},  //  -85
    {0xf8a95fcf88747d94, 0x75a44c6397ce912a},  //  -84
    {0x9b69dbe1b548ce7c, 0xc986afbe3ee11aba},  //  -83
    {0xc24452da229b021b, 0xfbe85badce996168},  //  -82
    {0xf2d56790ab41c2a2, 0xfae27299423fb9c3},  //  -81
    {0x97c560ba6b0919a5, 0xdccd879fc967d41a},  //  -80
    {0xbdb6b8e905cb600f, 0x5400e987bbc1c920},  //  -79
    {0xed246723473e3813, 0x290123e9aab23b68},  //  -78
    {0x9436c0760c86e30b, 0xf9a0b6720aaf6521},  //  -77
    {0xb94470938fa89bce, 0xf808e40e8d5b3e69},  //  -76
    {0xe7958cb87392c2c2, 0xb60b1d1230b20e04},  //  -75
    {0x90bd77f3483bb9b9, 0xb1c6f22b5e6f48c2},  //  -74
    {0xb4ecd5f01a4aa828, 0x1e38aeb6360b1af3},  //  -73
    {0xe2280b6c20dd5232, 0x25c6da63c38de1b0},  //  -72
    {0x8d590723948a535f, 0x579c487e5a38ad0e},  //  -71
    {0xb0af48ec79ace837, 0x2d835a9df0c6d851},  //  -70
    {0xdcdb1b2798182244, 0xf8e431456cf88e65},  //  -69
    {0x8a08f0f8bf0f156b, 0x1b8e9ecb641b58ff},  //  -68
    {0xac8b2d36eed2dac5, 0xe272467e3d222f3f},  //  -67
    {0xd7adf884aa879177, 0x5b0ed81dcc6abb0f},  //  -66
    {0x86ccbb52ea94baea, 0x98e947129fc2b4e9},  //  -65
    {0xa87fea27a539e9a5, 0x3f2398d747b36224},  //  -64
    {0xd29fe4b18e88640e, 0x8eec7f0d19a03aad},  //  -63
    {0x83a3eeeef9153e89, 0x1953cf68300424ac},  //  -62
    {0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd7},  //  -61
    {0xcdb02555653131b6, 0x3792f412cb06794d},  //  -60
    {0x808e17555f3ebf11, 0xe2bbd88bbee40bd0},  //  -59
    {0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec4},  //  -58
    {0xc8de047564d20a8b, 0xf245825a5a445275},  //  -57
    {0xfb158592be068d2e, 0xeed6e2f0f0d56712},  //  -56
    {0x9ced737bb6c4183d, 0x55464dd69685606b},  //  -55
    {0xc428d05aa4751e4c, 0xaa97e14c3c26b886},  //  -54
    {0xf53304714d9265df, 0xd53dd99f4b3066a8},  //  -53
    {0x993fe2c6d07b7fab, 0xe546a8038efe4029},  //  -52
    {0xbf8fdb78849a5f96, 0xde98520472bdd033},  //  -51
    {0xef73d256a5c0f77c, 0x963e66858f6d4440},  //  -50
    {0x95a8637627989aad, 0xdde7001379a44aa8},  //  -49
    {0xbb127c53b17ec159, 0x5560c018580d5d52},  //  -48
    {0xe9d71b689dde71af, 0xaab8f01e6e10b4a6},  //  -47
    {0x9226712162ab070d, 0xcab3961304ca70e8},  //  -46
    {0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d22},  //  -45
    {0xe45c10c42a2b3b05, 0x8cb89a7db77c506a},  //  -44
    {0x8eb98a7a9a5b04e3, 0x77f3608e92adb242},  //  -43
    {0xb267ed1940f1c61c, 0x55f038b237591ed3},  //  -42
    {0xdf01e85f912e37a3, 0x6b6c46dec52f6688},  //  -41
    {0x8b61313bbabce2c6, 0x2323ac4b3b3da015},  //  -40
    {0xae397d8aa96c1b77, 0xabec975e0a0d081a},  //  -39
    {0xd9c7dced53c72255, 0x96e7bd358c904a21},  //  -38
    {0x881cea14545c7575, 0x7e50d64177da2e54},  //  -37
    {0xaa242499697392d2, 0xdde50bd1d5d0b9e9},  //  -36
    {0xd4ad2dbfc3d07787, 0x955e4ec64b44e864},  //  -35
    {0x84ec3c97da624ab4, 0xbd5af13bef0b113e},  //  -34
    {0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58e},  //  -33
    {0xcfb11ead453994ba, 0x67de18eda5814af2},  //  -32
    {0x81ceb32c4b43fcf4, 0x80eacf948770ced7},  //  -31
    {0xa2425ff75e14fc31, 0xa1258379a94d028d},  //  -30
    {0xcad2f7f5359a3b3e, 0x096ee45813a04330},  //  -29
    {0xfd87b5f28300ca0d, 0x8bca9d6e188853fc},  //  -28
    {0x9e74d1b791e07e48, 0x775ea264cf55347d},  //  -27
    {0xc612062576589dda, 0x95364afe032a819d},  //  -26
    {0xf79687aed3eec551, 0x3a83ddbd83f52204},  //  -25
    {0x9abe14cd44753b52, 0xc4926a9672793542},  //  -24
    {0xc16d9a0095928a27, 0x75b7053c0f178293},  //  -23
    {0xf1c90080baf72cb1, 0x5324c68b12dd6338},  //  -22
    {0x971da05074da7bee, 0xd3f6fc16ebca5e03},  //  -21
    {0xbce5086492111aea, 0x88f4bb1ca6bcf584},  //  -20
    {0xec1e4a7db69561a5, 0x2b31e9e3d06c32e5},  //  -19
    {0x9392ee8e921d5d07, 0x3aff322e62439fcf},  //  -18
    {0xb877aa3236a4b449, 0x09befeb9fad487c2},  //  -17
    {0xe69594bec44de15b, 0x4c2ebe687989a9b3},  //  -16
    {0x901d7cf73ab0acd9, 0x0f9d37014bf60a10},  //  -15
    {0xb424dc35095cd80f, 0x538484c19ef38c94},  //  -14
    {0xe12e13424bb40e13, 0x2865a5f206b06fb9},  //  -13
    {0x8cbccc096f5088cb, 0xf93f87b7442e45d3},  //  -12
    {0xafebff0bcb24aafe, 0xf78f69a51539d748},  //  -11
    {0xdbe6fecebdedd5be, 0xb573440e5a884d1b},  //  -10
    {0x89705f4136b4a597, 0x31680a88f8953030},  //   -9
    {0xabcc77118461cefc, 0xfdc20d2b36ba7c3d},  //   -8
    {0xd6bf94d5e57a42bc, 0x3d32907604691b4c},  //   -7
    {0x8637bd05af6c69b5, 0xa63f9a49c2c1b10f},  //   -6
    {0xa7c5ac471b478423, 0x0fcf80dc33721d53},  //   -5
    {0xd1b71758e219652b, 0xd3c36113404ea4a8},  //   -4
    {0x83126e978d4fdf3b, 0x645a1cac083126e9},  //   -3
    {0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a3},  //   -2
    {0xcccccccccccccccc, 0xcccccccccccccccc},  //   -1
    {0x8000000000000000, 0x0000000000000000},  //    0
    {0xa000000000000000, 0x0000000000000000},  //    1
    {0xc800000000000000, 0x0000000000000000},  //    2
    {0xfa00000000000000, 0x0000000000000000},  //    3
    {0x9c40000000000000, 0x0000000000000000},  //    4
    {0xc350000000000000, 0x0000000000000000},  //    5
    {0xf424000000000000, 0x0000000000000000},  //    6
    {0x9896800000000000, 0x0000000000000000},  //    7
    {0xbebc200000000000, 0x0000000000000000},  //    8
    {0xee6b280000000000, 0x0000000000000000},  //    9
    {0x9502f90000000000, 0x0000000000000000},  //   10
    {0xba43b74000000000, 0x0000000000000000},  //   11
    {0xe8d4a51000000000, 0x0000000000000000},  //   12
    {0x9184e72a00000000, 0x0000000000000000},  //   13
    {0xb5e620f480000000, 0x0000000000000000},  //   14
    {0xe35fa931a0000000, 0x0000000000000000},  //   15
    {0x8e1bc9bf04000000, 0x0000000000000000},  //   16
    {0xb1a2bc2ec5000000, 0x0000000000000000},  //   17
    {0xde0b6b3a76400000, 0x0000000000000000},  //   18
    {0x8ac7230489e80000, 0x0000000000000000},  //   19
    {0xad78ebc5ac620000, 0x0000000000000000},  //   20
    {0xd8d726b7177a8000, 0x0000000000000000},  //   21
    {0x878678326eac9000, 0x0000000000000000},  //   22
    {0xa968163f0a57b400, 0x0000000000000000},  //   23
    {0xd3c21bcecceda100, 0x0000000000000000},  //   24
    {0x84595161401484a0, 0x0000000000000000},  //   25
    {0xa56fa5b99019a5c8, 0x0000000000000000},  //   26
    {0xcecb8f27f4200f3a, 0x0000000000000000},  //   27
    {0x813f3978f8940984, 0x4000000000000000},  //   28
    {0xa18f07d736b90be5, 0x5000000000000000},  //   29
    {0xc9f2c9cd04674ede, 0xa400000000000000},  //   30
    {0xfc6f7c4045812296, 0x4d00000000000000},  //   31
    {0x9dc5ada82b70b59d, 0xf020000000000000},  //   32
    {0xc5371912364ce305, 0x6c28000000000000},  //   33
    {0xf684df56c3e01bc6, 0xc732000000000000},  //   34
    {0x9a130b963a6c115c, 0x3c7f400000000000},  //   35
    {0xc097ce7bc90715b3, 0x4b9f100000000000},  //   36
    {0xf0bdc21abb48db20, 0x1e86d40000000000},  //   37
    {0x96769950b50d88f4, 0x1314448000000000},  //   38
    {0xbc143fa4e250eb31, 0x17d955a000000000},  //   39
    {0xeb194f8e1ae525fd, 0x5dcfab0800000000},  //   40
    {0x92efd1b8d0cf37be, 0x5aa1cae500000000},  //   41
    {0xb7abc627050305ad, 0xf14a3d9e40000000},  //   42
    {0xe596b7b0c643c719, 0x6d9ccd05d0000000},  //   43
    {0x8f7e32ce7bea5c6f, 0xe4820023a2000000},  //   44
    {0xb35dbf821ae4f38b, 0xdda2802c8a800000},  //   45
    {0xe0352f62a19e306e, 0xd50b2037ad200000},  //   46
    {0x8c213d9da502de45, 0x4526f422cc340000},  //   47
    {0xaf298d050e4395d6, 0x9670b12b7f410000},  //   48
    {0xdaf3f04651d47b4c, 0x3c0cdd765f114000},  //   49
    {0x88d8762bf324cd0f, 0xa5880a69fb6ac800},  //   50
    {0xab0e93b6efee0053, 0x8eea0d047a457a00},  //   51
    {0xd5d238a4abe98068, 0x72a4904598d6d880},  //   52
    {0x85a36366eb71f041, 0x47a6da2b7f864750},  //   53
    {0xa70c3c40a64e6c51, 0x999090b65f67d924},  //   54
    {0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d},  //   55
    {0x82818f1281ed449f, 0xbff8f10e7a8921a4},  //   56
    {0xa321f2d7226895c7, 0xaff72d52192b6a0d},  //   57
    {0xcbea6f8ceb02bb39, 0x9bf4f8a69f764490},  //   58
    {0xfee50b7025c36a08, 0x02f236d04753d5b4},  //   59
    {0x9f4f2726179a2245, 0x01d762422c946590},  //   60
    {0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef5},  //   61
    {0xf8ebad2b84e0d58b, 0xd2e0898765a7deb2},  //   62
    {0x9b934c3b330c8577, 0x63cc55f49f88eb2f},  //   63
    {0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fb},  //   64
    {0xf316271c7fc3908a, 0x8bef464e3945ef7a},  //   65
    {0x97edd871cfda3a56, 0x97758bf0e3cbb5ac},  //   66
    {0xbde94e8e43d0c8ec, 0x3d52eeed1cbea317},  //   67
    {0xed63a231d4c4fb27, 0x4ca7aaa863ee4bdd},  //   68
    {0x945e455f24fb1cf8, 0x8fe8caa93e74ef6a},  //   69
    {0xb975d6b6ee39e436, 0xb3e2fd538e122b44},  //   70
    {0xe7d34c64a9c85d44, 0x60dbbca87196b616},  //   71
    {0x90e40fbeea1d3a4a, 0xbc8955e946fe31cd},  //   72
    {0xb51d13aea4a488dd, 0x6babab6398bdbe41},  //   73
    {0xe264589a4dcdab14, 0xc696963c7eed2dd1},  //   74
    {0x8d7eb76070a08aec, 0xfc1e1de5cf543ca2},  //   75
    {0xb0de65388cc8ada8, 0x3b25a55f43294bcb},  //   76
    {0xdd15fe86affad912, 0x49ef0eb713f39ebe},  //   77
    {0x8a2dbf142dfcc7ab, 0x6e3569326c784337},  //   78
    {0xacb92ed9397bf996, 0x49c2c37f07965404},  //   79
    {0xd7e77a8f87daf7fb, 0xdc33745ec97be906},  //   80
    {0x86f0ac99b4e8dafd, 0x69a028bb3ded71a3},  //   81
    {0xa8acd7c0222311bc, 0xc40832ea0d68ce0c},  //   82
    {0xd2d80db02aabd62b, 0xf50a3fa490c30190},  //   83
    {0x83c7088e1aab65db, 0x792667c6da79e0fa},  //   84
    {0xa4b8cab1a1563f52, 0x577001b891185938},  //   85
    {0xcde6fd5e09abcf26, 0xed4c0226b55e6f86},  //   86
    {0x80b05e5ac60b6178, 0x544f8158315b05b4},  //   87
    {0xa0dc75f1778e39d6, 0x696361ae3db1c721},  //   88
    {0xc913936dd571c84c, 0x03bc3a19cd1e38e9},  //   89
    {0xfb5878494ace3a5f, 0x04ab48a04065c723},  //   90
    {0x9d174b2dcec0e47b, 0x62eb0d64283f9c76},  //   91
    {0xc45d1df942711d9a, 0x3ba5d0bd324f8394},  //   92
    {0xf5746577930d6500, 0xca8f44ec7ee36479},  //   93
    {0x9968bf6abbe85f20, 0x7e998b13cf4e1ecb},  //   94
    {0xbfc2ef456ae276e8, 0x9e3fedd8c321a67e},  //   95
    {0xefb3ab16c59b14a2, 0xc5cfe94ef3ea101e},  //   96
    {0x95d04aee3b80ece5, 0xbba1f1d158724a12},  //   97
    {0xbb445da9ca61281f, 0x2a8a6e45ae8edc97},  //   98
    {0xea1575143cf97226, 0xf52d09d71a3293bd},  //   99
    {0x924d692ca61be758, 0x593c2626705f9c56},  //  100
    {0xb6e0c377cfa2e12e, 0x6f8b2fb00c77836c},  //  101
    {0xe498f455c38b997a, 0x0b6dfb9c0f956447},  //  102
    {0x8edf98b59a373fec, 0x4724bd4189bd5eac},  //  103
    {0xb2977ee300c50fe7, 0x58edec91ec2cb657},  //  104
    {0xdf3d5e9bc0f653e1, 0x2f2967b66737e3ed},  //  105
    {0x8b865b215899f46c, 0xbd79e0d20082ee74},  //  106
    {0xae67f1e9aec07187, 0xecd8590680a3aa11},  //  107
    {0xda01ee641a708de9, 0xe80e6f4820cc9495},  //  108
    {0x884134fe908658b2, 0x3109058d147fdcdd},  //  109
    {0xaa51823e34a7eede, 0xbd4b46f0599fd415},  //  110
    {0xd4e5e2cdc1d1ea96, 0x6c9e18ac7007c91a},  //  111
    {0x850fadc09923329e, 0x03e2cf6bc604ddb0},  //  112
    {0xa6539930bf6bff45, 0x84db8346b786151c},  //  113
    {0xcfe87f7cef46ff16, 0xe612641865679a63},  //  114
    {0x81f14fae158c5f6e, 0x4fcb7e8f3f60c07e},  //  115
    {0xa26da3999aef7749, 0xe3be5e330f38f09d},  //  116
    {0xcb090c8001ab551c, 0x5cadf5bfd3072cc5},  //  117
    {0xfdcb4fa002162a63, 0x73d9732fc7c8f7f6},  //  118
    {0x9e9f11c4014dda7e, 0x2867e7fddcdd9afa},  //  119
    {0xc646d63501a1511d, 0xb281e1fd541501b8},  //  120
    {0xf7d88bc24209a565, 0x1f225a7ca91a4226},  //  121
    {0x9ae757596946075f, 0x3375788de9b06958},  //  122
    {0xc1a12d2fc3978937, 0x0052d6b1641c83ae},  //  123
    {0xf209787bb47d6b84, 0xc0678c5dbd23a49a},  //  124
    {0x9745eb4d50ce6332, 0xf840b7ba963646e0},  //  125
    {0xbd176620a501fbff, 0xb650e5a93bc3d898},  //  126
    {0xec5d3fa8ce427aff, 0xa3e51f138ab4cebe},  //  127
    {0x93ba47c980e98cdf, 0xc66f336c36b10137},  //  128
    {0xb8a8d9bbe123f017, 0xb80b0047445d4184},  //  129
    {0xe6d3102ad96cec1d, 0xa60dc059157491e5},  //  130
    {0x9043ea1ac7e41392, 0x87c89837ad68db2f},  //  131
    {0xb454e4a179dd1877, 0x29babe4598c311fb},  //  132
    {0xe16a1dc9d8545e94, 0xf4296dd6fef3d67a},  //  133
    {0x8ce2529e2734bb1d, 0x1899e4a65f58660c},  //  134
    {0xb01ae745b101e9e4, 0x5ec05dcff72e7f8f},  //  135
    {0xdc21a1171d42645d, 0x76707543f4fa1f73},  //  136
    {0x899504ae72497eba, 0x6a06494a791c53a8},  //  137
    {0xabfa45da0edbde69, 0x0487db9d17636892},  //  138
    {0xd6f8d7509292d603, 0x45a9d2845d3c42b6},  //  139
    {0x865b86925b9bc5c2, 0x0b8a2392ba45a9b2},  //  140
    {0xa7f26836f282b732, 0x8e6cac7768d7141e},  //  141
    {0xd1ef0244af2364ff, 0x3207d795430cd926},  //  142
    {0x8335616aed761f1f, 0x7f44e6bd49e807b8},  //  143
    {0xa402b9c5a8d3a6e7, 0x5f16206c9c6209a6},  //  144
    {0xcd036837130890a1, 0x36dba887c37a8c0f},  //  145
    {0x802221226be55a64, 0xc2494954da2c9789},  //  146
    {0xa02aa96b06deb0fd, 0xf2db9baa10b7bd6c},  //  147
    {0xc83553c5c8965d3d, 0x6f92829494e5acc7},  //  148
    {0xfa42a8b73abbf48c, 0xcb772339ba1f17f9},  //  149
    {0x9c69a97284b578d7, 0xff2a760414536efb},  //  150
    {0xc38413cf25e2d70d, 0xfef5138519684aba},  //  151
    {0xf46518c2ef5b8cd1, 0x7eb258665fc25d69},  //  152
    {0x98bf2f79d5993802, 0xef2f773ffbd97a61},  //  153
    {0xbeeefb584aff8603, 0xaafb550ffacfd8fa},  //  154
    {0xeeaaba2e5dbf6784, 0x95ba2a53f983cf38},  //  155
    {0x952ab45cfa97a0b2, 0xdd945a747bf26183},  //  156
    {0xba756174393d88df, 0x94f971119aeef9e4},  //  157
    {0xe912b9d1478ceb17, 0x7a37cd5601aab85d},  //  158
    {0x91abb422ccb812ee, 0xac62e055c10ab33a},  //  159
    {0xb616a12b7fe617aa, 0x577b986b314d6009},  //  160
    {0xe39c49765fdf9d94, 0xed5a7e85fda0b80b},  //  161
    {0x8e41ade9fbebc27d, 0x14588f13be847307},  //  162
    {0xb1d219647ae6b31c, 0x596eb2d8ae258fc8},  //  163
    {0xde469fbd99a05fe3, 0x6fca5f8ed9aef3bb},  //  164
    {0x8aec23d680043bee, 0x25de7bb9480d5854},  //  165
    {0xada72ccc20054ae9, 0xaf561aa79a10ae6a},  //  166
    {0xd910f7ff28069da4, 0x1b2ba1518094da04},  //  167
    {0x87aa9aff79042286, 0x90fb44d2f05d0842},  //  168
    {0xa99541bf57452b28, 0x353a1607ac744a53},  //  169
    {0xd3fa922f2d1675f2, 0x42889b8997915ce8},  //  170
    {0x847c9b5d7c2e09b7, 0x69956135febada11},  //  171
    {0xa59bc234db398c25, 0x43fab9837e699095},  //  172
    {0xcf02b2c21207ef2e, 0x94f967e45e03f4bb},  //  173
    {0x8161afb94b44f57d, 0x1d1be0eebac278f5},  //  174
    {0xa1ba1ba79e1632dc, 0x6462d92a69731732},  //  175
    {0xca28a291859bbf93, 0x7d7b8f7503cfdcfe},  //  176
    {0xfcb2cb35e702af78, 0x5cda735244c3d43e},  //  177
    {0x9defbf01b061adab, 0x3a0888136afa64a7},  //  178
    {0xc56baec21c7a1916, 0x088aaa1845b8fdd0},  //  179
    {0xf6c69a72a3989f5b, 0x8aad549e57273d45},  //  180
    {0x9a3c2087a63f6399, 0x36ac54e2f678864b},  //  181
    {0xc0cb28a98fcf3c7f, 0x84576a1bb416a7dd},  //  182
    {0xf0fdf2d3f3c30b9f, 0x656d44a2a11c51d5},  //  183
    {0x969eb7c47859e743, 0x9f644ae5a4b1b325},  //  184
    {0xbc4665b596706114, 0x873d5d9f0dde1fee},  //  185
    {0xeb57ff22fc0c7959, 0xa90cb506d155a7ea},  //  186
    {0x9316ff75dd87cbd8, 0x09a7f12442d588f2},  //  187
    {0xb7dcbf5354e9bece, 0x0c11ed6d538aeb2f},  //  188
    {0xe5d3ef282a242e81, 0x8f1668c8a86da5fa},  //  189
    {0x8fa475791a569d10, 0xf96e017d694487bc},  //  190
    {0xb38d92d760ec4455, 0x37c981dcc395a9ac},  //  191
    {0xe070f78d3927556a, 0x85bbe253f47b1417},  //  192
    {0x8c469ab843b89562, 0x93956d7478ccec8e},  //  193
    {0xaf58416654a6babb, 0x387ac8d1970027b2},  //  194
    {0xdb2e51bfe9d0696a, 0x06997b05fcc0319e},  //  195
    {0x88fcf317f22241e2, 0x441fece3bdf81f03},  //  196
    {0xab3c2fddeeaad25a, 0xd527e81cad7626c3},  //  197
    {0xd60b3bd56a5586f1, 0x8a71e223d8d3b074},  //  198
    {0x85c7056562757456, 0xf6872d5667844e49},  //  199
    {0xa738c6bebb12d16c, 0xb428f8ac016561db},  //  200
    {0xd106f86e69d785c7, 0xe13336d701beba52},  //  201
    {0x82a45b450226b39c, 0xecc0024661173473},  //  202
    {0xa34d721642b06084, 0x27f002d7f95d0190},  //  203
    {0xcc20ce9bd35c78a5, 0x31ec038df7b441f4},  //  204
    {0xff290242c83396ce, 0x7e67047175a15271},  //  205
    {0x9f79a169bd203e41, 0x0f0062c6e984d386},  //  206
    {0xc75809c42c684dd1, 0x52c07b78a3e60868},  //  207
    {0xf92e0c3537826145, 0xa7709a56ccdf8a82},  //  208
    {0x9bbcc7a142b17ccb, 0x88a66076400bb691},  //  209
    {0xc2abf989935ddbfe, 0x6acff893d00ea435},  //  210
    {0xf356f7ebf83552fe, 0x0583f6b8c4124d43},  //  211
    {0x98165af37b2153de, 0xc3727a337a8b704a},  //  212
    {0xbe1bf1b059e9a8d6, 0x744f18c0592e4c5c},  //  213
    {0xeda2ee1c7064130c, 0x1162def06f79df73},  //  214
    {0x9485d4d1c63e8be7, 0x8addcb5645ac2ba8},  //  215
    {0xb9a74a0637ce2ee1, 0x6d953e2bd7173692},  //  216
    {0xe8111c87c5c1ba99, 0xc8fa8db6ccdd0437},  //  217
    {0x910ab1d4db9914a0, 0x1d9c9892400a22a2},  //  218
    {0xb54d5e4a127f59c8, 0x2503beb6d00cab4b},  //  219
    {0xe2a0b5dc971f303a, 0x2e44ae64840fd61d},  //  220
    {0x8da471a9de737e24, 0x5ceaecfed289e5d2},  //  221
    {0xb10d8e1456105dad, 0x7425a83e872c5f47},  //  222
    {0xdd50f1996b947518, 0xd12f124e28f77719},  //  223
    {0x8a5296ffe33cc92f, 0x82bd6b70d99aaa6f},  //  224
    {0xace73cbfdc0bfb7b, 0x636cc64d1001550b},  //  225
    {0xd8210befd30efa5a, 0x3c47f7e05401aa4e},  //  226
    {0x8714a775e3e95c78, 0x65acfaec34810a71},  //  227
    {0xa8d9d1535ce3b396, 0x7f1839a741a14d0d},  //  228
    {0xd31045a8341ca07c, 0x1ede48111209a050},  //  229
    {0x83ea2b892091e44d, 0x934aed0aab460432},  //  230
    {0xa4e4b66b68b65d60, 0xf81da84d5617853f},  //  231
    {0xce1de40642e3f4b9, 0x36251260ab9d668e},  //  232
    {0x80d2ae83e9ce78f3, 0xc1d72b7c6b426019},  //  233
    {0xa1075a24e4421730, 0xb24cf65b8612f81f},  //  234
    {0xc94930ae1d529cfc, 0xdee033f26797b627},  //  235
    {0xfb9b7cd9a4a7443c, 0x169840ef017da3b1},  //  236
    {0x9d412e0806e88aa5, 0x8e1f289560ee864e},  //  237
    {0xc491798a08a2ad4e, 0xf1a6f2bab92a27e2},  //  238
    {0xf5b5d7ec8acb58a2, 0xae10af696774b1db},  //  239
    {0x9991a6f3d6bf1765, 0xacca6da1e0a8ef29},  //  240
    {0xbff610b0cc6edd3f, 0x17fd090a58d32af3},  //  241
    {0xeff394dcff8a948e, 0xddfc4b4cef07f5b0},  //  242
    {0x95f83d0a1fb69cd9, 0x4abdaf101564f98e},  //  243
    {0xbb764c4ca7a4440f, 0x9d6d1ad41abe37f1},  //  244
    {0xea53df5fd18d5513, 0x84c86189216dc5ed},  //  245
    {0x92746b9be2f8552c, 0x32fd3cf5b4e49bb4},  //  246
    {0xb7118682dbb66a77, 0x3fbc8c33221dc2a1},  //  247
    {0xe4d5e82392a40515, 0x0fabaf3feaa5334a},  //  248
    {0x8f05b1163ba6832d, 0x29cb4d87f2a7400e},  //  249
    {0xb2c71d5bca9023f8, 0x743e20e9ef511012},  //  250
    {0xdf78e4b2bd342cf6, 0x914da9246b255416},  //  251
    {0x8bab8eefb6409c1a, 0x1ad089b6c2f7548e},  //  252
    {0xae9672aba3d0c320, 0xa184ac2473b529b1},  //  253
    {0xda3c0f568cc4f3e8, 0xc9e5d72d90a2741e},  //  254
    {0x8865899617fb1871, 0x7e2fa67c7a658892},  //  255
    {0xaa7eebfb9df9de8d, 0xddbb901b98feeab7},  //  256
    {0xd51ea6fa85785631, 0x552a74227f3ea565},  //  257
    {0x8533285c936b35de, 0xd53a88958f87275f},  //  258
    {0xa67ff273b8460356, 0x8a892abaf368f137},  //  259
    {0xd01fef10a657842c, 0x2d2b7569b0432d85},  //  260
    {0x8213f56a67f6b29b, 0x9c3b29620e29fc73},  //  261
    {0xa298f2c501f45f42, 0x8349f3ba91b47b8f},  //  262
    {0xcb3f2f7642717713, 0x241c70a936219a73},  //  263
    {0xfe0efb53d30dd4d7, 0xed238cd383aa0110},  //  264
    {0x9ec95d1463e8a506, 0xf4363804324a40aa},  //  265
    {0xc67bb4597ce2ce48, 0xb143c6053edcd0d5},  //  266
    {0xf81aa16fdc1b81da, 0xdd94b7868e94050a},  //  267
    {0x9b10a4e5e9913128, 0xca7cf2b4191c8326},  //  268
    {0xc1d4ce1f63f57d72, 0xfd1c2f611f63a3f0},  //  269
    {0xf24a01a73cf2dccf, 0xbc633b39673c8cec},  //  270
    {0x976e41088617ca01, 0xd5be0503e085d813},  //  271
    {0xbd49d14aa79dbc82, 0x4b2d8644d8a74e18},  //  272
    {0xec9c459d51852ba2, 0xddf8e7d60ed1219e},  //  273
    {0x93e1ab8252f33b45, 0xcabb90e5c942b503},  //  274
    {0xb8da1662e7b00a17, 0x3d6a751f3b936243},  //  275
    {0xe7109bfba19c0c9d, 0x0cc512670a783ad4},  //  276
    {0x906a617d450187e2, 0x27fb2b80668b24c5},  //  277
    {0xb484f9dc9641e9da, 0xb1f9f660802dedf6},  //  278
    {0xe1a63853bbd26451, 0x5e7873f8a0396973},  //  279
    {0x8d07e33455637eb2, 0xdb0b487b6423e1e8},  //  280
    {0xb049dc016abc5e5f, 0x91ce1a9a3d2cda62},  //  281
    {0xdc5c5301c56b75f7, 0x7641a140cc7810fb},  //  282
    {0x89b9b3e11b6329ba, 0xa9e904c87fcb0a9d},  //  283
    {0xac2820d9623bf429, 0x546345fa9fbdcd44},  //  284
    {0xd732290fbacaf133, 0xa97c177947ad4095},  //  285
    {0x867f59a9d4bed6c0, 0x49ed8eabcccc485d},  //  286
    {0xa81f301449ee8c70, 0x5c68f256bfff5a74},  //  287
    {0xd226fc195c6a2f8c, 0x73832eec6fff3111},  //  288
    {0x83585d8fd9c25db7, 0xc831fd53c5ff7eab},  //  289
    {0xa42e74f3d032f525, 0xba3e7ca8b77f5e55},  //  290
    {0xcd3a1230c43fb26f, 0x28ce1bd2e55f35eb},  //  291
    {0x80444b5e7aa7cf85, 0x7980d163cf5b81b3},  //  292
    {0xa0555e361951c366, 0xd7e105bcc332621f},  //  293
    {0xc86ab5c39fa63440, 0x8dd9472bf3fefaa7},  //  294
    {0xfa856334878fc150, 0xb14f98f6f0feb951},  //  295
    {0x9c935e00d4b9d8d2, 0x6ed1bf9a569f33d3},  //  296
    {0xc3b8358109e84f07, 0x0a862f80ec4700c8},  //  297
    {0xf4a642e14c6262c8, 0xcd27bb612758c0fa},  //  298
    {0x98e7e9cccfbd7dbd, 0x8038d51cb897789c},  //  299
    {0xbf21e44003acdd2c, 0xe0470a63e6bd56c3},  //  300
    {0xeeea5d5004981478, 0x1858ccfce06cac74},  //  301
    {0x95527a5202df0ccb, 0x0f37801e0c43ebc8},  //  302
    {0xbaa718e68396cffd, 0xd30560258f54e6ba},  //  303
    {0xe950df20247c83fd, 0x47c6b82ef32a2069},  //  304
    {0x91d28b7416cdd27e, 0x4cdc331d57fa5441},  //  305
    {0xb6472e511c81471d, 0xe0133fe4adf8e952},  //  306
    {0xe3d8f9e563a198e5, 0x58180fddd97723a6},  //  307
    {0x8e679c2f5e44ff8f, 0x570f09eaa7ea7648},  //  308
    {0xb201833b35d63f73, 0x2cd2cc6551e513da},  //  309
    {0xde81e40a034bcf4f, 0xf8077f7ea65e58d1},  //  310
    {0x8b112e86420f6191, 0xfb04afaf27faf782},  //  311
    {0xadd57a27d29339f6, 0x79c5db9af1f9b563},  //  312
    {0xd94ad8b1c7380874, 0x18375281ae7822bc},  //  313
    {0x87cec76f1c830548, 0x8f2293910d0b15b5},  //  314
    {0xa9c2794ae3a3c69a, 0xb2eb3875504ddb22},  //  315
    {0xd433179d9c8cb841, 0x5fa60692a46151eb},  //  316
    {0x849feec281d7f328, 0xdbc7c41ba6bcd333},  //  317
    {0xa5c7ea73224deff3, 0x12b9b522906c0800},  //  318
    {0xcf39e50feae16bef, 0xd768226b34870a00},  //  319
    {0x81842f29f2cce375, 0xe6a1158300d46640},  //  320
    {0xa1e53af46f801c53, 0x60495ae3c1097fd0},  //  321
    {0xca5e89b18b602368, 0x385bb19cb14bdfc4},  //  322
    {0xfcf62c1dee382c42, 0x46729e03dd9ed7b5},  //  323
    {0x9e19db92b4e31ba9, 0x6c07a2c26a8346d1},  //  324
};

static uint128 get_pow10_significand(int dec_exp) {
  const int dec_exp_min = -292;
  return pow10_significands_data[dec_exp - dec_exp_min];
}

// Computes the decimal exponent as floor(log10(2**bin_exp)) if regular or
// floor(log10(3/4 * 2**bin_exp)) otherwise, without branching.
static int compute_dec_exp(int bin_exp, bool regular) {
  assert(bin_exp >= -1334 && bin_exp <= 2620);
  // log10_3_over_4_sig = -log10(3/4) * 2**log10_2_exp rounded to a power of 2
  const int log10_3_over_4_sig = 131072;
  // log10_2_sig = round(log10(2) * 2**log10_2_exp)
  const int log10_2_sig = 315653;
  const int log10_2_exp = 20;
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
static ZMIJ_INLINE unsigned char compute_exp_shift(int bin_exp, int dec_exp) {
  assert(dec_exp >= -350 && dec_exp <= 350);
  // log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
  const int log2_pow10_sig = 217707, log2_pow10_exp = 16;
  // pow10_bin_exp = floor(log2(10**-dec_exp))
  int pow10_bin_exp = -dec_exp * log2_pow10_sig >> log2_pow10_exp;
  // pow10 = ((pow10_hi << 64) | pow10_lo) * 2**(pow10_bin_exp - 127)
  return bin_exp + pow10_bin_exp + 1;
}

static inline int count_trailing_nonzeros(uint64_t x) {
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
  return ((size_t)70 - clz((x << 1) | 1)) / 8;  // size_t for native arithmetic
}

// Converts value in the range [0, 100) to a string. GCC generates a bit better
// code when value is pointer-size (https://www.godbolt.org/z/5fEPMT1cc).
static inline const char* digits2(size_t value) {
  // Align data since unaligned access may be slower when crossing a
  // hardware-specific boundary.
  ZMIJ_ALIGNAS(2)
  static const char data[] =
      "0001020304050607080910111213141516171819"
      "2021222324252627282930313233343536373839"
      "4041424344454647484950515253545556575859"
      "6061626364656667686970717273747576777879"
      "8081828384858687888990919293949596979899";
  return &data[value * 2];
}

enum {
  div10k_exp = 40,
};
static const uint32_t div10k_sig = (uint32_t)((1ull << div10k_exp) / 10000 + 1);
static const uint32_t neg10k = (uint32_t)((1ull << 32) - 10000);
enum {
  div100_exp = 19,
};
static const uint32_t div100_sig = (1 << div100_exp) / 100 + 1;
static const uint32_t neg100 = (1 << 16) - 100;
enum {
  div10_exp = 10,
};
static const uint32_t div10_sig = (1 << div10_exp) / 10 + 1;
static const uint32_t neg10 = (1 << 8) - 10;

static const uint64_t zeros = 0x0101010101010101u * '0';

static uint64_t to_bcd8(uint64_t abcdefgh) {
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

static inline char* write_if(char* buffer, uint32_t digit, bool condition) {
  *buffer = (char)('0' + digit);
  return buffer + condition;
}

static inline void write8(char* buffer, uint64_t value) {
  memcpy(buffer, &value, 8);
}

static inline uint64_t read8(char* buffer) {
  uint64_t r;
  memcpy(&r, buffer, 8);
  return r;
}

#if ZMIJ_USE_SSE && !ZMIJ_MSC_VER
using m128i = __m128i;
#else
typedef struct {
  long long data[2];
} m128i;
#endif

static m128i splat64(uint64_t x) {
  m128i result = {(long long)x, (long long)x};
  return result;
}
static m128i splat32(uint32_t x) { return splat64((uint64_t)x << 32 | x); }
static m128i splat16(uint16_t x) { return splat32((uint32_t)x << 16 | x); }
static uint64_t pack8(uint8_t a, uint8_t b, uint8_t c, uint8_t d,  //
                      uint8_t e, uint8_t f, uint8_t g, uint8_t h) {
  typedef uint64_t u64;
  return (u64)h << 56 | (u64)g << 48 | (u64)f << 40 | (u64)e << 32 |
         (u64)d << 24 | (u64)c << 16 | (u64)b << +8 | (u64)a;
}

// Writes a significand consisting of up to 9 decimal digits (7-9 for normals)
// and removes trailing zeros.
static char* write_significand9(char* buffer, uint32_t value, bool has9digits) {
  char* start = buffer;
  buffer = write_if(buffer, value / 100000000, has9digits);
  uint64_t bcd = to_bcd8(value % 100000000);
  write8(buffer, bcd | zeros);
  buffer += count_trailing_nonzeros(bcd);
  return buffer - (int)(buffer - start == 1);
}

// Writes a significand consisting of up to 17 decimal digits (16-17 for
// normals) and removes trailing zeros.  The significant digits start
// from buffer[1].  buffer[0] may contain '0' after this function if
// the significand has length 16.
static char* write_significand17_no_simd(char* buffer, uint64_t value,
                                         bool has17digits) {
  {
    // Each digit is denoted by a letter so value is abbccddeeffgghhii.
    uint32_t abbccddee = (uint32_t)(value / 100000000);
    uint32_t ffgghhii = (uint32_t)(value % 100000000);
    buffer = write_if(buffer, abbccddee / 100000000, has17digits);
    uint64_t bcd = to_bcd8(abbccddee % 100000000);
    write8(buffer, bcd | zeros);
    if (ffgghhii == 0) {
      write8(buffer + 8, zeros);
      return buffer + count_trailing_nonzeros(bcd);
    }
    bcd = to_bcd8(ffgghhii);
    write8(buffer + 8, bcd | zeros);
    return buffer + 8 + count_trailing_nonzeros(bcd);
  }
}

static char* write_significand17(char* buffer, uint64_t value, bool has17digits,
                                 long long value_div10) {
  if (!ZMIJ_USE_NEON && !ZMIJ_USE_SSE) {
    return write_significand17_no_simd(buffer, value, has17digits);
  }
#if ZMIJ_USE_NEON
  // An optimized version for NEON by Dougall Johnson.
  const int32_t neg10k = -10000 + 0x10000;
#  if ZMIJ_MSC_VER
  typedef int32_t int32x4[4];
  typedef int16_t int16x8[8];
#  else
  typedef int32x4_t int32x4;
  typedef int16x8_t int16x8;
#  endif
  typedef struct {
    uint64_t mul_const;
    uint64_t hundred_million;
    int32x4 multipliers32;
    int16x8 multipliers16;
  } mul_constants;
  static const mul_constants constants = {
      0xabcc77118461cefd,
      100000000,
      {div10k_sig, neg10k, div100_sig << 12, neg100},
      {0xce0, neg10}};
  const mul_constants* c = &constants;

  // Compiler barrier, or clang doesn't load from memory and generates 15 more
  // instructions
  ZMIJ_ASM(("" : "+r"(c)));

  uint64_t hundred_million = c->hundred_million;

  // Compiler barrier, or clang narrows the load to 32-bit and unpairs it.
  ZMIJ_ASM(("" : "+r"(hundred_million)));

  // Equivalent to abbccddee = value / 100000000, ffgghhii = value % 100000000.
  uint64_t abbccddee = lo64(uint128_rshift(umul128(value, c->mul_const), 90));
  uint64_t ffgghhii = value - abbccddee * hundred_million;

  // We could probably make this bit faster, but we're preferring to
  // reuse the constants for now.
  uint64_t a = lo64(uint128_rshift(umul128(abbccddee, c->mul_const), 90));
  uint64_t bbccddee = abbccddee - a * hundred_million;

  buffer = write_if(buffer, a, has17digits);

  uint64x1_t ffgghhii_bbccddee_64 = {((uint64_t)(ffgghhii) << 32) | bbccddee};
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

  buffer += 16 - ((zeroes != 0 ? clz(zeroes) : 64) >> 2);
  return buffer;
#elif ZMIJ_USE_SSE
  uint32_t last_digit = value - value_div10 * 10;

  // We always write 17 digits into the buffer, but the first one can be zero.
  // buffer points to the second place in the output buffer to allow for the
  // insertion of the decimal point, so we can use the first place as scratch.
  buffer += has17digits - 1;
  buffer[16] = (char)(last_digit + '0');

  uint32_t abcdefgh = value_div10 / (uint64_t)1e8;
  uint32_t ijklmnop = value_div10 % (uint64_t)1e8;

  ZMIJ_ALIGNAS(64) static const struct {
    m128i div10k = splat64(div10k_sig);
    m128i neg10k = splat64(::neg10k);
    m128i div100 = splat32(div100_sig);
    m128i div10 = splat16((1 << 16) / 10 + 1);
#  if ZMIJ_USE_SSE4_1
    m128i neg100 = splat32(::neg100);
    m128i neg10 = splat16((1 << 8) - 10);
    m128i bswap = m128i{pack8(15, 14, 13, 12, 11, 10, 9, 8),
                        pack8(7, 6, 5, 4, 3, 2, 1, 0)};
#  else
    m128i hundred = splat32(100);
    m128i moddiv10 = splat16(10 * (1 << 8) - 1);
#  endif
    m128i zeros = splat64(::zeros);
  } consts;

  const __m128i div10k = _mm_load_si128((__m128i*)&consts.div10k);
  const __m128i neg10k = _mm_load_si128((__m128i*)&consts.neg10k);
  const __m128i div100 = _mm_load_si128((__m128i*)&consts.div100);
  const __m128i div10 = _mm_load_si128((__m128i*)&consts.div10);
#  if ZMIJ_USE_SSE4_1
  const __m128i neg100 = _mm_load_si128((__m128i*)&consts.neg100);
  const __m128i neg10 = _mm_load_si128((__m128i*)&consts.neg10);
  const __m128i bswap = _mm_load_si128((__m128i*)&consts.bswap);
#  else
  const __m128i hundred = _mm_load_si128((__m128i*)&consts.hundred);
  const __m128i moddiv10 = _mm_load_si128((__m128i*)&consts.moddiv10);
#  endif
  const __m128i zeros = _mm_load_si128((__m128i*)&consts.zeros);

  // The BCD sequences are based on ones provided by Xiang JunBo.
  __m128i x = _mm_set_epi64x(abcdefgh, ijklmnop);
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
#  else
  __m128i y_div_100 = _mm_srli_epi16(_mm_mulhi_epu16(y, div100), 3);
  __m128i y_mod_100 = _mm_sub_epi16(y, _mm_mullo_epi16(y_div_100, hundred));
  __m128i z = _mm_or_si128(_mm_slli_epi32(y_mod_100, 16), y_div_100);
  __m128i bcd_shuffled =
      _mm_sub_epi16(_mm_slli_epi16(z, 8),
                    _mm_mullo_epi16(moddiv10, _mm_mulhi_epu16(z, div10)));
  __m128i bcd = _mm_shuffle_epi32(bcd_shuffled, _MM_SHUFFLE(0, 1, 2, 3));
#  endif  // ZMIJ_USE_SSE4_1

  __m128i digits = _mm_or_si128(bcd, zeros);

  // determine number of leading zeros
  __m128i mask128 = _mm_cmpgt_epi8(bcd, _mm_setzero_si128());
  uint32_t mask = _mm_movemask_epi8(mask128);
  // We don't need a zero-check here: if the mask were zero, either the
  // significand is zero which is handled elsewhere or the only non-zero digit
  // is the last digit which we factored off. But in that case the number would
  // be printed with a different exponent that shifts the last digit into the
  // first position.
  size_t len = ((size_t)64) - clz(mask);  // size_t for native arithmetic

  _mm_storeu_si128((__m128i*)buffer, digits);
  return buffer + (last_digit != 0 ? 17 : len);
#endif    // ZMIJ_USE_SSE
}

typedef struct {
  long long sig;
  int exp;
#if ZMIJ_USE_SSE
  long long sig_div10;
#endif
} to_decimal_result;

void set_div10(to_decimal_result* r, long long value) {
#if ZMIJ_USE_SSE
  r->sig_div10 = value;
#endif
}

static to_decimal_result normalize32(to_decimal_result dec, bool subnormal) {
  if (!subnormal) return dec;
  while (dec.sig < (uint64_t)1e8) {
    dec.sig *= 10;
    --dec.exp;
  }
  set_div10(&dec, dec.sig / 10);
  return dec;
}

static to_decimal_result normalize64(to_decimal_result dec, bool subnormal) {
  if (!subnormal) return dec;
  while (dec.sig < (uint64_t)1e16) {
    dec.sig *= 10;
    --dec.exp;
  }
  set_div10(&dec, dec.sig / 10);
  return dec;
}

static to_decimal_result to_decimal_schubfach32(uint32_t bin_sig,
                                                int64_t bin_exp, bool regular,
                                                bool subnormal) {
  int dec_exp = compute_dec_exp(bin_exp, regular);
  unsigned char exp_shift = compute_exp_shift(bin_exp, dec_exp);
  uint128 pow10 = get_pow10_significand(-dec_exp);

  // Fallback to Schubfach to guarantee correctness in boundary cases.
  // This requires switching to strict overestimates of powers of 10.
  ++pow10.hi;

  // Shift the significand so that boundaries are integer.
  const int bound_shift = 2;
  uint32_t bin_sig_shifted = bin_sig << bound_shift;

  // Compute the estimates of lower and upper bounds of the rounding interval
  // by multiplying them by the power of 10 and applying modified rounding.
  uint32_t lsb = bin_sig & 1;
  uint32_t lower = (bin_sig_shifted - (regular + 1)) << exp_shift;
  lower = umulhi_inexact_to_odd32(pow10.hi, pow10.lo, lower) + lsb;
  uint32_t upper = (bin_sig_shifted + 2) << exp_shift;
  upper = umulhi_inexact_to_odd32(pow10.hi, pow10.lo, upper) - lsb;

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  long long div10 = (upper >> bound_shift) / 10;
  uint32_t shorter = div10 * 10;
  if ((shorter << bound_shift) >= lower) {
    to_decimal_result result = {(int64_t)shorter, dec_exp};
    set_div10(&result, div10);
    return normalize32(result, subnormal);
  }

  uint32_t scaled_sig =
      umulhi_inexact_to_odd32(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
  uint32_t longer_below = scaled_sig >> bound_shift;
  uint32_t longer_above = longer_below + 1;

  // Pick the closest of dec_sig_below and dec_sig_above and check if it's in
  // the rounding interval.
  int32_t cmp = (int32_t)(scaled_sig - ((longer_below + longer_above) << 1));
  bool below_closer = cmp < 0 || (cmp == 0 && (longer_below & 1) == 0);
  bool below_in = (longer_below << bound_shift) >= lower;
  uint32_t dec_sig = (below_closer & below_in) ? longer_below : longer_above;
  to_decimal_result result = {(int64_t)dec_sig, dec_exp};
  set_div10(&result, dec_sig / 10);
  return normalize32(result, subnormal);
}

static to_decimal_result to_decimal_schubfach64(uint64_t bin_sig,
                                                int64_t bin_exp, bool regular,
                                                bool subnormal) {
  int dec_exp = compute_dec_exp(bin_exp, regular);
  unsigned char exp_shift = compute_exp_shift(bin_exp, dec_exp);
  uint128 pow10 = get_pow10_significand(-dec_exp);

  // Fallback to Schubfach to guarantee correctness in boundary cases.
  // This requires switching to strict overestimates of powers of 10.
  ++pow10.lo;

  // Shift the significand so that boundaries are integer.
  const int bound_shift = 2;
  uint64_t bin_sig_shifted = bin_sig << bound_shift;

  // Compute the estimates of lower and upper bounds of the rounding interval
  // by multiplying them by the power of 10 and applying modified rounding.
  uint64_t lsb = bin_sig & 1;
  uint64_t lower = (bin_sig_shifted - (regular + 1)) << exp_shift;
  lower = umulhi_inexact_to_odd64(pow10.hi, pow10.lo, lower) + lsb;
  uint64_t upper = (bin_sig_shifted + 2) << exp_shift;
  upper = umulhi_inexact_to_odd64(pow10.hi, pow10.lo, upper) - lsb;

  // The idea of using a single shorter candidate is by Cassio Neri.
  // It is less or equal to the upper bound by construction.
  long long div10 = (upper >> bound_shift) / 10;
  uint64_t shorter = div10 * 10;
  if ((shorter << bound_shift) >= lower) {
    to_decimal_result result = {(int64_t)shorter, dec_exp};
    set_div10(&result, div10);
    return normalize64(result, subnormal);
  }

  uint64_t scaled_sig =
      umulhi_inexact_to_odd64(pow10.hi, pow10.lo, bin_sig_shifted << exp_shift);
  uint64_t longer_below = scaled_sig >> bound_shift;
  uint64_t longer_above = longer_below + 1;

  // Pick the closest of dec_sig_below and dec_sig_above and check if it's in
  // the rounding interval.
  int64_t cmp = (int64_t)(scaled_sig - ((longer_below + longer_above) << 1));
  bool below_closer = cmp < 0 || (cmp == 0 && (longer_below & 1) == 0);
  bool below_in = (longer_below << bound_shift) >= lower;
  uint64_t dec_sig = (below_closer & below_in) ? longer_below : longer_above;
  to_decimal_result result = {(int64_t)(dec_sig), dec_exp};
  set_div10(&result, dec_sig / 10);
  return normalize64(result, subnormal);
}

// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - exp_offset.
static ZMIJ_INLINE to_decimal_result to_decimal_normal32(uint32_t bin_sig,
                                                         int64_t raw_exp,
                                                         bool regular) {
  int64_t bin_exp = raw_exp - float_exp_offset;
  const int num_bits = 32;
  // An optimization from yy by Yaoyuan Guo:
  while (regular) {
    int dec_exp = use_umul128_hi64 ? umul128_hi64(bin_exp, 0x4d10500000000000)
                                   : compute_dec_exp(bin_exp, true);
    unsigned char exp_shift = compute_exp_shift(bin_exp, dec_exp);
    uint128 pow10 = get_pow10_significand(-dec_exp);

    uint32_t integral = 0;    // integral part of bin_sig * pow10
    uint64_t fractional = 0;  // fractional part of bin_sig * pow10
    uint128_t p = umul128(pow10.hi, bin_sig << exp_shift);
    integral = hi64(p);
    fractional = lo64(p);
    const uint64_t half_ulp = (uint64_t)1 << 63;

    // Exact half-ulp tie when rounding to nearest integer.
    int64_t cmp = (int64_t)(fractional - half_ulp);
    if (cmp == 0) break;

    // An optimization of integral % 10 by Dougall Johnson.
    // Relies on range calculation: (max_bin_sig << max_exp_shift) * max_u128.
    // (1 << 63) / 5 == (1 << 64) / 10 without an intermediate int128.
    const uint64_t div10_sig64 = (1ull << 63) / 5 + 1;
    long long div10 =
        ZMIJ_USE_INT128 ? umul128_hi64(integral, div10_sig64) : integral / 10;
    uint64_t digit = integral - div10 * 10;
    // or it narrows to 32-bit and doesn't use madd/msub
    ZMIJ_ASM(("" : "+r"(digit)));

    // Switch to a fixed-point representation with the least significant
    // integral digit in the upper bits and fractional digits in the lower bits.
    const int num_integral_bits = num_bits == 64 ? 4 : 32;
    const int num_fractional_bits = 64 - num_integral_bits;
    const uint64_t ten = (uint64_t)10 << num_fractional_bits;
    // Fixed-point remainder of the scaled significand modulo 10.
    uint64_t scaled_sig_mod10 =
        (digit << num_fractional_bits) | (fractional >> num_integral_bits);

    // scaled_half_ulp = 0.5 * pow10 in the fixed-point format.
    // dec_exp is chosen so that 10**dec_exp <= 2**bin_exp < 10**(dec_exp + 1).
    // Since 1ulp == 2**bin_exp it will be in the range [1, 10) after scaling
    // by 10**dec_exp. Add 1 to combine the shift with division by two.
    uint64_t scaled_half_ulp = pow10.hi >> (num_integral_bits - exp_shift + 1);
    uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;

    // Check for boundary case when rounding down to nearest 10 and
    // near-boundary case when rounding up to nearest 10.
    if (scaled_sig_mod10 == scaled_half_ulp ||
        // Case where upper == ten is insufficient: 1.342178e+08f.
        ten - upper <= 1u)  // upper == ten || upper == ten - 1
    {
      break;
    }

    bool round_up = upper >= ten;
    int64_t shorter = (int64_t)(integral - digit);
    int64_t longer = (int64_t)(integral + (cmp >= 0));
    if (ZMIJ_AARCH64) {  // Faster version without ccmp.
      int64_t dec_sig = scaled_sig_mod10 < scaled_half_ulp ? shorter : longer;
      to_decimal_result result = {round_up ? shorter + 10 : dec_sig, dec_exp};
      return result;
    }
    shorter += round_up * 10;
    bool use_shorter = (scaled_sig_mod10 <= scaled_half_ulp) + round_up != 0;
    to_decimal_result result = {use_shorter ? shorter : longer, dec_exp};
    set_div10(&result, div10 + use_shorter * round_up);
    return result;
  }
  return to_decimal_schubfach32(bin_sig, bin_exp, regular, false);
}

// Converts a binary FP number bin_sig * 2**bin_exp to the shortest decimal
// representation, where bin_exp = raw_exp - exp_offset.
static ZMIJ_INLINE to_decimal_result to_decimal_normal64(uint64_t bin_sig,
                                                         int64_t raw_exp,
                                                         bool regular) {
  int64_t bin_exp = raw_exp - double_exp_offset;
  // An optimization from yy by Yaoyuan Guo:
  while (regular) {
    int dec_exp = use_umul128_hi64 ? umul128_hi64(bin_exp, 0x4d10500000000000)
                                   : compute_dec_exp(bin_exp, true);
    unsigned char exp_shift = compute_exp_shift(bin_exp, dec_exp);
    uint128 pow10 = get_pow10_significand(-dec_exp);

    uint64_t integral = 0;    // integral part of bin_sig * pow10
    uint64_t fractional = 0;  // fractional part of bin_sig * pow10
    uint128 p = umul192_hi128(pow10.hi, pow10.lo, bin_sig << exp_shift);
    integral = p.hi;
    fractional = p.lo;
    const uint64_t half_ulp = (uint64_t)1 << 63;

    // Exact half-ulp tie when rounding to nearest integer.
    int64_t cmp = (int64_t)(fractional - half_ulp);
    if (cmp == 0) break;

    // An optimization of integral % 10 by Dougall Johnson.
    // Relies on range calculation: (max_bin_sig << max_exp_shift) * max_u128.
    // (1 << 63) / 5 == (1 << 64) / 10 without an intermediate int128.
    const uint64_t div10_sig64 = (1ull << 63) / 5 + 1;
    long long div10 =
        ZMIJ_USE_INT128 ? umul128_hi64(integral, div10_sig64) : integral / 10;
    uint64_t digit = integral - div10 * 10;
    // or it narrows to 32-bit and doesn't use madd/msub
    ZMIJ_ASM(("" : "+r"(digit)));

    // Switch to a fixed-point representation with the least significant
    // integral digit in the upper bits and fractional digits in the lower bits.
    const int num_integral_bits = 4;
    const int num_fractional_bits = 64 - num_integral_bits;
    const uint64_t ten = (uint64_t)10 << num_fractional_bits;
    // Fixed-point remainder of the scaled significand modulo 10.
    uint64_t scaled_sig_mod10 =
        (digit << num_fractional_bits) | (fractional >> num_integral_bits);

    // scaled_half_ulp = 0.5 * pow10 in the fixed-point format.
    // dec_exp is chosen so that 10**dec_exp <= 2**bin_exp < 10**(dec_exp + 1).
    // Since 1ulp == 2**bin_exp it will be in the range [1, 10) after scaling
    // by 10**dec_exp. Add 1 to combine the shift with division by two.
    uint64_t scaled_half_ulp = pow10.hi >> (num_integral_bits - exp_shift + 1);
    uint64_t upper = scaled_sig_mod10 + scaled_half_ulp;

    // Check for boundary case when rounding down to nearest 10 and
    // near-boundary case when rounding up to nearest 10.
    if (scaled_sig_mod10 == scaled_half_ulp ||
        // Case where upper == ten is insufficient: 1.342178e+08f.
        ten - upper <= 1u)  // upper == ten || upper == ten - 1
    {
      break;
    }

    bool round_up = upper >= ten;
    int64_t shorter = (int64_t)(integral - digit);
    int64_t longer = (int64_t)(integral + (cmp >= 0));
    if (ZMIJ_AARCH64) {  // Faster version without ccmp.
      int64_t dec_sig = scaled_sig_mod10 < scaled_half_ulp ? shorter : longer;
      to_decimal_result result = {round_up ? shorter + 10 : dec_sig, dec_exp};
      return result;
    }
    shorter += round_up * 10;
    bool use_shorter = (scaled_sig_mod10 <= scaled_half_ulp) + round_up != 0;
    to_decimal_result result = {use_shorter ? shorter : longer, dec_exp};
    set_div10(&result, div10 + use_shorter * round_up);
    return result;
  }
  return to_decimal_schubfach64(bin_sig, bin_exp, regular, false);
}

char* zmij_detail_write_float(float value, char* buffer) {
  uint32_t bits = float_to_bits(value);
  // It is beneficial to extract exponent and significand early.
  int64_t bin_exp = float_get_exp(bits);   // binary exponent
  uint32_t bin_sig = float_get_sig(bits);  // binary significand

  *buffer = '-';
  buffer += float_is_negative(bits);

  to_decimal_result dec;
  if (bin_exp == 0 || bin_exp == float_exp_mask) {
    if (bin_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return buffer + 3;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return buffer + 1;
    }
    dec = to_decimal_schubfach32(bin_sig, 1 - float_exp_offset, true, true);
  } else {
    dec = to_decimal_normal32(bin_sig | float_implicit_bit, bin_exp,
                              bin_sig != 0);
  }
  int dec_exp = dec.exp;

  // Write significand.
  char* start = buffer;
  if (dec.sig < (uint32_t)1e7) {
    dec.sig *= 10;
    --dec_exp;
  }
  bool has9digits = dec.sig >= (uint32_t)1e8;
  dec_exp += FLT_DECIMAL_DIG - 2 + has9digits;
  buffer = write_significand9(buffer + 1, dec.sig, has9digits);
  start[0] = start[1];
  start[1] = '.';

  // Write exponent.
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian()) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  dec_exp = dec_exp >= 0 ? dec_exp : -dec_exp;
  memcpy(buffer, digits2(dec_exp), 2);
  buffer[2] = '\0';
  return buffer + 2;
}

// It is slightly faster to return a pointer to the end than the size.
char* zmij_detail_write_double(double value, char* buffer) {
  uint64_t bits = double_to_bits(value);
  // It is beneficial to extract exponent and significand early.
  int64_t bin_exp = double_get_exp(bits);   // binary exponent
  uint64_t bin_sig = double_get_sig(bits);  // binary significand

  *buffer = '-';
  buffer += double_is_negative(bits);

  to_decimal_result dec;
  if (bin_exp == 0 || bin_exp == double_exp_mask) {
    if (bin_exp != 0) {
      memcpy(buffer, bin_sig == 0 ? "inf" : "nan", 4);
      return buffer + 3;
    }
    if (bin_sig == 0) {
      memcpy(buffer, "0", 2);
      return buffer + 1;
    }
    dec = to_decimal_schubfach64(bin_sig, 1 - double_exp_offset, true, true);
  } else {
    dec = to_decimal_normal64(bin_sig | double_implicit_bit, bin_exp,
                              bin_sig != 0);
  }
  int dec_exp = dec.exp;

  // Write significand.
  char* start = buffer;
  bool has17digits = dec.sig >= (uint64_t)1e16;
  dec_exp += DBL_DECIMAL_DIG - 2 + has17digits;
  long long sig_div10 = 0;
#if ZMIJ_USE_SSE
  sig_div10 = dec.sig_div10;
#endif

  if (dec_exp >= -4 && dec_exp < 0) {
    char* point = buffer + 1;
    memcpy(buffer, "0.0000000", 8);
    buffer = write_significand17(buffer + 1 - dec_exp, dec.sig, has17digits,
                                 sig_div10);
    if (ZMIJ_USE_SSE) *point = '.';
    *buffer = '\0';
    return buffer;
  }

  // Could merge this path with the scientific path, or increase the upper
  // bound if this branch is bad on real world data.
  if (dec_exp >= 0 && dec_exp < 16) {
    // Avoid reading uninitialized memory (would be unnecessary in asm).
    write8(buffer + 16, 0);

    buffer = write_significand17_no_simd(buffer, dec.sig, has17digits);

    // Branchless move to make space for the '.' without OOB accesses.
    char* part1 = start + dec_exp + (dec_exp < 2);
    char* part2 = part1 + (dec_exp < 2) + (dec_exp < 9 ? 7 : 0);
    uint64_t value1 = read8(part1);
    uint64_t value2 = read8(part2);
    write8(part1 + 1, value1);
    write8(part2 + 1, value2);

    char* dot = start + dec_exp + 1;
    *dot = '.';

    buffer = buffer > dot ? buffer + 1 : dot;
    *buffer = '\0';
    return buffer;
  }

  buffer = write_significand17(buffer + 1, dec.sig, has17digits, sig_div10);
  start[0] = start[1];
  start[1] = '.';
  buffer -= (buffer - 1 == start + 1);  // Remove trailing point.

  // Write exponent.
  uint16_t e_sign = dec_exp >= 0 ? ('+' << 8 | 'e') : ('-' << 8 | 'e');
  if (is_big_endian()) e_sign = e_sign << 8 | e_sign >> 8;
  memcpy(buffer, &e_sign, 2);
  buffer += 2;
  dec_exp = dec_exp >= 0 ? dec_exp : -dec_exp;

  // digit = dec_exp / 100
  uint32_t digit = use_umul128_hi64
                       ? umul128_hi64(dec_exp, 0x290000000000000)
                       : ((uint32_t)dec_exp * div100_sig) >> div100_exp;
  uint32_t digit_with_nuls = '0' + digit;
  if (is_big_endian()) digit_with_nuls <<= 24;
  memcpy(buffer, &digit_with_nuls, 4);
  buffer += dec_exp >= 100;
  memcpy(buffer, digits2(dec_exp - digit * 100), 2);
  return buffer + 2;
}
