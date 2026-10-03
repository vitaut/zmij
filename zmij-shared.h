// A double-to-string conversion library: https://github.com/vitaut/zmij/
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.
//
// Configuration macros shared by zmij.cc and zmij-int.cc. This is an internal
// header: none of these affect the public interface in zmij.h.

#ifndef ZMIJ_SHARED_H_
#define ZMIJ_SHARED_H_

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

// 256-bit paths for the u128 body chunks.
#ifdef ZMIJ_USE_AVX2
// Use the provided definition.
static_assert(!ZMIJ_USE_AVX2 || ZMIJ_USE_SSE4_1);
#elif defined(__AVX2__)
#  define ZMIJ_USE_AVX2 ZMIJ_USE_SSE4_1
#else
#  define ZMIJ_USE_AVX2 0
#endif

// Fused multiply-add, for the float-reciprocal digit kernels.  This is
// a microarchtiecture level 3 feature, and usually comes paired with AVX2.
// MSVC has no __FMA__, but /arch:AVX2 implies FMA there; clang-cl defines
// __FMA__ when FMA is on, so it takes the first branch.
#ifdef ZMIJ_USE_FMA
// Use the provided definition.
static_assert(!ZMIJ_USE_FMA || ZMIJ_USE_SSE4_1);
#elif defined(__FMA__) || \
    (defined(_MSC_VER) && !defined(__clang__) && defined(__AVX2__))
#  define ZMIJ_USE_FMA ZMIJ_USE_SSE4_1
#else
#  define ZMIJ_USE_FMA 0
#endif

#ifdef __x86_64__
#  define ZMIJ_X86_64 1
#else
#  define ZMIJ_X86_64 0
#endif

#ifdef _MSC_VER
#  define ZMIJ_MSC_VER _MSC_VER
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

#endif  // ZMIJ_SHARED_H_
