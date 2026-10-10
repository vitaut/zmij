// Tests for zmij implementation details, kept separate from the public-API
// tests in zmij-test.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <gtest/gtest.h>
#include <stddef.h>  // size_t
#include <stdio.h>
#include <stdlib.h>  // malloc

#include <cmath>   // std::isfinite
#include <limits>  // std::numeric_limits
#include <string>

// Lets tests force an allocation failure: the code under test calls malloc
// unqualified, so this zmij overload shadows ::malloc.
namespace zmij {
bool fail_malloc = false;
void* malloc(size_t n) { return fail_malloc ? nullptr : ::malloc(n); }
}  // namespace zmij

// Include zmij.cc instead of linking with the library to test internal
// functions.
#include "zmij.cc"

// A copyable, inline-storage bigint for tests. bigint holds a raw limbs pointer,
// so the copy re-points at this object's own storage.
struct fixed_bigint : bigint {
  // Holds the widest pow10_compute value: 10**max_exp times a result_bits
  // number.
  static constexpr int cap =
      (pow10::max_exp * 34 / 10 + pow10::result_bits) / 32 + 4;
  uint32_t storage[cap];

  explicit fixed_bigint(uint128_t value = 0) noexcept
      : bigint(value, storage, cap) {}
  fixed_bigint(const fixed_bigint& other) noexcept : bigint(0, storage, cap) {
    num_limbs = other.num_limbs;
    memcpy(storage, other.storage, size_t(num_limbs) * sizeof(uint32_t));
  }
  auto operator=(const fixed_bigint& other) noexcept -> fixed_bigint& {
    if (this == &other) return *this;
    num_limbs = other.num_limbs;
    memcpy(storage, other.storage, size_t(num_limbs) * sizeof(uint32_t));
    return *this;
  }

  // Builds the value from n little-endian uint64 limbs.
  static auto from_u64s(const uint64_t* v, int n) -> fixed_bigint {
    fixed_bigint r;
    r.num_limbs = 2 * n;
    for (int i = 0; i < n; ++i) {
      r.limbs[2 * i] = uint32_t(v[i]);
      r.limbs[2 * i + 1] = uint32_t(v[i] >> 32);
    }
    r.trim();
    return r;
  }

  void add(uint32_t k) noexcept {
    uint64_t carry = k;
    for (int i = 0; i < num_limbs && carry != 0; ++i) {
      uint64_t s = uint64_t(limbs[i]) + carry;
      limbs[i] = uint32_t(s);
      carry = s >> 32;
    }
    if (carry != 0) {
      assert(num_limbs < max_limbs);
      limbs[num_limbs++] = uint32_t(carry);
    }
  }
};

static auto mul(const bigint& a, const bigint& b) -> fixed_bigint {
  fixed_bigint r;
  if (a.num_limbs == 0 || b.num_limbs == 0) return r;
  r.num_limbs = a.num_limbs + b.num_limbs;
  assert(r.num_limbs <= r.max_limbs);
  for (int i = 0; i < r.num_limbs; ++i) r.limbs[i] = 0;
  for (int i = 0; i < a.num_limbs; ++i) {
    uint64_t carry = 0;
    for (int j = 0; j < b.num_limbs; ++j) {
      uint64_t cur =
          uint64_t(r.limbs[i + j]) + uint64_t(a.limbs[i]) * b.limbs[j] + carry;
      r.limbs[i + j] = uint32_t(cur);
      carry = cur >> 32;
    }
    r.limbs[i + b.num_limbs] = uint32_t(carry);
  }
  r.trim();
  return r;
}

// Three-way comparison of two nonnegative bigints.
static auto cmp(const bigint& a, const bigint& b) -> int {
  if (a.num_limbs != b.num_limbs) return a.num_limbs < b.num_limbs ? -1 : 1;
  for (int i = a.num_limbs; i-- > 0;) {
    if (a.limbs[i] != b.limbs[i]) return a.limbs[i] < b.limbs[i] ? -1 : 1;
  }
  return 0;
}

// Returns the number of significant bits in a nonzero value.
static auto bit_length(const bigint& value) -> int {
  assert(value.num_limbs != 0);
  int top_bits = 64 - clz(value.limbs[value.num_limbs - 1]);  // limbs: 32 bits
  return (value.num_limbs - 1) * 32 + top_bits;
}

// Divides value by 10**n in place, discarding the remainder. Only division by
// 10**9 is available, so scale value to pad the divisor to a whole number of
// 10**9 groups: floor(v / 10**n) == floor(v * 10**pad / 10**(n + pad)).
static void div_pow10(bigint& value, int n) {
  int num_groups = (n + 8) / 9, pad = num_groups * 9 - n;
  value.mul(uint32_t(pow10s[pad]));
  for (int i = 0; i < num_groups; ++i) value.divmod_1e9();
}

// Returns a value of at most 128 bits as a uint128.
static auto to_uint128(const bigint& value) -> uint128 {
  assert(value.num_limbs <= 4);
  uint64_t part[4] = {};
  for (int i = 0; i < value.num_limbs; ++i) part[i] = value.limbs[i];
  return {part[3] << 32 | part[2], part[1] << 32 | part[0]};
}

// Each table entry must be a power of ten rounded down to 128 significant bits:
// entry dec_exp is floor(10**dec_exp / 2**e) for the e that makes it exactly
// 128 bits wide. Checked against exact big integers with no floating point.
TEST(zmij_impl_test, pow10) {
  constexpr int dec_exp_min = -307;
  constexpr int dec_exp_max =
      dec_exp_min + pow10_significand_table::num_pow10s - 1;

  auto check = [](int dec_exp, const bigint& sig) {
    uint128 expected = to_uint128(sig);
    uint128 actual = static_data.pow10_significands[dec_exp];
    EXPECT_EQ(actual.hi, expected.hi) << "dec_exp=" << dec_exp;
    EXPECT_EQ(actual.lo, expected.lo) << "dec_exp=" << dec_exp;
  };

  // Non-negative exponents: 10**dec_exp is the integer p, grown one power per
  // step. Truncating p to 128 bits divides it by 2**e, which bigint does as a
  // division by 10**e once p is scaled by 5**e.
  fixed_bigint p(1);
  for (int dec_exp = 0; dec_exp <= dec_exp_max; ++dec_exp) {
    if (dec_exp != 0) p.mul(10);  // p == 10**dec_exp
    fixed_bigint sig = p;
    int e = bit_length(p) - 128;
    if (e < 0) {
      sig.shl(-e);  // p is narrower than the significand
    } else {
      sig.mul_pow5(e);
      div_pow10(sig, e);
    }
    check(dec_exp, sig);
  }

  // Negative exponents: 10**dec_exp is 1 / q with q == 10**-dec_exp, so the
  // significand is floor(2**k / q). Taking k as 127 plus q's width puts the
  // quotient in [2**127, 2**128), i.e. exactly 128 bits wide.
  fixed_bigint q(1);
  for (int dec_exp = -1; dec_exp >= dec_exp_min; --dec_exp) {
    q.mul(10);  // q == 10**-dec_exp
    fixed_bigint sig(1);
    sig.shl(127 + bit_length(q));
    div_pow10(sig, -dec_exp);
    check(dec_exp, sig);
  }
}

// pow10::compute(x, m) must return e with m == floor(10**x / 2**e) exactly and
// m normalized (top bit set), over the whole supported range |x| <= max_exp (a
// superset of every x write_big can pass). Checked against exact big integers
// with no floating point.
TEST(zmij_impl_test, pow10_compute) {
  constexpr int xmax = pow10::max_exp;

  // Non-negative exponents: 10**x is the integer P, grown one power per step.
  fixed_bigint p(1);
  for (int x = 0; x <= xmax; ++x) {
    if (x != 0) p.mul(10);  // p == 10**x
    uint64_t m[pow10::result_limbs];
    int e = pow10::compute(x, m);
    ASSERT_NE(m[pow10::result_limbs - 1] >> 63, 0u) << "x=" << x;  // normalized
    fixed_bigint mm = fixed_bigint::from_u64s(m, pow10::result_limbs);
    if (e >= 0) {  // floor(P / 2**e): mm*2**e <= P < (mm+1)*2**e
      fixed_bigint lo = mm;
      lo.shl(e);
      fixed_bigint hi = mm;
      hi.add(1);
      hi.shl(e);
      ASSERT_LE(cmp(lo, p), 0) << "x=" << x;
      ASSERT_LT(cmp(p, hi), 0) << "x=" << x;
    } else {  // 10**x * 2**(-e) is integral and must equal mm exactly
      fixed_bigint exact = p;
      exact.shl(-e);
      ASSERT_EQ(cmp(mm, exact), 0) << "x=" << x;
    }
  }

  // Negative exponents: e < 0, so 2**(-e) is integral and 10**|x| = q.
  // m == floor(2**(-e) / q): mm*q <= 2**(-e) < (mm+1)*q.
  fixed_bigint q(1);
  for (int x = -1; x >= -xmax; --x) {
    q.mul(10);  // q == 10**|x|
    uint64_t m[pow10::result_limbs];
    int e = pow10::compute(x, m);
    ASSERT_LT(e, 0) << "x=" << x;
    ASSERT_NE(m[pow10::result_limbs - 1] >> 63, 0u) << "x=" << x;  // normalized
    fixed_bigint mm = fixed_bigint::from_u64s(m, pow10::result_limbs);
    fixed_bigint two(1);
    two.shl(-e);
    fixed_bigint mmp1 = mm;
    mmp1.add(1);
    ASSERT_LE(cmp(mul(mm, q), two), 0) << "x=" << x;
    ASSERT_LT(cmp(two, mul(mmp1, q)), 0) << "x=" << x;
  }
}

TEST(zmij_impl_test, umul128) {
#if ZMIJ_USE_CONSTEXPR
  constexpr auto max_product = umul128(UINT64_MAX, UINT64_MAX);
  static_assert(uint64_t(max_product) == 1, "constexpr lower half");
  static_assert(uint64_t(max_product >> 64) == UINT64_MAX - 1,
                "constexpr upper half");
#endif

  struct product_case {
    uint64_t x, y, hi, lo;
  };
  const product_case cases[] = {
      {0, UINT64_MAX, 0, 0},
      {0x100000001, 0x100000001, 1, 0x200000001},
      {UINT64_MAX, 0x100000001, 0x100000000, 0xfffffffeffffffff},
      {UINT64_MAX, UINT64_MAX, UINT64_MAX - 1, 1},
  };
  for (const auto& c : cases) {
    auto product = umul128(c.x, c.y);
    EXPECT_EQ(uint64_t(product), c.lo) << "x=" << c.x << ", y=" << c.y;
    EXPECT_EQ(uint64_t(product >> 64), c.hi) << "x=" << c.x << ", y=" << c.y;
  }
}

TEST(zmij_impl_test, utilities) {
  EXPECT_EQ(clz(1), 63);
  EXPECT_EQ(clz(~0ull), 0);

  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000000ull), 0);
  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000001ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x00000000'00000009ull), 1);
  EXPECT_EQ(count_trailing_nonzeros(0x00090000'09000000ull), 7);
  EXPECT_EQ(count_trailing_nonzeros(0x01000000'00000000ull), 8);
  EXPECT_EQ(count_trailing_nonzeros(0x09000000'00000000ull), 8);
}

static auto to_string(const bigint& value) -> std::string {
  fixed_bigint n(0);  // A mutable copy to consume via divmod_1e9.
  n.num_limbs = value.num_limbs;
  memcpy(n.limbs, value.limbs, size_t(n.num_limbs) * sizeof(*n.limbs));
  std::string s;
  // Extract 9-digit groups, least significant first.
  while (n.num_limbs != 0) {
    uint32_t group = n.divmod_1e9();
    char buf[16];  // The most significant group is unpadded, the rest are not.
    snprintf(buf, sizeof(buf), n.num_limbs != 0 ? "%09u" : "%u", group);
    s.insert(0, buf);
  }
  return s.empty() ? "0" : s;
}

TEST(zmij_impl_test, bigint) {
  // Construction from a 128-bit value and base-10**9 output.
  EXPECT_EQ(to_string(fixed_bigint(0)), "0");
  EXPECT_EQ(to_string(fixed_bigint(123456789)), "123456789");
  EXPECT_EQ(to_string(fixed_bigint(1000000000000000000ull)),
            "1000000000000000000");

  // shl multiplies by 2**bits (word-aligned and unaligned).
  fixed_bigint a(1);
  a.shl(64);
  EXPECT_EQ(to_string(a), "18446744073709551616");
  fixed_bigint b(1);
  b.shl(80);
  EXPECT_EQ(to_string(b), "1208925819614629174706176");
}

TEST(zmij_impl_test, shr_round_even) {
  // Divides a 128-bit value by 2**bits, rounding ties to even.
  auto rshift = [](uint128_t value, int bits) {
    return uint64_t(shr_round_even(value, bits));
  };
  // Ties round to even.
  EXPECT_EQ(rshift(6, 1), 3u);  // 3.0 -> 3 (exact)
  EXPECT_EQ(rshift(5, 1), 2u);  // 2.5 -> 2 (tie down to even)
  EXPECT_EQ(rshift(7, 1), 4u);  // 3.5 -> 4 (tie up to even)
  EXPECT_EQ(rshift(3, 1), 2u);  // 1.5 -> 2 (tie up to even)
  // Non-ties round to nearest.
  EXPECT_EQ(rshift(1, 2), 0u);  // 0.25 -> 0
  EXPECT_EQ(rshift(3, 2), 1u);  // 0.75 -> 1
  // Shifts crossing the 64-bit boundary.
  EXPECT_EQ(rshift(uint128_t(1) << 63, 64), 0u);  // 0.5 -> 0 (tie to even)
  EXPECT_EQ(rshift(uint128_t(3) << 62, 64), 1u);  // 0.75 -> 1
  // A full-width value: (2**64 - 1)**2 >> 64 = 2**64 - 2, remainder rounds
  // down.
  EXPECT_EQ(rshift(umul128(~uint64_t(0), ~uint64_t(0)), 64), ~uint64_t(0) - 1);
  // n >= 128 shifts everything out; 2**127 is the only in-range tie.
  EXPECT_EQ(rshift(uint128_t(1) << 126, 128), 0u);  // 0.25 -> 0
  EXPECT_EQ(rshift(uint128_t(1) << 127, 128), 0u);  // 0.5 -> 0 (tie to even)
  EXPECT_EQ(rshift(uint128_t(3) << 126, 128), 1u);  // 0.75 -> 1
  EXPECT_EQ(rshift(uint128_t(1) << 127, 200), 0u);  // far past the width
}

TEST(zmij_impl_test, bigint_divmod_1e9) {
  // Returns value % 10**9 and leaves value / 10**9 in place.
  fixed_bigint n(123456789012345678ull);
  EXPECT_EQ(n.divmod_1e9(), 12345678u);  // Low 9 digits.
  EXPECT_EQ(to_string(n), "123456789");  // Remaining high digits.

  // A value below 10**9 becomes empty, returning the value itself.
  fixed_bigint small(42);
  EXPECT_EQ(small.divmod_1e9(), 42u);
  EXPECT_EQ(small.num_limbs, 0);

  // An exact multiple of 10**9 yields a zero remainder.
  fixed_bigint exact(3000000000ull);
  EXPECT_EQ(exact.divmod_1e9(), 0u);
  EXPECT_EQ(to_string(exact), "3");
}

TEST(zmij_impl_test, bigint_mul) {
  // mul multiplies by a factor below 2**32, carrying across limbs.
  fixed_bigint a(1);
  a.mul(1'000'000'000u);
  EXPECT_EQ(to_string(a), "1000000000");
  fixed_bigint b(0xffffffffull);  // Forces a carry into a new limb.
  b.mul(0xffffffffu);
  EXPECT_EQ(to_string(b), "18446744065119617025");

  // mul_pow5 multiplies by 5**n across the 5**13 chunk boundary.
  fixed_bigint c(1);
  c.mul_pow5(1);
  EXPECT_EQ(to_string(c), "5");
  fixed_bigint d(1);
  d.mul_pow5(13);
  EXPECT_EQ(to_string(d), "1220703125");
  fixed_bigint e(1);
  e.mul_pow5(27);  // 5**27, spanning two full chunks and a remainder.
  EXPECT_EQ(to_string(e), "7450580596923828125");
}

// The on-the-fly shortest path must match the tested table-based fast path,
// validating the pow10 kernel and trim logic.
TEST(zmij_impl_test, shortest_big_double) {
  double edges[] = {1.0,
                    0.0,
                    -0.0,
                    43210.0,
                    43210.1,
                    10000.0,
                    0.0001,
                    0.00001,
                    1234567.0,
                    6.62607015e-34,
                    5.444310685350916e+14,
                    std::numeric_limits<double>::max(),
                    std::numeric_limits<double>::min(),
                    std::numeric_limits<double>::denorm_min()};
  for (double v : edges) {
    char fast[64], big[64];
    EXPECT_EQ(std::string(big, zmij::detail::write_big(big, sizeof(big), v)),
              std::string(fast, zmij::detail::write(fast, v)))
        << v;
  }
}

TEST(zmij_impl_test, write_big_allocation_failure) {
  // write_big allocates only for a long double wider than double.
  if (std::numeric_limits<long double>::digits <=
      std::numeric_limits<double>::digits) {
    GTEST_SKIP() << "long double doesn't allocate";
  }

  // A value that needs the full long double precision so the public wrappers
  // can't fall back to the double fast path.
  long double value = std::nextafter(1.0L, 2.0L);
  char buf[64];

  zmij::fail_malloc = true;
  EXPECT_EQ(zmij::detail::write_big(buf, sizeof(buf), value, 30,
                                    zmij::format::scientific),
            0u);
  EXPECT_EQ(zmij::write_scientific(buf, sizeof(buf), value, 30), nullptr);
  EXPECT_EQ(zmij::write_general(buf, sizeof(buf), value, 30), nullptr);
  EXPECT_EQ(zmij::write_fixed(buf, sizeof(buf), value, 30), nullptr);
  zmij::fail_malloc = false;

  // The same call succeeds once allocation works again.
  EXPECT_NE(zmij::write_scientific(buf, sizeof(buf), value, 30), nullptr);
}
