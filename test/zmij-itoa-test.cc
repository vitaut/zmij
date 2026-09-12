// Tests for zmij integer formatting, comparing against fmt. Compiled into
// every flag-variant test build (see add_zmij_test) so the BCD layer is
// exercised under each preprocessor configuration; includes the
// implementation like zmij-impl-test.cc so each variant tests its own build.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <gtest/gtest.h>
#include <stdint.h>  // uint64_t
#include <string.h>  // memset

#include <limits>  // std::numeric_limits
#include <string>  // std::string
#include <vector>  // std::vector

#include "fmt/format.h"
#include "zmij-int.cc"

namespace {

#if ZMIJ_USE_INT128
using wide_uint = unsigned __int128;
#else
using wide_uint = uint64_t;
#endif

// std::numeric_limits and std::is_signed are unreliable for __int128 in
// strict-conformance mode, so compute the properties directly.
template <typename T> constexpr auto is_signed_int() -> bool {
  return T(-1) < T(0);
}

// Maximum value of T as an unsigned magnitude.
template <typename T> constexpr auto max_magnitude() -> wide_uint {
  return is_signed_int<T>()
             ? (wide_uint(1) << (sizeof(T) * 8 - 1)) - 1
             : (sizeof(T) == sizeof(wide_uint)
                    ? ~wide_uint(0)
                    : (wide_uint(1) << (sizeof(T) * 8 % (sizeof(wide_uint) * 8))) - 1);
}

template <typename T> constexpr auto buffer_size() -> size_t {
  return sizeof(T) <= 4 ? size_t(zmij::int32_buffer_size)
         : sizeof(T) <= 8 ? size_t(zmij::int64_buffer_size)
                          : size_t(zmij::int128_buffer_size);
}

// Formats value with zmij::write into an exactly buffer_size<T>()-sized
// buffer and verifies the bytes before and past it stay untouched.
template <typename T> auto zmij_to_string(T value) -> std::string {
  const size_t size = buffer_size<T>();
  char storage[8 + buffer_size<T>() + 8];
  memset(storage, '?', sizeof(storage));
  char* buffer = storage + 8;
  char* end = zmij::write(buffer, size, value);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(storage[i], '?') << "buffer underrun at offset "
                               << ptrdiff_t(i) - 8;
  }
  for (size_t i = 8 + size; i < sizeof(storage); ++i) {
    EXPECT_EQ(storage[i], '?') << "buffer overrun at offset " << i - 8;
  }
  return std::string(buffer, end);
}

// A test value as a sign-and-magnitude pair so that candidates outside T's
// range (e.g. min - 1) are representable during generation and filtered out.
struct candidate {
  wide_uint magnitude;
  bool negative;
};

template <typename T> auto in_range(candidate c) -> bool {
  if (!c.negative || c.magnitude == 0)
    return c.magnitude <= max_magnitude<T>();
  return is_signed_int<T>() && c.magnitude <= max_magnitude<T>() + 1;
}

// Converts an in-range candidate to T via two's-complement wrapping.
template <typename T> auto to_value(candidate c) -> T {
  wide_uint bits = c.negative ? wide_uint(0) - c.magnitude : c.magnitude;
  return T(bits);
}

// All powers of two and ten representable in wide_uint, each with both
// adjacent values, both signs, plus zero and T's min/max and their neighbors.
template <typename T> auto boundary_candidates() -> std::vector<candidate> {
  std::vector<candidate> result;
  auto add = [&result](wide_uint magnitude, bool negative) {
    result.push_back(candidate{magnitude, negative});
    if (magnitude > 0) result.push_back(candidate{magnitude - 1, negative});
    result.push_back(candidate{magnitude + 1, negative});
  };
  add(0, false);
  for (int k = 0; k < int(sizeof(wide_uint)) * 8; ++k) {
    add(wide_uint(1) << k, false);
    add(wide_uint(1) << k, true);
  }
  for (wide_uint p = 1; p <= ~wide_uint(0) / 10; p *= 10) {
    add(p, false);
    add(p, true);
  }
  // min and max of T. add() also covers min + 1 and max - 1.
  add(max_magnitude<T>(), false);
  add(is_signed_int<T>() ? max_magnitude<T>() + 1 : 0, is_signed_int<T>());
  return result;
}

template <typename T> class itoa_test : public ::testing::Test {};

using int_types = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t,
                                   uint32_t, int64_t, uint64_t
#if ZMIJ_USE_INT128
                                   ,
                                   __int128, unsigned __int128
#endif
                                   >;
TYPED_TEST_SUITE(itoa_test, int_types);

TYPED_TEST(itoa_test, boundary_values) {
  for (candidate c : boundary_candidates<TypeParam>()) {
    if (!in_range<TypeParam>(c)) continue;
    TypeParam value = to_value<TypeParam>(c);
    EXPECT_EQ(zmij_to_string(value), fmt::format("{}", value));
  }
}

TEST(itoa_api_test, truncation) {
  char buffer[4];
  char* end = zmij::write(buffer, 4, 123456);
  EXPECT_EQ(std::string(buffer, end), "1234");
  end = zmij::write(buffer, 4, -123456);
  EXPECT_EQ(std::string(buffer, end), "-123");
  end = zmij::write(buffer, 0, 42);
  EXPECT_EQ(end, buffer);
}

}  // namespace
