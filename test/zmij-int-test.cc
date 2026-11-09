// Tests for zmij integer formatting, comparing against fmt. Compiled into
// every flag-variant test build (see add_zmij_test) so the BCD layer is
// exercised under each preprocessor configuration; includes the
// implementation like zmij-impl-test.cc so each variant tests its own build.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "zmij-int.cc"

#include <gtest/gtest.h>
#include <limits.h>  // CHAR_BIT
#include <stddef.h>  // ptrdiff_t, size_t
#include <stdint.h>  // uint64_t
#include <string.h>  // memset

#include <string>       // std::string
#include <type_traits>  // std::conditional
#include <vector>       // std::vector

#include "fmt/format.h"

namespace {

using wide_uint =
    typename std::conditional<ZMIJ_USE_INT128, zmij::detail::uint128_t,
                              uint64_t>::type;

// std::numeric_limits and std::is_signed are unreliable for __int128 in
// strict-conformance mode, so compute the properties directly.
template <typename T> constexpr auto is_signed() -> bool {
  return T(-1) < T(0);
}

// Maximum value of T represented as wide_uint.
template <typename T> constexpr auto max_value() -> wide_uint {
  return ~wide_uint(0) >>
         ((sizeof(wide_uint) - sizeof(T)) * CHAR_BIT + is_signed<T>());
}

template <typename T> constexpr auto buffer_size() -> size_t {
  return sizeof(T) <= 4   ? (is_signed<T>() ? zmij::int32_buffer_size
                                            : zmij::uint32_buffer_size)
         : sizeof(T) <= 8 ? (is_signed<T>() ? zmij::int64_buffer_size
                                            : zmij::uint64_buffer_size)
                          : (is_signed<T>() ? zmij::int128_buffer_size
                                            : zmij::uint128_buffer_size);
}

// Returns the offset of the first byte in [p, p + n) other than '?', or n.
auto find_clobbered(const char* p, size_t n) -> size_t {
  size_t i = 0;
  while (i < n && p[i] == '?') ++i;
  return i;
}

// Formats value with zmij::write into an exactly buffer_size<T>()-sized
// buffer and verifies the bytes before and past it stay untouched.
template <typename T> auto zmij_to_string(T value) -> std::string {
  constexpr size_t guard_size = 8;
  constexpr size_t size = buffer_size<T>();
  char storage[size + 2 * guard_size];
  memset(storage, '?', sizeof(storage));
  char* buffer = storage + guard_size;
  char* end = zmij::write(buffer, size, value);
  size_t under = find_clobbered(storage, guard_size);
  EXPECT_EQ(under, guard_size) << "buffer underrun at offset "
                               << ptrdiff_t(under) - ptrdiff_t(guard_size);
  size_t over = find_clobbered(buffer + size, guard_size);
  EXPECT_EQ(over, guard_size) << "buffer overrun at offset " << size + over;
  return std::string(buffer, end);
}

// A test value as a sign-and-magnitude pair so that candidates outside T's
// range (e.g. min - 1) are representable during generation and filtered out.
struct candidate {
  wide_uint magnitude;
  bool negative;
};

template <typename T> auto in_range(candidate c) -> bool {
  if (!c.negative || c.magnitude == 0) return c.magnitude <= max_value<T>();
  return is_signed<T>() && c.magnitude <= max_value<T>() + 1;
}

// Converts an in-range candidate to T without an out-of-range signed cast.
template <typename T> auto to_value(candidate c) -> T {
  if (!c.negative || c.magnitude == 0) return T(c.magnitude);
  return T(-T(c.magnitude - 1) - 1);
}

// How far around each boundary to test.
constexpr int boundary_radius = 100;

// All powers of two and ten representable in wide_uint, both signs, plus zero
// and T's min/max, each with all values within boundary_radius of it.
template <typename T> auto boundary_candidates() -> std::vector<candidate> {
  std::vector<candidate> result;
  auto add = [&result](wide_uint magnitude, bool negative) {
    for (int d = -boundary_radius; d <= boundary_radius; ++d) {
      wide_uint delta = wide_uint(d < 0 ? -d : d);
      if ((d > 0) != negative) {  // Away from zero: the magnitude grows.
        if (magnitude <= ~wide_uint(0) - delta)
          result.push_back({magnitude + delta, negative});
      } else if (delta <= magnitude) {
        result.push_back({magnitude - delta, negative});
      } else {  // Crosses zero.
        result.push_back({delta - magnitude, !negative});
      }
    }
  };
  add(0, false);
  for (int k = 0; k < int(sizeof(wide_uint)) * CHAR_BIT; ++k) {
    add(wide_uint(1) << k, false);
    add(wide_uint(1) << k, true);
  }
  for (wide_uint p = 1; p <= ~wide_uint(0) / 10; p *= 10) {
    add(p, false);
    add(p, true);
  }
  // min and max of T.
  add(max_value<T>(), false);
  if (is_signed<T>()) add(max_value<T>() + 1, true);
  return result;
}

template <typename T> class int_test : public testing::Test {};

using int_types = testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t,
                                 uint32_t, int64_t, uint64_t
#if ZMIJ_USE_INT128
                                 ,
                                 __int128, unsigned __int128
#endif
                                 >;
TYPED_TEST_SUITE(int_test, int_types);

TYPED_TEST(int_test, boundary_values) {
  for (candidate c : boundary_candidates<TypeParam>()) {
    if (!in_range<TypeParam>(c)) continue;
    TypeParam value = to_value<TypeParam>(c);
    EXPECT_EQ(zmij_to_string(value), fmt::to_string(value));
  }
}

TEST(int_api_test, truncation) {
  char buffer[4];
  char* end = zmij::write(buffer, 4, 123456);
  EXPECT_EQ(std::string(buffer, end), "1234");
  end = zmij::write(buffer, 4, -123456);
  EXPECT_EQ(std::string(buffer, end), "-123");
  end = zmij::write(buffer, 0, 42);
  EXPECT_EQ(end, buffer);
}

}  // namespace
