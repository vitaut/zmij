#include <stdio.h>

#define ZMIJ_ENABLE_COMPILETIME_EVALUATION
#define ZMIJ_HEADER_ONLY  // implied by compiletime evaluation

#include "zmij.h"

#ifdef ZMIJ_ENABLE_COMPILETIME_EVALUATION
#  include <array>
#  include <limits>
#  include <string_view>

namespace zmij_constexpr {
constexpr auto to_string(
    double value,
    char* buf = std::array<char, zmij::double_buffer_size>{}.data())
    -> std::string_view {
  const auto end = zmij::detail::write(value, buf);
  return {buf, static_cast<size_t>(end - buf)};
}
constexpr auto to_string(
    float value, char* buf = std::array<char, zmij::float_buffer_size>{}.data())
    -> std::string_view {
  const auto end = zmij::detail::write(value, buf);
  return {buf, static_cast<size_t>(end - buf)};
}

static_assert(to_string(6.62607015e-34) == "6.62607015e-34");
static_assert(to_string(5.444310685350916e+14) == "5.444310685350916e+14");
static_assert(to_string(6.62607e-34f) == "6.62607e-34");
static_assert(to_string(1.3421781e+08f) == "1.3421781e+08");
static_assert(to_string(std::numeric_limits<double>::denorm_min()) == "5e-324");
static_assert(to_string(std::numeric_limits<float>::denorm_min()) == "1e-45");
static_assert(to_string(1e-323) == "1e-323");
static_assert(to_string(0.) == "0");
static_assert(to_string(-0.f) == "-0");
static_assert(to_string(std::numeric_limits<double>::infinity()) == "inf");
static_assert(to_string(-std::numeric_limits<float>::infinity()) == "-inf");
static_assert(to_string(-std::numeric_limits<double>::quiet_NaN()) == "-nan");
static_assert(to_string(std::numeric_limits<float>::quiet_NaN()) == "nan");
}  // namespace zmij_constexpr
#endif

int main() {
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 6.62607015e-34);
  puts(buf);
}
