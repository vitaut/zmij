#include <stdio.h>

#define ZMIJ_ENABLE_COMPILETIME_EVALUATION
#define ZMIJ_HEADER_ONLY  // implied by compiletime evaluation

#include "zmij.h"

#ifdef ZMIJ_ENABLE_COMPILETIME_EVALUATION
#  include <array>
#  include <string_view>

namespace zmij_constexpr {
constexpr auto to_string(
    double value,
    char* buf = std::array<char, zmij::double_buffer_size>{}.data())
    -> std::string_view {
  const auto end = zmij::detail::write(value, buf);
  return {buf, static_cast<size_t>(end - buf)};
}
}  // namespace zmij_constexpr
#endif

int main() {
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 6.62607015e-34);
  puts(buf);
}
