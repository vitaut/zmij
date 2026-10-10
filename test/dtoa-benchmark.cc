// Benchmark for https://github.com/vitaut/zmij/.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"
#include "dragonbox/dragonbox_to_chars.h"
#include "zmij.h"

namespace zmij {
int dtoa(...);
int to_string(...);
int write(...);
int write_scientific(...);
}  // namespace zmij

auto dtoa_zmij(double value, char* buffer) -> char* {
  if constexpr (!std::is_same_v<decltype(zmij::dtoa(value, buffer)), int>)
    zmij::dtoa(value, buffer);
  if constexpr (!std::is_same_v<decltype(zmij::to_string(value, buffer)), int>)
    zmij::to_string(value, buffer);
  using result = decltype(zmij::write(buffer, 34, value));
  // `reinterpret_cast`s here keep both branches well-formed regardless of
  // whether `zmij::write` returns `char*` or an integer count, which has
  // varied across the project's history. The cast is a no-op in the live
  // branch and never runs in the discarded one.
  if constexpr (std::is_same_v<result, char*>)
    return reinterpret_cast<char*>(zmij::write(buffer, 34, value));
  else if constexpr (!std::is_same_v<result, int>)
    return buffer + reinterpret_cast<size_t>(zmij::write(buffer, 34, value));
  return nullptr;
}

REGISTER_DTOA(zmij);

// Historical sources used by abtest.py may not have the precision API.
template <typename Float> static auto register_scientific() -> int {
  if constexpr (std::is_same_v<decltype(zmij::write_scientific(
                                  static_cast<char*>(nullptr), size_t(0),
                                  Float(), 0)),
                              char*>) {
    return register_precision_method_(
        "zmij/scientific", [](double value, char* buffer, int precision) -> char* {
          // Includes room for the sign and exponent at precision 100.
          return zmij::write_scientific(buffer, 128, Float(value), precision);
        },
        {6, 17, 18, 30, 50, 100});
  }
  return 0;
}

static int scientific_registered = register_scientific<double>();

auto dtoa_dragonbox(double value, char* buffer) -> char* {
  return jkj::dragonbox::to_chars(value, buffer,
                                  jkj::dragonbox::policy::cache::full);
}

REGISTER_DTOA(dragonbox);
