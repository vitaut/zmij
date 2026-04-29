// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"
#include "dragonbox/dragonbox_to_chars.h"
#include "zmij.h"

namespace zmij {
int dtoa(...);
int to_string(...);
int write(...);
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

auto dtoa_dragonbox(double value, char* buffer) -> char* {
  return jkj::dragonbox::to_chars(value, buffer, jkj::dragonbox::policy::cache::full);
}

REGISTER_DTOA(dragonbox);
