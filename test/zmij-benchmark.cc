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
  if constexpr (!std::is_same_v<decltype(zmij::write(buffer, 25, value)), int>)
    return buffer + zmij::write(buffer, 25, value);
  return nullptr;
}

REGISTER_METHOD(zmij);

auto dtoa_dragonbox(double value, char* buffer) -> char* {
  return jkj::dragonbox::to_chars(value, buffer, jkj::dragonbox::policy::cache::full);
}

REGISTER_METHOD(dragonbox);
