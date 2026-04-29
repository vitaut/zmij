// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"
#include "dragonbox/dragonbox_to_chars.h"
#include "zmij.h"

auto ftoa_zmij(float value, char* buffer) -> char* {
  using result = decltype(zmij::write(buffer, zmij::float_buffer_size, value));
  // `reinterpret_cast`s here keep both branches well-formed regardless of
  // whether `zmij::write` returns `char*` or an integer count, which has
  // varied across the project's history. The cast is a no-op in the live
  // branch and never runs in the discarded one.
  if constexpr (std::is_same_v<result, char*>)
    return reinterpret_cast<char*>(
        zmij::write(buffer, zmij::float_buffer_size, value));
  else if constexpr (!std::is_same_v<result, int>)
    return buffer + reinterpret_cast<size_t>(
                        zmij::write(buffer, zmij::float_buffer_size, value));
  return nullptr;
}

REGISTER_FTOA(zmij);

auto ftoa_dragonbox(float value, char* buffer) -> char* {
  return jkj::dragonbox::to_chars(value, buffer,
                                  jkj::dragonbox::policy::cache::full);
}

REGISTER_FTOA(dragonbox);
