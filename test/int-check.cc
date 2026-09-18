// Exhaustively checks zmij's 32-bit integer formatting against fmt: every
// bit pattern is formatted both as int32_t and as uint32_t. Like float-check
// this is a standalone executable and not part of the ctest suite; run it
// manually. Takes a few seconds on a modern multi-core machine.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <thread>
#include <vector>

#include "fmt/format.h"
#include "zmij.h"

namespace {

std::atomic<uint64_t> num_errors(0);

template <typename T> void check(T value, char (&buffer)[64]) {
  char* end = zmij::write(buffer, sizeof(buffer), value);
  fmt::format_int expected(value);
  if (size_t(end - buffer) != expected.size() ||
      memcmp(buffer, expected.data(), expected.size()) != 0) {
    if (num_errors++ < 10) {
      fmt::print("FAIL: {} -> \"{}\", expected \"{}\"\n", value,
                 fmt::string_view(buffer, size_t(end - buffer)),
                 fmt::string_view(expected.data(), expected.size()));
    }
  }
}

void check_range(uint64_t begin, uint64_t end) {
  char buffer[64];
  for (uint64_t i = begin; i < end; ++i) {
    uint32_t bits = uint32_t(i);
    check(bits, buffer);
    check(int32_t(bits), buffer);
  }
}

}  // namespace

int main() {
  unsigned num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 1;
  constexpr uint64_t total = uint64_t(1) << 32;
  uint64_t chunk = total / num_threads + 1;

  std::vector<std::thread> threads;
  for (unsigned t = 0; t != num_threads; ++t) {
    uint64_t begin = t * chunk;
    uint64_t end = begin + chunk < total ? begin + chunk : total;
    threads.emplace_back(check_range, begin, end);
  }
  for (auto& thread : threads) thread.join();

  if (num_errors != 0) {
    fmt::print("{} mismatches\n", num_errors.load());
    return 1;
  }
  fmt::print("all {} values OK (int32 and uint32)\n", total);
  return 0;
}
