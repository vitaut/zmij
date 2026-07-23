// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

#ifndef ZMIJ_TO_CHARS_H_
#define ZMIJ_TO_CHARS_H_

#include <stddef.h>      // size_t
#include <string.h>      // memcpy
#include <system_error>  // std::errc

#include "zmij.h"

namespace zmij {

// Like std::to_chars_result, but available without C++17.
struct to_chars_result {
  char* ptr;
  std::errc ec;
};

/// Writes the shortest correctly rounded decimal representation of `value` to
/// [`first`, `last`) without a null terminator, like std::to_chars. On success
/// returns {ptr, std::errc()} with ptr past the last character written; if the
/// output is too small returns {last, std::errc::value_too_large} and writes
/// nothing.
inline auto to_chars(char* first, char* last, float value) -> to_chars_result {
  char buffer[float_buffer_size];
  size_t size = size_t(detail::write(value, buffer) - buffer);
  if (size > size_t(last - first)) return {last, std::errc::value_too_large};
  memcpy(first, buffer, size);
  return {first + size, std::errc()};
}
inline auto to_chars(char* first, char* last, double value) -> to_chars_result {
  char buffer[double_buffer_size];
  size_t size = size_t(detail::write(value, buffer) - buffer);
  if (size > size_t(last - first)) return {last, std::errc::value_too_large};
  memcpy(first, buffer, size);
  return {first + size, std::errc()};
}

}  // namespace zmij

#endif  // ZMIJ_TO_CHARS_H_
