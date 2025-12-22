// A double-to-string conversion algorithm based on Schubfach.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE) or alternatively
// the Boost Software License, Version 1.0.

namespace zmij {
namespace detail {
template <typename Float> void to_string(Float value, char* buffer) noexcept;
}  // namespace detail

constexpr int buffer_size = 25;

/// Writes the shortest correctly rounded decimal representation of `value` to
/// `buffer`. `buffer` should point to a buffer of size `buffer_size` or larger.
inline void dtoa(double value, char* buffer) noexcept {
  return detail::to_string(value, buffer);
}

}  // namespace zmij
