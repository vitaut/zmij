// Integer-formatting benchmarks for https://github.com/vitaut/zmij/,
// comparing zmij against fmt::format_int.
//
// Three workloads:
//  - json_u8/i8_*: serialize an array of byte-valued numbers ([0, 255] as
//    u32/u64, [-128, 127] as i32/i64) as a JSON array "[ 123, 124, 0, -14 ]".
//  - json_twitter_mk2_*: the same JSON-array serialization for a u64 stream
//    whose digit-count sequence comes from a second-order Markov chain
//    fitted on the integers of simdjson's twitter.json (see
//    twitter_digit_counts).
//    The zmij32 variant dispatches to the u32 writer for values that fit,
//    a user-side wrapper over the public API.
//  - log_uniform_*: plain conversion of values log-uniformly distributed
//    over the full range of the type (the same number of samples per digit
//    count) for u32/i32, u64/i64 and u128/i128.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <benchmark/benchmark.h>
#include <limits.h>  // CHAR_BIT
#include <stdint.h>  // uint64_t

#include <algorithm>    // std::shuffle
#include <random>       // std::mt19937_64
#include <type_traits>  // std::conditional
#include <vector>       // std::vector

#include "fmt/format.h"
#include "zmij.h"

namespace {

using wide_uint =
    typename std::conditional<ZMIJ_USE_INT128, zmij::detail::uint128_t,
                              uint64_t>::type;

// Long enough to prevent replay across benchmark iterations from reducing
// branch-miss rates. On Zen 5, rates reached steady state only beyond ~128k
// items.
constexpr size_t json_array_size = 1 << 20;
constexpr size_t samples_per_digit_count = 16384;

template <typename U> auto random_bits(std::mt19937_64& gen) -> U {
  if constexpr (sizeof(U) > 8) return (U(gen()) << 64) | gen();
  return U(gen());
}

// Returns json_array_size values sampled uniformly from [lo, hi] and
// converted to T.
template <typename T> auto byte_range_data(int lo, int hi) -> std::vector<T> {
  std::mt19937_64 gen(7);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<T> data(json_array_size);
  for (auto& value : data) value = T(dist(gen));
  return data;
}

// Returns a shuffled log-uniform sample of T: samples_per_digit_count
// pseudorandom values per decimal magnitude length, with random signs for
// signed T.
template <typename T> auto log_uniform_data() -> std::vector<T> {
  constexpr bool is_signed = T(-1) < T(0);
  constexpr wide_uint max_value =
      ~wide_uint(0) >>
      ((sizeof(wide_uint) - sizeof(T)) * CHAR_BIT + is_signed);
  std::mt19937_64 gen(42);
  std::vector<T> data;
  wide_uint lo = 0, hi = 9;
  for (;;) {
    wide_uint span = hi - lo + 1;
    for (size_t i = 0; i != samples_per_digit_count; ++i) {
      T value = T(lo + random_bits<wide_uint>(gen) % span);
      if (is_signed && (gen() & 1) != 0) value = T(0) - value;
      data.push_back(value);
    }
    if (hi == max_value) break;
    lo = hi + 1;
    // Clamp before multiplying: for a type spanning all of wide_uint the
    // top digit count is partial and hi * 10 + 9 would wrap.
    hi = hi > (max_value - 9) / 10 ? max_value : hi * 10 + 9;
  }
  std::shuffle(data.begin(), data.end(), gen);
  return data;
}

// Decimal digit counts of the 2,108 integers in simdjson's twitter.json, in
// document order. Used to generate streams from the marginal distribution
// (order 0) or first- and second-order Markov chains fitted to the circular
// sequence. Circularity gives every reachable context a successor. All three
// models have the same stationary marginal, so timing differences isolate
// the effect of sequence predictability.
const unsigned char twitter_digit_counts[] = {
    18, 9,  10, 3,  3,  1,  3,  4,  1,  1,  9,  1,  1,  18, 9,  1,  2,  2,  3,
    1,  4,  5,  18, 8,  2,  2,  2,  3,  4,  3,  2,  4,  5,  5,  2,  2,  18, 2,
    2,  3,  3,  3,  3,  3,  3,  3,  3,  2,  1,  8,  1,  2,  18, 2,  2,  3,  3,
    3,  3,  3,  3,  3,  3,  18, 18, 18, 9,  9,  1,  2,  4,  3,  2,  3,  5,  5,
    1,  1,  9,  1,  2,  18, 9,  1,  2,  2,  2,  4,  4,  2,  4,  5,  6,  18, 9,
    3,  3,  2,  4,  5,  5,  2,  2,  2,  1,  9,  1,  2,  18, 9,  3,  3,  1,  4,
    5,  5,  18, 8,  3,  3,  2,  4,  5,  5,  4,  4,  1,  2,  18, 2,  2,  3,  3,
    4,  3,  3,  3,  3,  3,  4,  1,  2,  2,  8,  1,  2,  18, 2,  2,  3,  3,  4,
    3,  3,  3,  3,  3,  18, 18, 10, 3,  3,  1,  2,  4,  1,  1,  18, 9,  3,  3,
    1,  4,  5,  5,  1,  1,  18, 18, 9,  10, 2,  2,  1,  3,  5,  3,  1,  1,  9,
    1,  2,  18, 10, 3,  3,  1,  3,  4,  18, 10, 10, 3,  3,  1,  4,  4,  1,  1,
    10, 1,  2,  1,  1,  10, 1,  2,  10, 2,  2,  18, 10, 1,  1,  1,  1,  4,  1,
    1,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,
    10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,
    2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  3,  5,  18, 18, 9,  9,  1,  2,  4,
    3,  3,  2,  5,  5,  1,  1,  9,  2,  2,  9,  2,  2,  18, 2,  2,  3,  3,  3,
    3,  4,  3,  3,  3,  18, 1,  1,  9,  1,  2,  9,  2,  2,  9,  2,  2,  18, 2,
    3,  3,  3,  3,  3,  4,  3,  3,  3,  18, 18, 10, 3,  3,  1,  1,  3,  18, 10,
    3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 4,  4,  2,  4,  4,
    18, 10, 1,  2,  3,  3,  2,  4,  5,  5,  2,  2,  2,  2,  2,  1,  2,  2,  10,
    1,  2,  18, 10, 3,  3,  1,  4,  5,  4,  1,  1,  18, 10, 3,  4,  1,  1,  3,
    18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 9,  4,  4,  2,
    4,  5,  6,  18, 10, 1,  2,  3,  3,  4,  4,  2,  3,  5,  5,  1,  1,  3,  3,
    1,  1,  3,  3,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,
    1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,  18, 10, 3,
    3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18,
    10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 2,  3,  1,  1,
    3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,
    1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10,
    3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,
    18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10,
    1,  2,  18, 9,  1,  2,  3,  3,  2,  4,  5,  6,  18, 9,  1,  2,  2,  2,  6,
    2,  4,  2,  5,  5,  3,  3,  3,  1,  9,  1,  2,  18, 10, 2,  3,  1,  1,  3,
    18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,
    1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 2,
    3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18,
    10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,
    2,  18, 10, 10, 3,  3,  1,  3,  3,  1,  1,  3,  3,  10, 1,  2,  18, 10, 3,
    4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18,
    10, 2,  2,  1,  1,  5,  1,  1,  18, 10, 3,  3,  1,  1,  3,  18, 10, 3,  3,
    1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,  18, 10,
    3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,
    18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,
    1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 2,
    2,  1,  2,  3,  18, 9,  3,  3,  1,  3,  4,  2,  1,  2,  2,  2,  1,  2,  2,
    9,  1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,
    2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,
    2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,
    1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  3,  4,  1,  1,  18,
    10, 1,  2,  1,  2,  1,  1,  6,  1,  1,  2,  3,  2,  2,  18, 3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  18, 10, 2,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,
    3,  2,  1,  2,  1,  10, 1,  2,  18, 9,  3,  3,  1,  4,  5,  5,  1,  1,  18,
    8,  1,  2,  2,  3,  3,  3,  2,  5,  5,  6,  18, 10, 1,  2,  3,  3,  1,  2,
    3,  1,  1,  1,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,
    1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10,
    3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,
    18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,
    1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,
    3,  2,  5,  4,  18, 10, 1,  2,  3,  3,  2,  4,  5,  5,  1,  1,  1,  1,  10,
    1,  2,  18, 10, 3,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,
    1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,
    1,  2,  1,  10, 1,  2,  18, 10, 4,  4,  2,  1,  5,  1,  1,  18, 10, 2,  3,
    1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10,
    3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,
    18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10,
    1,  2,  18, 8,  2,  2,  3,  3,  1,  4,  5,  6,  18, 18, 9,  8,  4,  3,  3,
    5,  5,  5,  1,  1,  2,  2,  2,  3,  9,  2,  2,  1,  1,  2,  2,  3,  3,  8,
    1,  2,  9,  2,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,
    2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  4,  5,  1,  1,  3,  3,
    18, 18, 9,  9,  1,  2,  3,  3,  1,  2,  5,  4,  1,  1,  9,  1,  2,  18, 10,
    2,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,
    18, 10, 2,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10,
    1,  2,  18, 10, 3,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,
    1,  10, 1,  2,  18, 10, 2,  2,  1,  1,  5,  1,  1,  2,  2,  18, 2,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  18, 10, 10, 1,  2,  3,  3,  1,  4,  4,  1,  1,
    3,  3,  10, 1,  2,  18, 10, 1,  2,  4,  4,  1,  4,  5,  4,  1,  1,  18, 10,
    1,  1,  1,  1,  4,  1,  1,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,
    1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,
    3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18,
    10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,
    3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,
    1,  3,  4,  1,  1,  9,  1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,
    1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 2,  3,  1,  1,  3,  18, 10,
    3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,
    18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 2,  3,  1,
    1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,
    4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18,
    10, 2,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,
    2,  18, 10, 2,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,
    10, 1,  2,  18, 18, 9,  10, 3,  3,  1,  4,  4,  1,  1,  9,  1,  2,  18, 10,
    3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,
    18, 18, 9,  10, 2,  2,  1,  5,  5,  5,  1,  1,  9,  1,  2,  18, 10, 3,  4,
    1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10,
    3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10, 1,  2,
    18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10,
    1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,
    1,  10, 1,  2,  18, 10, 3,  3,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,
    1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,
    3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,
    1,  1,  3,  2,  1,  2,  1,  10, 1,  2,  18, 10, 5,  5,  2,  4,  5,  5,  1,
    1,  2,  2,  2,  2,  2,  2,  18, 9,  4,  2,  2,  1,  5,  5,  1,  1,  3,  3,
    18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,  1,  10,
    1,  2,  18, 10, 3,  4,  1,  1,  3,  18, 10, 3,  3,  1,  1,  3,  2,  1,  2,
    1,  10, 1,  2,  18, 18, 10, 10, 2,  2,  1,  3,  3,  1,  1,  10, 1,  2,  18,
    10, 1,  2,  3,  3,  1,  1,  5,  5,  1,  1,  3,  3,  18, 9,  3,  3,  1,  4,
    5,  18, 10, 4,  4,  1,  1,  4,  1,  1,  1,  1,  10, 1,  2,  18, 10, 1,  2,
    1,  3,  5,  3,  18, 10, 1,  2,  3,  3,  2,  4,  5,  5,  2,  2,  2,  2,  2,
    1,  2,  2,  10, 1,  2,  18, 9,  3,  2,  1,  2,  5,  18, 8,  4,  3,  2,  3,
    5,  5,  1,  1,  2,  2,  18, 2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  1,  1,
    2,  3,  8,  1,  2,  18, 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  18, 18, 10,
    1,  2,  3,  3,  3,  3,  2,  3,  4,  1,  1,  2,  2,  2,  2,  18, 3,  1};

template <int order> auto twitter_markov_data() -> std::vector<uint64_t> {
  static_assert(order >= 0 && order <= 2);
  constexpr int num_states = 21;  // digit counts 1..20, indexed directly
  constexpr int n = int(sizeof(twitter_digit_counts));
  auto context = [](int a, int b) {
    return (order > 1 ? a * num_states : 0) + (order > 0 ? b : 0);
  };
  std::vector<std::vector<int>> successors(num_states * num_states);
  for (int i = 0; i < n; ++i) {
    int a = twitter_digit_counts[i], b = twitter_digit_counts[(i + 1) % n];
    successors[context(a, b)].push_back(twitter_digit_counts[(i + 2) % n]);
  }
  // Sorting makes each stream depend only on the transition counts.
  for (auto& s : successors) std::sort(s.begin(), s.end());
  uint64_t pow10[20] = {1};
  for (int i = 1; i < 20; ++i) pow10[i] = pow10[i - 1] * 10;

  std::mt19937_64 gen(2108);
  std::vector<uint64_t> data(json_array_size);
  int a = twitter_digit_counts[0], b = twitter_digit_counts[1];
  for (auto& value : data) {
    const auto& next = successors[context(a, b)];
    int d = next[gen() % next.size()];
    a = b;
    b = d;
    uint64_t lo = d == 1 ? 0 : pow10[d - 1];
    uint64_t span = d == 20 ? UINT64_MAX - lo + 1 : pow10[d] - lo;
    value = lo + gen() % span;
  }
  return data;
}

struct zmij_writer {
  template <typename T> auto operator()(char* out, T value) const -> char* {
    // n = 64 >= every integer buffer size, so this takes the direct path.
    return zmij::write(out, 64, value);
  }
};

// Like zmij_writer, but routes u64 values that fit in u32 to the u32 overload.
// The range check is highly predictable on realistic streams.
struct zmij_writer32 {
  auto operator()(char* out, uint64_t value) const -> char* {
    return value <= UINT32_MAX ? zmij::write(out, 64, uint32_t(value))
                               : zmij::write(out, 64, value);
  }
};

// Call fmt::detail::format_decimal directly, the same
// digit-writing primitive fmt::format_int wraps. It writes straight into the
// output buffer with no intermediate copy, but requires the digit count up
// front (count_digits), since it fills back-to-front within [out, out + n).
struct fmt_writer {
  template <typename T> auto operator()(char* out, T value) const -> char* {
    auto abs_value = static_cast<fmt::detail::uint32_or_64_or_128_t<T>>(value);
    if constexpr (T(-1) < T(0)) {
      *out = '-';
      if (value < 0) {
        ++out;
        abs_value = 0 - abs_value;
      }
    }
    return fmt::detail::format_decimal(out, abs_value,
                                       fmt::detail::count_digits(abs_value));
  }
};

template <typename T, typename Writer>
void benchmark_json_array(benchmark::State& state, const std::vector<T>& data,
                          Writer write) {
  // Sized for up to 20-digit entries plus separators, with slack past the
  // end for the writers' fixed-size block stores.
  std::vector<char> out(data.size() * 26 + 64);
  char* end = out.data();
  for (auto _ : state) {
    char* p = out.data();
    // Branch-free separators: each iteration first steps past the '[' or the
    // previous element's ',', and writes its own ',' without advancing so the
    // next iteration -- or the closing " ]" -- can overwrite it.
    *p = '[';
    for (size_t i = 0; i != data.size(); ++i) {
      ++p;
      *p++ = ' ';
      p = write(p, data[i]);
      *p = ',';
    }
    *p++ = ' ';
    *p++ = ']';
    end = p;
    benchmark::DoNotOptimize(p);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(int64_t(state.iterations()) * data.size());
  state.SetBytesProcessed(int64_t(state.iterations()) * (end - out.data()));
}

template <typename T, typename Writer>
void benchmark_json_u8(benchmark::State& state) {
  static const auto data = byte_range_data<T>(0, 255);
  benchmark_json_array(state, data, Writer());
}

template <typename T, typename Writer>
void benchmark_json_i8(benchmark::State& state) {
  static const auto data = byte_range_data<T>(-128, 127);
  benchmark_json_array(state, data, Writer());
}

template <int order, typename Writer>
void benchmark_json_twitter(benchmark::State& state) {
  static const auto data = twitter_markov_data<order>();
  benchmark_json_array(state, data, Writer());
}

template <typename T, typename Writer>
void benchmark_log_uniform(benchmark::State& state) {
  static const auto data = log_uniform_data<T>();
  Writer write;
  char buffer[64];
  for (auto _ : state) {
    for (T value : data) {
      benchmark::DoNotOptimize(write(buffer, value));
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(int64_t(state.iterations()) * data.size());
}

BENCHMARK_TEMPLATE(benchmark_json_twitter, 0, zmij_writer32)
    ->Name("json_twitter_iid_u64_zmij32");
BENCHMARK_TEMPLATE(benchmark_json_twitter, 1, zmij_writer32)
    ->Name("json_twitter_mk1_u64_zmij32");
BENCHMARK_TEMPLATE(benchmark_json_twitter, 2, zmij_writer)
    ->Name("json_twitter_mk2_u64_zmij");
BENCHMARK_TEMPLATE(benchmark_json_twitter, 2, zmij_writer32)
    ->Name("json_twitter_mk2_u64_zmij32");
BENCHMARK_TEMPLATE(benchmark_json_twitter, 2, fmt_writer)
    ->Name("json_twitter_mk2_u64_fmt");

BENCHMARK_TEMPLATE(benchmark_json_u8, uint32_t, zmij_writer)
    ->Name("json_u8_as_u32_zmij");
BENCHMARK_TEMPLATE(benchmark_json_u8, uint32_t, fmt_writer)
    ->Name("json_u8_as_u32_fmt");
BENCHMARK_TEMPLATE(benchmark_json_i8, int32_t, zmij_writer)
    ->Name("json_i8_as_i32_zmij");
BENCHMARK_TEMPLATE(benchmark_json_i8, int32_t, fmt_writer)
    ->Name("json_i8_as_i32_fmt");
BENCHMARK_TEMPLATE(benchmark_json_u8, uint64_t, zmij_writer)
    ->Name("json_u8_as_u64_zmij");
BENCHMARK_TEMPLATE(benchmark_json_u8, uint64_t, fmt_writer)
    ->Name("json_u8_as_u64_fmt");
BENCHMARK_TEMPLATE(benchmark_json_i8, int64_t, zmij_writer)
    ->Name("json_i8_as_i64_zmij");
BENCHMARK_TEMPLATE(benchmark_json_i8, int64_t, fmt_writer)
    ->Name("json_i8_as_i64_fmt");

BENCHMARK_TEMPLATE(benchmark_log_uniform, uint32_t, zmij_writer)
    ->Name("log_uniform_u32_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, uint32_t, fmt_writer)
    ->Name("log_uniform_u32_fmt");
BENCHMARK_TEMPLATE(benchmark_log_uniform, uint64_t, zmij_writer)
    ->Name("log_uniform_u64_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, uint64_t, fmt_writer)
    ->Name("log_uniform_u64_fmt");
BENCHMARK_TEMPLATE(benchmark_log_uniform, int32_t, zmij_writer)
    ->Name("log_uniform_i32_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, int32_t, fmt_writer)
    ->Name("log_uniform_i32_fmt");
BENCHMARK_TEMPLATE(benchmark_log_uniform, int64_t, zmij_writer)
    ->Name("log_uniform_i64_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, int64_t, fmt_writer)
    ->Name("log_uniform_i64_fmt");

#if ZMIJ_USE_INT128
BENCHMARK_TEMPLATE(benchmark_log_uniform, unsigned __int128, zmij_writer)
    ->Name("log_uniform_u128_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, unsigned __int128, fmt_writer)
    ->Name("log_uniform_u128_fmt");
BENCHMARK_TEMPLATE(benchmark_log_uniform, __int128, zmij_writer)
    ->Name("log_uniform_i128_zmij");
BENCHMARK_TEMPLATE(benchmark_log_uniform, __int128, fmt_writer)
    ->Name("log_uniform_i128_fmt");
#endif  // ZMIJ_USE_INT128

}  // namespace

BENCHMARK_MAIN();
