// Integer-formatting benchmarks for https://github.com/vitaut/zmij/,
// comparing zmij against fmt (fmt::format_int, its fastest integer path).
//
// Three workloads:
//  - json_u8/i8_*: serialize an array of byte-valued numbers ([0, 255] as
//    u32/u64, [-128, 127] as i32/i64) as a JSON array "[ 123, 124, 0, -14 ]".
//  - json_twitter_mk2_*: the same JSON-array serialization for a u64 stream
//    whose digit-count sequence comes from a second-order Markov chain
//    fitted on the integers of simdjson's twitter.json (see twitter_digits).
//    The zmij32 variant dispatches to the u32 writer for values that fit,
//    a user-side wrapper over the public API.
//  - log_uniform_*: plain conversion of values log-uniformly distributed
//    over the full range of the type (the same number of entries per digit
//    count) for i32, i64 and i128.
//
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <benchmark/benchmark.h>
#include <stdint.h>  // uint64_t
#include <string.h>  // memcpy

#include <algorithm>  // std::shuffle
#include <random>     // std::mt19937_64
#include <vector>     // std::vector

#include "fmt/format.h"
#include "zmij.h"

namespace {

#if ZMIJ_USE_INT128
using wide_uint = unsigned __int128;
#else
using wide_uint = uint64_t;
#endif

// Stream lengths. Long enough that the branch predictor cannot memorize
// the per-iteration branch-outcome sequence across benchmark iterations
// (on Zen 5 a replayed 64k-item stream is still ~96% memorized; miss
// rates only saturate past ~128k items).
constexpr size_t json_array_size = 1 << 20;
constexpr size_t entries_per_digit_count = 16384;

template <typename U> auto random_bits(std::mt19937_64& gen) -> U {
  if constexpr (sizeof(U) > 8)
    return (U(gen()) << 64) | gen();
  else
    return U(gen());
}

// Byte-valued numbers equidistributed over [lo, hi], stored as T.
template <typename T> auto byte_range_data(int lo, int hi) -> std::vector<T> {
  std::mt19937_64 gen(7);
  std::uniform_int_distribution<int> dist(lo, hi);
  std::vector<T> data(json_array_size);
  for (auto& value : data) value = T(dist(gen));
  return data;
}

// Log-uniform over the full range of T: the same number of entries for each
// magnitude digit count, uniform within a digit count, random sign for
// signed types.
template <typename T> auto log_uniform_data() -> std::vector<T> {
  constexpr bool is_signed = T(-1) < T(0);
  const wide_uint max_magnitude =
      is_signed ? (wide_uint(1) << (sizeof(T) * 8 - 1)) - 1
      : sizeof(T) == sizeof(wide_uint)
          ? ~wide_uint(0)
          : (wide_uint(1) << (sizeof(T) * 8 % (sizeof(wide_uint) * 8))) - 1;
  std::mt19937_64 gen(42);
  std::vector<T> data;
  wide_uint lo = 0, hi = 9;
  for (;;) {
    if (hi > max_magnitude) hi = max_magnitude;
    wide_uint span = hi - lo + 1;  // 0 only for the full wide_uint range
    for (size_t i = 0; i != entries_per_digit_count; ++i) {
      wide_uint bits = random_bits<wide_uint>(gen);
      T value = T(lo + (span != 0 ? bits % span : bits));
      if (is_signed && (gen() & 1) != 0) value = T(0) - value;
      data.push_back(value);
    }
    if (hi == max_magnitude) break;
    lo = hi + 1;
    hi = hi * 10 + 9;
  }
  std::shuffle(data.begin(), data.end(), gen);
  return data;
}

// Digit-count sequence of the 2108 integers in simdjson's twitter.json, in
// document order (status ids, user ids, counts, ...). Basis for value streams
// whose digit-count sequence has twitter's structure: order 0 draws i.i.d.
// from the marginal distribution, orders 1/2 from Markov chains fitted on the
// circular sequence (circularity guarantees every reachable context was
// observed). All three share the same marginal, so timing differences between
// them isolate the effect of sequence predictability on the writers.
const unsigned char twitter_digits[] = {
    18,9,10,3,3,1,3,4,1,1,9,1,1,18,9,1,2,2,3,1,4,5,18,8,2,2,2,3,4,3,2,4,5,5,2,2,18,2,2,3,
    3,3,3,3,3,3,3,2,1,8,1,2,18,2,2,3,3,3,3,3,3,3,3,18,18,18,9,9,1,2,4,3,2,3,5,5,1,1,9,1,
    2,18,9,1,2,2,2,4,4,2,4,5,6,18,9,3,3,2,4,5,5,2,2,2,1,9,1,2,18,9,3,3,1,4,5,5,18,8,3,3,
    2,4,5,5,4,4,1,2,18,2,2,3,3,4,3,3,3,3,3,4,1,2,2,8,1,2,18,2,2,3,3,4,3,3,3,3,3,18,18,10,
    3,3,1,2,4,1,1,18,9,3,3,1,4,5,5,1,1,18,18,9,10,2,2,1,3,5,3,1,1,9,1,2,18,10,3,3,1,3,4,18,
    10,10,3,3,1,4,4,1,1,10,1,2,1,1,10,1,2,10,2,2,18,10,1,1,1,1,4,1,1,18,10,3,4,1,1,3,18,10,3,3,
    1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,3,5,18,18,
    9,9,1,2,4,3,3,2,5,5,1,1,9,2,2,9,2,2,18,2,2,3,3,3,3,4,3,3,3,18,1,1,9,1,2,9,2,2,9,2,
    2,18,2,3,3,3,3,3,4,3,3,3,18,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,4,4,2,4,
    4,18,10,1,2,3,3,2,4,5,5,2,2,2,2,2,1,2,2,10,1,2,18,10,3,3,1,4,5,4,1,1,18,10,3,4,1,1,3,18,
    10,3,3,1,1,3,2,1,2,1,10,1,2,18,9,4,4,2,4,5,6,18,10,1,2,3,3,4,4,2,3,5,5,1,1,3,3,1,1,3,
    3,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,
    1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,
    3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,
    1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,9,1,2,3,3,2,4,5,
    6,18,9,1,2,2,2,6,2,4,2,5,5,3,3,3,1,9,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,
    2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,
    10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,10,3,3,1,3,3,1,1,3,3,10,1,2,18,
    10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,2,1,1,5,1,1,18,10,3,3,1,1,3,18,10,3,3,
    1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,
    3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,2,1,2,3,
    18,9,3,3,1,3,4,2,1,2,2,2,1,2,2,9,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,
    10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,
    2,18,10,3,3,1,3,4,1,1,18,10,1,2,1,2,1,1,6,1,1,2,3,2,2,18,3,3,3,3,3,3,3,3,3,3,18,10,2,3,
    1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,9,3,3,1,4,5,5,1,1,18,8,1,2,2,3,3,3,2,5,5,6,18,
    10,1,2,3,3,1,2,3,1,1,1,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,
    1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,
    3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,2,5,4,18,10,1,2,3,3,2,4,5,5,1,1,1,1,
    10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,
    2,1,10,1,2,18,10,4,4,2,1,5,1,1,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,
    1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,8,2,
    2,3,3,1,4,5,6,18,18,9,8,4,3,3,5,5,5,1,1,2,2,2,3,9,2,2,1,1,2,2,3,3,8,1,2,9,2,2,18,10,
    3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,4,5,1,1,3,3,18,18,9,9,1,2,3,3,1,
    2,5,4,1,1,9,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,
    1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,2,1,1,5,1,1,
    2,2,18,2,3,3,3,3,3,3,3,3,3,18,10,10,1,2,3,3,1,4,4,1,1,3,3,10,1,2,18,10,1,2,4,4,1,4,5,4,
    1,1,18,10,1,1,1,1,4,1,1,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,
    10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,
    3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,3,4,1,1,9,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,
    3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,
    1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,
    3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,2,3,1,1,3,
    18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,18,9,10,3,3,1,4,4,1,1,9,1,2,18,10,3,4,1,1,3,18,10,3,3,1,
    1,3,2,1,2,1,10,1,2,18,18,9,10,2,2,1,5,5,5,1,1,9,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,
    2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,
    2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,3,1,1,3,18,10,3,3,1,
    1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,
    3,1,1,3,2,1,2,1,10,1,2,18,10,5,5,2,4,5,5,1,1,2,2,2,2,2,2,18,9,4,2,2,1,5,5,1,1,3,3,18,
    10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,2,18,10,3,4,1,1,3,18,10,3,3,1,1,3,2,1,2,1,10,1,
    2,18,18,10,10,2,2,1,3,3,1,1,10,1,2,18,10,1,2,3,3,1,1,5,5,1,1,3,3,18,9,3,3,1,4,5,18,10,4,4,
    1,1,4,1,1,1,1,10,1,2,18,10,1,2,1,3,5,3,18,10,1,2,3,3,2,4,5,5,2,2,2,2,2,1,2,2,10,1,2,18,
    9,3,2,1,2,5,18,8,4,3,2,3,5,5,1,1,2,2,18,2,3,3,3,3,3,3,3,3,3,1,1,2,3,8,1,2,18,3,3,3,
    3,3,3,3,3,3,3,18,18,10,1,2,3,3,3,3,2,3,4,1,1,2,2,2,2,18,3,1
};

template <int order> auto twitter_markov_data() -> std::vector<uint64_t> {
  constexpr int num_states = 21;  // digit counts 1..20, indexed directly
  const int n = int(sizeof(twitter_digits));
  std::vector<uint32_t> c0(num_states, 0);
  std::vector<uint32_t> c1(num_states * num_states, 0);
  std::vector<uint32_t> c2(num_states * num_states * num_states, 0);
  for (int i = 0; i < n; ++i) {
    int a = twitter_digits[i];
    int b = twitter_digits[(i + 1) % n];
    int c = twitter_digits[(i + 2) % n];
    ++c0[a];
    ++c1[a * num_states + b];
    ++c2[(a * num_states + b) * num_states + c];
  }
  uint64_t pow10[20] = {1};
  for (int i = 1; i < 20; ++i) pow10[i] = pow10[i - 1] * 10;

  std::mt19937_64 gen(2108);
  auto pick = [&gen](const uint32_t* row, uint32_t total) {
    uint64_t r = gen() % total;
    for (int s = 0;; ++s) {
      if (r < row[s]) return s;
      r -= row[s];
    }
  };
  std::vector<uint64_t> data(json_array_size);
  int a = twitter_digits[0], b = twitter_digits[1];
  for (auto& value : data) {
    int d = order == 0 ? pick(c0.data(), uint32_t(n))
            : order == 1
                ? pick(&c1[b * num_states], c0[b])
                : pick(&c2[(a * num_states + b) * num_states],
                       c1[a * num_states + b]);
    a = b;
    b = d;
    uint64_t lo = d == 1 ? 0 : pow10[d - 1];
    value = lo + gen() % (pow10[d] - lo);
  }
  return data;
}

struct zmij_write {
  template <typename T> auto operator()(char* out, T value) const -> char* {
    // n = 64 >= every integer buffer size, so this takes the direct path.
    return zmij::write(out, 64, value);
  }
};

// Like zmij_write, but routes 64-bit values that fit in 32 bits to the u32
// kernel. The range check is a data-dependent but (on realistic streams)
// highly predictable branch; values on the fast path skip the u64 kernel's
// fixed 16-digit body.
struct zmij_write32 {
  auto operator()(char* out, uint64_t value) const -> char* {
    if (value <= UINT32_MAX) return zmij::write(out, 64, uint32_t(value));
    return zmij::write(out, 64, value);
  }
  auto operator()(char* out, int64_t value) const -> char* {
    if (value == int64_t(int32_t(value)))
      return zmij::write(out, 64, int32_t(value));
    return zmij::write(out, 64, value);
  }
};

// For 32/64-bit we call fmt::detail::format_decimal directly, the same
// digit-writing primitive fmt::format_int wraps. It writes straight into the
// output buffer with no intermediate copy, but requires the digit count up
// front (count_digits), since it fills back-to-front within [out, out + n).
// __int128 has no format_decimal, so 128-bit goes through fmt::format_to.
struct fmt_write {
  template <typename T> auto operator()(char* out, T value) const -> char* {
    if constexpr (sizeof(T) > 8) {
      return fmt::format_to(out, "{}", value);
    } else if constexpr (T(-1) < T(0)) {
      auto abs_value = static_cast<fmt::detail::uint32_or_64_or_128_t<T>>(value);
      *out = '-';
      if (value < 0) {
        ++out;
        abs_value = 0 - abs_value;
      }
      return (*this)(out, abs_value);
    } else {
      int num_digits = fmt::detail::count_digits(value);
      fmt::detail::format_decimal(out, value, num_digits);
      return out + num_digits;
    }
  }
};

template <typename T, typename Write>
void json_array(benchmark::State& state, const std::vector<T>& data,
                Write write) {
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

template <typename T, typename Write>
void convert(benchmark::State& state, const std::vector<T>& data,
             Write write) {
  char buffer[64];
  for (auto _ : state) {
    for (T value : data) benchmark::DoNotOptimize(write(buffer, value));
  }
  state.SetItemsProcessed(int64_t(state.iterations()) * data.size());
}

void json_u8_as_u32_zmij(benchmark::State& state) {
  static const auto data = byte_range_data<uint32_t>(0, 255);
  json_array(state, data, zmij_write());
}
void json_u8_as_u32_fmt(benchmark::State& state) {
  static const auto data = byte_range_data<uint32_t>(0, 255);
  json_array(state, data, fmt_write());
}
void json_i8_as_i32_zmij(benchmark::State& state) {
  static const auto data = byte_range_data<int32_t>(-128, 127);
  json_array(state, data, zmij_write());
}
void json_i8_as_i32_fmt(benchmark::State& state) {
  static const auto data = byte_range_data<int32_t>(-128, 127);
  json_array(state, data, fmt_write());
}
void json_twitter_iid_u64_zmij32(benchmark::State& state) {
  static const auto data = twitter_markov_data<0>();
  json_array(state, data, zmij_write32());
}
void json_twitter_mk1_u64_zmij32(benchmark::State& state) {
  static const auto data = twitter_markov_data<1>();
  json_array(state, data, zmij_write32());
}
void json_twitter_mk2_u64_zmij(benchmark::State& state) {
  static const auto data = twitter_markov_data<2>();
  json_array(state, data, zmij_write());
}
void json_twitter_mk2_u64_fmt(benchmark::State& state) {
  static const auto data = twitter_markov_data<2>();
  json_array(state, data, fmt_write());
}
void json_twitter_mk2_u64_zmij32(benchmark::State& state) {
  static const auto data = twitter_markov_data<2>();
  json_array(state, data, zmij_write32());
}
BENCHMARK(json_twitter_iid_u64_zmij32);
BENCHMARK(json_twitter_mk1_u64_zmij32);
BENCHMARK(json_twitter_mk2_u64_zmij);
BENCHMARK(json_twitter_mk2_u64_zmij32);
BENCHMARK(json_twitter_mk2_u64_fmt);


void json_u8_as_u64_zmij(benchmark::State& state) {
  static const auto data = byte_range_data<uint64_t>(0, 255);
  json_array(state, data, zmij_write());
}
void json_u8_as_u64_fmt(benchmark::State& state) {
  static const auto data = byte_range_data<uint64_t>(0, 255);
  json_array(state, data, fmt_write());
}
void json_i8_as_i64_zmij(benchmark::State& state) {
  static const auto data = byte_range_data<int64_t>(-128, 127);
  json_array(state, data, zmij_write());
}
void json_i8_as_i64_fmt(benchmark::State& state) {
  static const auto data = byte_range_data<int64_t>(-128, 127);
  json_array(state, data, fmt_write());
}
BENCHMARK(json_u8_as_u32_zmij);
BENCHMARK(json_u8_as_u32_fmt);
BENCHMARK(json_i8_as_i32_zmij);
BENCHMARK(json_i8_as_i32_fmt);
BENCHMARK(json_u8_as_u64_zmij);
BENCHMARK(json_u8_as_u64_fmt);
BENCHMARK(json_i8_as_i64_zmij);
BENCHMARK(json_i8_as_i64_fmt);

void log_uniform_u32_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<uint32_t>();
  convert(state, data, zmij_write());
}
void log_uniform_u32_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<uint32_t>();
  convert(state, data, fmt_write());
}
void log_uniform_u64_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<uint64_t>();
  convert(state, data, zmij_write());
}
void log_uniform_u64_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<uint64_t>();
  convert(state, data, fmt_write());
}
BENCHMARK(log_uniform_u32_zmij);
BENCHMARK(log_uniform_u32_fmt);
BENCHMARK(log_uniform_u64_zmij);
BENCHMARK(log_uniform_u64_fmt);

void log_uniform_i32_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<int32_t>();
  convert(state, data, zmij_write());
}
void log_uniform_i32_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<int32_t>();
  convert(state, data, fmt_write());
}
void log_uniform_i64_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<int64_t>();
  convert(state, data, zmij_write());
}
void log_uniform_i64_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<int64_t>();
  convert(state, data, fmt_write());
}
BENCHMARK(log_uniform_i32_zmij);
BENCHMARK(log_uniform_i32_fmt);
BENCHMARK(log_uniform_i64_zmij);
BENCHMARK(log_uniform_i64_fmt);

#if ZMIJ_USE_INT128
void log_uniform_u128_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<unsigned __int128>();
  convert(state, data, zmij_write());
}
void log_uniform_u128_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<unsigned __int128>();
  convert(state, data, fmt_write());
}
BENCHMARK(log_uniform_u128_zmij);
BENCHMARK(log_uniform_u128_fmt);
void log_uniform_i128_zmij(benchmark::State& state) {
  static const auto data = log_uniform_data<__int128>();
  convert(state, data, zmij_write());
}
void log_uniform_i128_fmt(benchmark::State& state) {
  static const auto data = log_uniform_data<__int128>();
  convert(state, data, fmt_write());
}
BENCHMARK(log_uniform_i128_zmij);
BENCHMARK(log_uniform_i128_fmt);
#endif

}  // namespace

BENCHMARK_MAIN();
