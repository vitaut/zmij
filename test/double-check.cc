// A verifier for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <stdint.h>  // uint32_t
#include <string.h>  // memcpy

#include <atomic>
#include <thread>
#include <vector>

#include "../zmij.cc"
#include "dragonbox/dragonbox.h"
#include "fmt/format.h"
#include "modular-search.h"

namespace {

auto format_duration(int seconds, bool show_seconds = true) -> std::string {
  int d = seconds / 86400, h = seconds / 3600 % 24, m = seconds / 60 % 60;
  std::string s;
  if (d) s += fmt::format("{}d ", d);
  if (d || h) s += fmt::format("{:02}h ", h);
  s += fmt::format("{:02}m", m);
  if (show_seconds) s += fmt::format(" {:02}s", seconds % 60);
  return s;
}

// clang-format off
const uint64_t pow10[] = {
                  1,
                 1'0,
                1'0'0,
               1'0'0'0,
              1'0'0'0'0,
             1'0'0'0'0'0,
            1'0'0'0'0'0'0,
           1'0'0'0'0'0'0'0,
          1'0'0'0'0'0'0'0'0,
         1'0'0'0'0'0'0'0'0'0,
        1'0'0'0'0'0'0'0'0'0'0,
       1'0'0'0'0'0'0'0'0'0'0'0,
      1'0'0'0'0'0'0'0'0'0'0'0'0,
     1'0'0'0'0'0'0'0'0'0'0'0'0'0,
    1'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
   1'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
  1'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0'0,
};
// clang-format on

using traits = float_traits<double>;

constexpr auto debias(int raw_exp) -> int {
  return raw_exp - traits::exp_offset;
}

inline auto verify(uint64_t bits, uint64_t bin_sig, int bin_exp, int raw_exp,
                   bool& has_errors) -> bool {
  to_decimal_result actual = to_decimal<double>(bin_sig, raw_exp, true, static_data);
  long long actual_sig = actual.sig * 10 + actual.last_digit;

  double value;
  memcpy(&value, &bits, sizeof(double));
  auto expected = jkj::dragonbox::to_decimal(value);

  uint32_t abbccddee = uint32_t(actual_sig / 100'000'000);
  uint32_t ffgghhii = uint32_t(actual_sig % 100'000'000);
  int num_zeros = 0;
  if (ffgghhii == 0)
    num_zeros = 16 - to_bcd8(abbccddee % 100'000'000).len;
  else
    num_zeros = 8 - to_bcd8(ffgghhii).len;
  if (num_zeros != 0) {
    expected.significand *= pow10[num_zeros];
    expected.exponent -= num_zeros;
  }

  if (actual_sig == expected.significand && actual.exp == expected.exponent)
    return true;

  if (has_errors) return false;
  has_errors = true;
  fmt::print(
      "Output mismatch for {} ({} * 2**{}): {} * 10**{} != {} * 10**{}\n",
      value, bin_sig, bin_exp, actual_sig, actual.exp, expected.significand,
      expected.exponent);
  return false;
}

auto is_pow10_exact_for_bin_exp(int bin_exp) -> bool {
  int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exact_begin = -0, exact_end = 55;
  static_assert(static_data.pow10_significands[exact_begin].hi ==
                0x8000000000000000);
  static_assert(static_data.pow10_significands[exact_end].hi == 0xd0cf4b50cfe20765);
  return -dec_exp >= exact_begin && -dec_exp <= exact_end;
}

struct stats {
  std::atomic<uint64_t> num_processed_doubles = 0;
  std::atomic<uint64_t> num_special_cases = 0;
  std::atomic<uint64_t> num_errors = 0;
};

template <int raw_exp>
void run(uint64_t bin_sig_first, uint64_t bin_sig_last, stats& s) {
  constexpr int bin_exp = debias(raw_exp);
  constexpr int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exp_shift = compute_exp_shift(bin_exp, dec_exp);
  constexpr uint64_t pow10_lo = static_data.pow10_significands[-dec_exp].lo;
  constexpr uint64_t exp_bits =
      uint64_t(raw_exp) << traits::num_sig_bits ^ traits::implicit_bit;

  // With great power of 10 comes great responsibility to check the
  // approximation error. The exact power of 10 significand is in the range
  // [pow10, pow10 + 1), where pow10 = (pow10_hi << 64) | pow10_lo.

  // Check for possible carry due to pow10 approximation error.
  // This checks all cases where integral and fractional can be off in
  // to_decimal. The rest is taken care of by the conservative boundary
  // checks on the fast path.
  bool has_errors = false;
  uint64_t last_index = 0;
  s.num_special_cases += find_carried_away_doubles<pow10_lo, exp_shift>(
      bin_sig_first, bin_sig_last, [&](uint64_t index) {
        if ((index % (1 << 20)) == 0) {
          s.num_processed_doubles += index - last_index;
          last_index = index;
        }
        uint64_t bin_sig = bin_sig_first + index;
        uint64_t bits = exp_bits ^ bin_sig;
        if (!verify(bits, bin_sig, bin_exp, raw_exp, has_errors))
          ++s.num_errors;
      });
  s.num_processed_doubles += bin_sig_last - bin_sig_first - last_index + 1;
}

template <int n>
void dispatch(int thread_index, int raw_exp, uint64_t bin_sig_first,
              uint64_t bin_sig_last, stats& s) {
  if constexpr (n == 10) {
    if (thread_index == 0) {
      fmt::print(stderr, "Unsupported exponent {}\n", raw_exp);
      exit(1);
    }
  } else {
    if (raw_exp == n) return run<n>(bin_sig_first, bin_sig_last, s);
    dispatch<n + 1>(thread_index, raw_exp, bin_sig_first, bin_sig_last, s);
  }
}

// Enumerates all doubles whose exact decimal value has a .5 at the digit
// position right after the shortest representation.
//
// For bin_sig * 2**bin_exp, the exact decimal significand is a half-integer
// when the lowest num_fixed_bits of bin_sig form a 10...0 pattern:
//
//   bin_sig (53 bits):
//   |1|  free bits (b)  |1|0 ... 0|
//    ^                   ^-------^
//    implicit bit        fixed bits
//
// where num_fixed_bits = dec_exp - bin_exp,
// dec_exp = floor(bin_exp * log10(2)),
// and num_fixed_bits in [1, num_sig_bits].
void check_exact_half_cases(int start_bin_exp = -1) {
  unsigned num_threads = std::thread::hardware_concurrency();
  fmt::print("Checking exact .5 cases from bin_exp={} with {} threads\n",
             start_bin_exp, num_threads);

  std::atomic<uint64_t> total = 0, errors = 0;
  std::atomic<int> current_bin_exp = start_bin_exp;
  std::atomic<bool> done = false;
  auto start = std::chrono::steady_clock::now();

  std::thread progress([&]() {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      double rate = total / elapsed;
      int eta = int((6.0e15 - total) / rate);
      fmt::print(stderr,
                 "\rbin_exp={:4d}  {:.2e} checked  Elapsed: {}  ETA: {}   ",
                 current_bin_exp.load(), double(total.load()),
                 format_duration(int(elapsed)), format_duration(eta, false));
      if (done) break;
    }
    fmt::print(stderr, "\n");
  });

  // bin_exp >= 0 produces integers with no fractional .5 boundary.
  for (int bin_exp = start_bin_exp;; --bin_exp) {
    int dec_exp = compute_dec_exp(bin_exp);
    int num_fixed_bits = dec_exp - bin_exp;
    if (num_fixed_bits < 1) continue;
    if (num_fixed_bits > traits::num_sig_bits) break;
    current_bin_exp = bin_exp;
    int raw_exp = bin_exp + traits::exp_offset;

    uint64_t count = uint64_t(1) << (traits::num_sig_bits - num_fixed_bits);
    auto work = [=, &total, &errors](uint64_t b_first, uint64_t b_last) {
      uint64_t local_errors = 0, local_count = 0;
      for (uint64_t b = b_first; b < b_last; ++b) {
        uint64_t bin_sig = traits::implicit_bit | (b << num_fixed_bits) |
                           (1ULL << (num_fixed_bits - 1));
        uint64_t bits = (uint64_t(raw_exp) << traits::num_sig_bits) |
                        (bin_sig & (traits::implicit_bit - 1));
        bool has_errors = false;
        if (!verify(bits, bin_sig, bin_exp, raw_exp, has_errors))
          ++local_errors;
        if ((++local_count & ((1 << 20) - 1)) == 0) {
          total += 1 << 20;
          local_count = 0;
        }
      }
      total += local_count;
      errors += local_errors;
    };

    if (count < num_threads * 1024) {
      work(0, count);
      continue;
    }
    std::vector<std::thread> threads(num_threads);
    for (unsigned i = 0; i < num_threads; ++i) {
      uint64_t b_first = count * i / num_threads;
      uint64_t b_last = count * (i + 1) / num_threads;
      threads[i] = std::thread(work, b_first, b_last);
    }
    for (auto& t : threads) t.join();
  }

  done = true;
  progress.join();
  auto finish = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(finish - start).count();
  fmt::print("Checked {:.6e} exact .5 cases in {}, {} errors\n",
             double(total.load()), format_duration(int(elapsed)),
             errors.load());
}

}  // namespace

auto main(int argc, char** argv) -> int {
  if (argc >= 2 && strcmp(argv[1], "half") == 0) {
    int start_bin_exp = -1;
    if (argc >= 3) sscanf(argv[2], "%d", &start_bin_exp);
    check_exact_half_cases(start_bin_exp);
    return 0;
  }

  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <raw_exp | half [start_bin_exp]>\n", argv[0]);
    return 1;
  }

  int raw_exp = 0;
  sscanf(argv[1], "%d", &raw_exp);

  // Verify correctness for doubles with a given binary exponent and
  // the first num_significands significands.
  // raw_exp=1 verified on commit 410dff3f with 13,220,633,789,575 hits.
  // raw_exp=2 verified on commit 9946e53c with 26,441,267,578,985 hits.
  // raw_exp=3 verified on commit 89933f51 with  3,312,278,778,759 hits.
  // raw_exp=4 verified on commit 89933f51 with  6,624,557,557,418 hits.
  // raw_exp=5 verified on commit 9946e53c with 13,220,633,789,557 hits.
  // raw_exp=6 verified on commit 9946e53c with  3,305,158,447,517 hits.
  // raw_exp=7 verified on commit 9946e53c with  6,610,316,894,858 hits.
  constexpr uint64_t num_significands = uint64_t(1) << 52;

  int bin_exp = debias(raw_exp);
  if (raw_exp == 0 || raw_exp == traits::exp_mask) {
    fmt::print(stderr, "Unsupported exponent\n");
    return 1;
  }
  int num_inexact_exponents = 0;
  for (int exp = 0; exp < traits::exp_mask; ++exp) {
    if (!is_pow10_exact_for_bin_exp(debias(exp))) ++num_inexact_exponents;
  }
  fmt::print("Verifying binary exponent {} (0x{:03x}); {} total\n", bin_exp,
             raw_exp, num_inexact_exponents);

  int dec_exp = compute_dec_exp(bin_exp, true);
  int exp_shift = compute_exp_shift(bin_exp, dec_exp);
  fmt::print("dec_exp={} exp_shift={}\n", dec_exp, exp_shift);
  if (is_pow10_exact_for_bin_exp(bin_exp)) {
    fmt::print("Power of 10 is exact for bin_exp={}\n", bin_exp);
    return 0;
  }

  unsigned num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads(num_threads);
  fmt::print("Using {} threads\n", num_threads);

  stats s;
  auto start = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < num_threads; ++i) {
    uint64_t bin_sig_first = (num_significands * i / num_threads);
    uint64_t bin_sig_last = (num_significands * (i + 1) / num_threads) - 1;

    // Skip irregular because those are tested elsewhere.
    if (bin_sig_first == 0) ++bin_sig_first;
    bin_sig_first |= traits::implicit_bit;
    bin_sig_last |= traits::implicit_bit;

    fmt::print("Thread {:3} processing 0x{:016x} - 0x{:016x}\n", i,
               bin_sig_first, bin_sig_last);
    threads[i] = std::thread([i, raw_exp, bin_sig_first, bin_sig_last, &s] {
      dispatch<1>(i, raw_exp, bin_sig_first, bin_sig_last, s);
    });
  }

  std::atomic<bool> done(false);
  std::thread progress([&]() {
    auto last_update_time = std::chrono::steady_clock::now();
    for (;;) {
      auto now = std::chrono::steady_clock::now();
      if (now - last_update_time >= std::chrono::seconds(1) || done) {
        last_update_time = now;

        double elapsed_s = std::chrono::duration<double>(now - start).count();

        double eta_s = 0;
        double rate = s.num_processed_doubles / elapsed_s;  // items / s
        double remaining = double(num_significands - s.num_processed_doubles);

        auto eta = std::chrono::seconds(int(remaining / rate + 0.5));

        fmt::print("\rProgress: {:6.2f}%  ETA: {:02d} hour(s) {:02d} minute(s)",
                   s.num_processed_doubles * 100.0 / num_significands,
                   int(eta.count() / 3600), int((eta.count() / 60) % 60));

        fflush(stdout);
        if (done) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fmt::print("\n");
  });

  for (int i = 0; i < num_threads; ++i) threads[i].join();
  done = true;
  progress.join();
  auto finish = std::chrono::steady_clock::now();

  using seconds = std::chrono::duration<double>;
  fmt::print(
      "Found {} special cases and {} errors among {} values in {:.2f} "
      "seconds\n",
      s.num_special_cases.load(), s.num_errors.load(),
      s.num_processed_doubles.load(),
      std::chrono::duration_cast<seconds>(finish - start).count());
  return s.num_errors != 0 ? 1 : 0;
}
