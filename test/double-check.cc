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
#include "fmt/base.h"
#include "modular-search.h"

namespace {

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
  to_decimal_result actual = to_decimal_normal<double>(bin_sig, raw_exp, true);

  double value;
  memcpy(&value, &bits, sizeof(double));
  auto expected = jkj::dragonbox::to_decimal(value);

  uint32_t abbccddee = uint32_t(actual.sig / 100'000'000);
  uint32_t ffgghhii = uint32_t(actual.sig % 100'000'000);
  int num_zeros = 0;
  if (ffgghhii == 0)
    num_zeros = 16 - count_trailing_nonzeros(to_bcd8(abbccddee % 100'000'000));
  else
    num_zeros = 8 - count_trailing_nonzeros(to_bcd8(ffgghhii));
  if (num_zeros != 0) {
    expected.significand *= pow10[num_zeros];
    expected.exponent -= num_zeros;
  }

  if (actual.sig == expected.significand && actual.exp == expected.exponent)
    return true;

  if (has_errors) return false;
  has_errors = true;
  fmt::print(
      "Output mismatch for {} ({} * 2**{}): {} * 10**{} != {} * 10**{}\n",
      value, bin_sig, bin_exp, actual.sig, actual.exp, expected.significand,
      expected.exponent);
  return false;
}

auto is_pow10_exact_for_bin_exp(int bin_exp) -> bool {
  int dec_exp = compute_dec_exp(bin_exp, true);
  constexpr int exact_begin = -0, exact_end = 55;
  static_assert(pow10_significands[exact_begin].hi == 0x8000000000000000);
  static_assert(pow10_significands[exact_end].hi == 0xd0cf4b50cfe20765);
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
  constexpr int exp_shift = compute_exp_shift<64, true>(bin_exp, dec_exp);
  constexpr uint64_t pow10_lo = pow10_significands[-dec_exp].lo;
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

}  // namespace

auto main(int argc, char** argv) -> int {
  if (argc != 2) {
    fmt::print(stderr, "Usage: {} <raw_exp>\n", argv[0]);
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
  int exp_shift = compute_exp_shift<64, true>(bin_exp, dec_exp);
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
