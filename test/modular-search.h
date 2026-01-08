#include <stdint.h>
#include <stdlib.h>

#include "fmt/base.h"

using uint128_t = unsigned __int128;

constexpr uint64_t not_found = ~uint64_t();

// Finds the smallest n >= 0 such that (n * step) % mod is in [lower, upper],
// where upper < mod, by solving a linear congruential inequality via
// modular interval reduction.
template <uint64_t step, uint128_t mod>
inline auto find_min_n(uint64_t lower, uint64_t upper) noexcept -> uint64_t {
  if constexpr (step == 0) {
    return not_found;
  } else {
    if (lower > upper) return not_found;
    if (lower == 0) return 0;  // Current position is already a hit.

    // Check for direct hit without wrapping.
    uint64_t n = (lower - 1) / step + 1;  // ceil(lower / step)
    if (uint128_t(n) * step <= upper) return n;

    // Apply recursive modular interval reduction.
    uint64_t rem_upper = upper % step;
    uint64_t rem_lower = lower % step;
    n = find_min_n<mod % step, step>(rem_upper != 0 ? step - rem_upper : 0,
                                     rem_lower != 0 ? step - rem_lower : 0);
    if (n == not_found) return not_found;
    return uint64_t((n * mod + lower + step - 1) / step);
  }
}

// Finds cases of missing carry when multiplying a significand by an
// underestimate of a power of 10 without enumerating all doubles.
template <uint64_t pow10_lo, int exp_shift, typename HitFun>
auto find_carried_away_doubles(uint64_t bin_sig_first, uint64_t bin_sig_last,
                               HitFun on_hit) noexcept -> uint64_t {
  uint64_t start = pow10_lo * (bin_sig_first << exp_shift);
  constexpr uint64_t step = pow10_lo * (1 << exp_shift);

  uint64_t threshold = ~uint64_t() - (bin_sig_last << exp_shift) + 1;

  uint64_t num_doubles = bin_sig_last - bin_sig_first + 1;
  uint64_t double_count = 0;
  uint64_t hit_count = 0;
  for (;;) {
    // If start is already above threshold, distance to hit is 0.
    uint64_t n = 0;
    if (start < threshold) {
      // Target is [threshold - start, 2**64 - 1 - start].
      // This range will never wrap because start < threshold.
      n = find_min_n<step, uint128_t(1) << 64>(threshold - start,
                                               ~uint64_t() - start);
      if (n == not_found) {
        fmt::print(stderr, "Failed to find the next hit\n");
        exit(1);
      }
    }

    ++hit_count;
    double_count += n;
    if (double_count >= num_doubles) return hit_count;

    on_hit(double_count);

    // Advance: To find the next hit, we must move at least one step.
    uint64_t hit_val = start + n * step;
    start = hit_val + step;
    ++double_count;
  }
}
