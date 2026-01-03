#include <stdint.h>

using uint128_t = unsigned __int128;

constexpr uint64_t not_found = ~uint64_t();

// Finds the smallest n >= 0 such that (n * step) % mod is in [lower, upper],
// where upper < mod, by solving a linear congruential inequality via
// modular interval reduction.
template <uint64_t step, uint128_t mod>
inline auto find_min_n(uint64_t lower, uint64_t upper) -> uint64_t {
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
