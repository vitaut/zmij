#!/usr/bin/env python3
"""
A script to verify the correctness of the zmij double-to-string algorithm.

Copyright (c) 2025 - present, Victor Zverovich
Distributed under the MIT license (see LICENSE) or alternatively
the Boost Software License, Version 1.0.
https://github.com/vitaut/zmij/

It ports zmij's `to_decimal<double>`, re-derives its edge cases, and checks
them across all significands of every binary exponent, counted with O(log)
floor_sum.

Inspired by verify.py by YaoYuan. zmij rounds with carry tests on a 64-bit
`fractional` (not modular comparisons split across three edge-case searches),
so the boundaries are re-derived here, and solutions are counted with an O(log)
floor_sum solver rather than continued fractions and the three-gap theorem.

Overview
--------

zmij converts a binary float sig2 * 2^bin_exp to the shortest decimal, using a
Schubfach-style single multiply by a power-of-ten cache, with a Xiang JunBo
(xjb) twist: it scales by 10^(-dec_exp-1) to directly produce the shortened
15-16 digit significand and derives the extra (17th) digit from the fractional
part.

For each input (regular double path):

    dec_exp = floor(bin_exp * log10(2))          # 315653 >> 20
    shift   = compute_exp_shift(...) + 6         # 217707 >> 16; extra_shift 6
    cache   = pow10_hi128(-dec_exp - 1)          # rounded-down 128-bit sig
    p       = umul192_hi128(cache, sig << shift)  # top 128 bits of the product
    integral   = p >> 70
    fractional = (p >> 6) mod 2^64
    half_ulp   = (cache_hi >> (7 - shift)) + even,   even = 1 - (sig & 1)

The rounding decision uses carry/overflow tests:

    round_up   = fractional + half_ulp >= 2^64   (carry past 2^64)
    round_down = half_ulp > fractional
    digit      = (fractional * 10 + biased_half) >> 64,  biased_half = 2^63 + 6
    (special: fractional == 2^62 -> digit = 2)   # round 2.5 to 2
    integral  += round_up
    has_last_digit = (round_up + round_down) == 0

Exact powers of two (fraction field 0) take an asymmetric-boundary branch:
the lower gap is half as wide.

The finite `fractional` equals floor(2^64 * true_fraction) up to a slack of 2
units (each of `fractional` and `half_ulp` is off by at most 1: the cache is
rounded down and the product and >>6 truncate). A misround can therefore only
happen when the exact value lands within that slack of a decision boundary. We
use floor_sum to enumerate, across all 2^52 significands of every binary
exponent, exactly the significands whose finite `fractional` lands within a
window of each boundary (round_up carry, round_down, each last-digit tie, and
the 2^62 special case). Each enumerated candidate is then checked against the
correctly-rounded shortest decimal (Python's repr).

    fractional = floor(cache * sig / 2^(70 - shift)) mod 2^64

so "fractional == V" is the modular window condition

    (cache * sig) mod 2^(134 - shift) in [V << (70 - shift), ...]

which count_mod_mul_solutions / enumerate_mod_mul_solutions answer directly.
"""

import struct
from fractions import Fraction
from typing import Iterator, Set, Tuple


def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """
    Compute sum_{i=0}^{n-1} floor((a*i + b) / m) in O(log(a + m)) time.

    Requires n >= 0 and m >= 1. `a` and `b` may be any integers (including
    negative). Evaluated by a Euclidean-style recursion that swaps the roles
    of `a` and `m` (the same continued-fraction descent as Euclid's gcd).
    """
    assert n >= 0 and m >= 1
    total = 0
    if a < 0:
        a_mod = a % m
        total -= n * (n - 1) // 2 * ((a_mod - a) // m)
        a = a_mod
    if b < 0:
        b_mod = b % m
        total -= n * ((b_mod - b) // m)
        b = b_mod
    while True:
        if a >= m:
            total += n * (n - 1) // 2 * (a // m)
            a %= m
        if b >= m:
            total += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            break
        n = y_max // m
        b = y_max % m
        m, a = a, m
    return total


def count_mod_mul_solutions(num: int, mod: int,
                            x_min: int, x_max: int,
                            y_min: int, y_max: int) -> int:
    """
    Count the x in [x_min, x_max] for which (num * x) % mod lies in the closed
    interval [y_min, y_max].

    Handles every case (non-coprime num/mod, degenerate intervals,
    x_max >= mod, ...) without special-casing, and always returns a valid
    non-negative count.
    """
    assert num > 0 and mod > 0
    assert 0 <= x_min <= x_max
    assert 0 <= y_min <= y_max

    n = x_max - x_min + 1
    b = num * x_min

    hi = min(y_max, mod - 1)
    if y_min > hi:
        return 0

    base = floor_sum(n, mod, num, b)

    def count_le(t: int) -> int:
        return base - floor_sum(n, mod, num, b - t - 1)

    return count_le(hi) - count_le(y_min - 1)


def enumerate_mod_mul_solutions(num: int, mod: int,
                                x_min: int, x_max: int,
                                y_min: int, y_max: int
                                ) -> Iterator[Tuple[int, int]]:
    """
    Yield each (x, y) with y = num * x % mod, x in [x_min, x_max], y in
    [y_min, y_max], in increasing order of x.

    `floor_sum` only counts; to enumerate we binary-search on the prefix count
    #{x' in [x_min, x] : y in range}, which is monotonic in x. Only meaningful
    when the number of solutions is small (as in the edge-case searches here).
    """
    lo = max(y_min, 0)
    hi = min(y_max, mod - 1)
    if lo > hi or x_min > x_max:
        return

    def prefix(x: int) -> int:
        if x < x_min:
            return 0
        return count_mod_mul_solutions(num, mod, x_min, x, lo, hi)

    total = prefix(x_max)
    for j in range(1, total + 1):
        left, right = x_min, x_max
        while left < right:
            mid = (left + right) // 2
            if prefix(mid) >= j:
                right = mid
            else:
                left = mid + 1
        x = left
        yield x, num * x % mod


# --- zmij port -------------------------------------------------------------

MASK64 = (1 << 64) - 1
EXTRA_SHIFT = 6
BIASED_HALF = (1 << 63) + 6

# double fields
EXP_OFFSET = 1075  # exp_bias(1023) + num_sig_bits(52)
SIG_BITS = 52


def compute_dec_exp(bin_exp: int, regular: bool = True) -> int:
    """floor(log10(2^bin_exp)) (regular) or floor(log10(3/4 * 2^bin_exp))."""
    log10_3_over_4_sig = 131072
    log10_2_sig, log10_2_exp = 315653, 20
    return (bin_exp * log10_2_sig
            - (0 if regular else log10_3_over_4_sig)) >> log10_2_exp


def compute_exp_shift(bin_exp: int, dec_exp: int) -> int:
    """Base shift so the fractional part lands in a fixed bit window."""
    log2_pow10_sig, log2_pow10_exp = 217707, 16
    pow10_bin_exp = (-dec_exp * log2_pow10_sig) >> log2_pow10_exp
    return bin_exp + pow10_bin_exp + 1


def umul192_hi128(cache: int, y: int) -> int:
    """Top 128 bits of the 192-bit product cache128 * y (drops low 64 bits)."""
    return (cache * y) >> 64


def umul128_add_hi64(x: int, y: int, c: int) -> int:
    return (x * y + c) >> 64


def pow10_hi128(p: int) -> int:
    """
    Return the top 128 bits of 10^p, normalized so bit 127 is set.
    `p` is the decimal exponent and may be negative (reciprocal powers).
    """
    pow10 = 10 ** abs(p)
    b = pow10.bit_length()
    if p >= 0:
        return (pow10 << 128) >> b
    return (1 << (b + 127)) // pow10


def exp_params(raw_exp: int) -> Tuple[int, int, int, int]:
    """Per-exponent (bin_exp, dec_exp, shift, cache) for the regular path."""
    bin_exp = raw_exp - EXP_OFFSET
    dec_exp = compute_dec_exp(bin_exp)
    shift = compute_exp_shift(bin_exp, dec_exp + 1) + EXTRA_SHIFT
    cache = pow10_hi128(-dec_exp - 1)
    return bin_exp, dec_exp, shift, cache


def to_decimal(sig: int, raw_exp: int) -> Tuple[int, int, int, bool]:
    """
    Port of zmij's to_decimal<double>.
    Returns (integral, dec_exp, digit, has_last_digit).
    """
    bin_exp = raw_exp - EXP_OFFSET
    # Irregular (asymmetric boundaries) iff sig is a power of two.
    regular = sig != (1 << SIG_BITS)

    dec_exp = compute_dec_exp(bin_exp, regular)
    shift = compute_exp_shift(bin_exp, dec_exp + 1) + EXTRA_SHIFT
    cache = pow10_hi128(-dec_exp - 1)
    cache_hi = cache >> 64
    p = umul192_hi128(cache, sig << shift)
    integral = p >> (64 + EXTRA_SHIFT)
    fractional = (p >> EXTRA_SHIFT) & MASK64
    half_ulp = cache_hi >> (EXTRA_SHIFT + 1 - shift)

    if not regular:
        round_up = half_ulp > MASK64 - fractional
        round_down = (half_ulp >> 1) > fractional
        integral += round_up
        digit = umul128_add_hi64(fractional, 10, (1 << 63) - 1)
        lo = umul128_add_hi64(
            (fractional - (half_ulp >> 1)) & MASK64, 10, MASK64)
        if digit < lo:
            digit = lo
        return integral, dec_exp, digit, (round_up + round_down) == 0

    half_ulp += 1 - (sig & 1)
    round_up = fractional + half_ulp > MASK64
    round_down = half_ulp > fractional
    integral += round_up
    digit = umul128_add_hi64(fractional, 10, BIASED_HALF)
    if fractional == (1 << 62):
        digit = 2
    return integral, dec_exp, digit, (round_up + round_down) == 0


# --- reference: exact value + correctly-rounded shortest decimal ------------

def double_from_fields(sig: int, raw_exp: int) -> float:
    """Reconstruct the double from the (significand, raw exponent) fields."""
    implicit = 1 << SIG_BITS
    if sig >= implicit:  # normal: sig carries the implicit bit
        bits = (raw_exp << SIG_BITS) | (sig - implicit)
    else:  # subnormal: biased exponent is 0, so raw_exp is unused
        bits = sig
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def check_value(sig: int, raw_exp: int) -> bool:
    """True iff zmij produces the right value AND the shortest decimal."""
    value = double_from_fields(sig, raw_exp)
    integral, dec_exp, digit, has_last_digit = to_decimal(sig, raw_exp)
    final_sig = integral * 10 + (digit if has_last_digit else 0)
    got = Fraction(final_sig) * Fraction(10) ** dec_exp
    want = Fraction(repr(value))
    # reject the one value-preserving non-shortest case: an emitted trailing 0
    return got == want and not (has_last_digit and digit == 0)


# --- verification ----------------------------------------------------------
#
# check_value checks the shortest representation, not just the numeric value:
# has_last_digit with a zero last digit emits a redundant trailing 0, and it is
# rejected. This is sound to check only near boundaries because has_last_digit
# is piecewise-constant in `fractional`, flipping exactly at the round_up /
# round_down boundaries (fractional == 2^64 - half_ulp and fractional ==
# half_ulp) enumerated below.
#
# The finite `fractional` and `half_ulp` each differ from the exact scaled
# values by at most 1 unit, so a misround can only occur when the exact value
# lands within a couple of units of a decision boundary. We enumerate exactly
# those significands with floor_sum.
#
# A single boundary can be hit by astronomically many significands at once
# (when the cache is exact, but also for the reciprocal caches near dec_exp
# 16-17). For such large clusters every member is an exact tie - proven per
# cluster in check_boundary via a danger-band count - so the rounding outcome
# depends only on (fractional, sig parity) and is handled correctly by design
# (biased_half's +6, the 2^62 -> digit 2 special case). We therefore check one
# representative per (fractional, parity) instead of every significand.

BOUNDARY_WINDOW = 4  # conservative; 2 suffices (errors are <= 1 each)


def exact_fractional(sig: int, bin_exp: int, dec_exp: int) -> int:
    """Exact floor(2^64 * fractional part of sig*2^bin_exp / 10^(dec_exp+1))."""
    scaled = (Fraction(sig) * Fraction(2) ** bin_exp
              / Fraction(10) ** (dec_exp + 1))
    return int((scaled - int(scaled)) * (1 << 64))


def count_boundary_keys(cache: int, mod: int, s: int, sig_min: int,
                        sig_max: int, v_lo: int, v_hi: int) -> int:
    """
    Exact number of distinct (fractional, parity) keys among the significands
    in [sig_min, sig_max] whose fractional lies in [v_lo, v_hi]. Counts each
    parity class by reparametrizing sig = 2t + p (so residue is linear in t).
    """
    keys = 0
    for f in range(v_lo, v_hi + 1):
        lo, hi = f << s, (f << s) | ((1 << s) - 1)
        for p in (0, 1):
            t_min = max(-(-(sig_min - p) // 2), 0)  # ceil((sig_min - p) / 2)
            t_max = (sig_max - p) // 2
            if t_min > t_max:
                continue
            c = (cache * p) % mod
            a, b = (lo - c) % mod, (hi - c) % mod
            if a <= b:
                n = count_mod_mul_solutions(2 * cache, mod, t_min, t_max, a, b)
            else:  # window wraps past 0 after the parity shift
                n = (count_mod_mul_solutions(2 * cache, mod, t_min, t_max,
                                             a, mod - 1)
                     + count_mod_mul_solutions(2 * cache, mod, t_min, t_max,
                                               0, b))
            keys += n > 0
    return keys


def check_boundary(raw_exp: int, bin_exp: int, dec_exp: int, cache: int,
                   mod: int, s: int, sig_min: int, sig_max: int, target: int,
                   exceptions: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Check every significand whose `fractional` is within BOUNDARY_WINDOW of
    `target` against the correctly-rounded reference. Returns (hits, tested):
    the number of significands landing in the window (the whole cluster in the
    large-cluster case) and the number checked directly (all of them for small
    clusters, one representative per key for large ones).
    """
    v_lo = max(target - BOUNDARY_WINDOW, 0)
    v_hi = min(target + BOUNDARY_WINDOW, (1 << 64) - 1)
    y_lo = v_lo << s
    y_hi = (v_hi << s) | ((1 << s) - 1)

    count = count_mod_mul_solutions(cache, mod, sig_min, sig_max, y_lo, y_hi)
    if count == 0:
        return 0, 0

    candidates = enumerate_mod_mul_solutions(
        cache, mod, sig_min, sig_max, y_lo, y_hi)
    cluster_limit = 512  # perf dispatch: above this, sample + prove ties
    if count <= cluster_limit:
        for sig, _ in candidates:
            if not check_value(sig, raw_exp):
                exceptions.add((raw_exp, sig))
        return count, count

    # Large cluster: check one representative per (fractional, parity) key
    # instead of every member. Sound only if all members sharing a key round
    # identically, i.e. each member's exact fractional equals its finite
    # `fractional`. Since `cache` is rounded down, exact = finite or finite + 1,
    # and the +1 carry occurs only when the low s bits of cache*sig fall in the
    # danger band [2^s - sig_max, 2^s - 1]. The assert enforces that no windowed
    # member lands there, so sampling by key is sound.
    danger_lo = (1 << s) - sig_max
    for f in range(v_lo, v_hi + 1):
        band_lo = (f << s) + danger_lo
        band_hi = ((f + 1) << s) - 1
        n = count_mod_mul_solutions(cache, mod, sig_min, sig_max,
                                    band_lo, band_hi)
        assert n == 0, (raw_exp, target, f)

    # Every member now has an exact fractional value equal to the finite
    # fractional value, so correctness depends only on (fractional, parity):
    # at most 2 distinct keys, both seen within the first few candidates.
    # pull_cap bounds the work; the coverage check below confirms every key
    # was seen.
    seen = set()
    pull_cap = 8
    for i, (sig, residue) in enumerate(candidates):
        if i >= pull_cap:
            break
        key = (residue >> s, sig & 1)
        if key in seen:
            continue
        seen.add(key)
        # Subsumed by the danger-band proof above; a cheap direct confirmation
        # that this representative really is an exact tie.
        assert (residue >> s) == exact_fractional(sig, bin_exp, dec_exp), \
            (raw_exp, sig, count)
        if not check_value(sig, raw_exp):
            exceptions.add((raw_exp, sig))

    # We must have seen every (fractional, parity) class present, not just the
    # first pull_cap candidates' worth.
    assert len(seen) == count_boundary_keys(cache, mod, s, sig_min, sig_max,
                                            v_lo, v_hi), (raw_exp, target)
    return count, len(seen)


def find_regular_edge_cases(raw_exp: int, sig_min: int, sig_max: int,
                            exceptions: Set[Tuple[int, int]]
                            ) -> Tuple[int, int]:
    """Check boundary significands in [sig_min, sig_max] for one exponent."""
    assert sig_min <= sig_max, (raw_exp, sig_min, sig_max)
    bin_exp, dec_exp, shift, cache = exp_params(raw_exp)
    cache_hi = cache >> 64

    # fractional = floor(cache * sig / 2^s) mod 2^64, matching to_decimal's
    # (p >> EXTRA_SHIFT) over the 128-bit product.
    s = 64 + EXTRA_SHIFT - shift
    assert s >= 0
    mod = 1 << (s + 64)

    # Fractional values where a rounding decision can flip: round_down
    # (fractional == half_ulp), round_up (fractional == 2^64 - half_ulp), the
    # 2^62 "round 2.5 to 2" special case, and the nine last-digit boundaries
    # where digit = (fractional*10 + biased_half) >> 64 crosses k * 2^64.
    half_ulp = cache_hi >> (EXTRA_SHIFT + 1 - shift)
    targets = [half_ulp, (1 << 64) - half_ulp, 1 << 62]
    for k in range(1, 10):
        targets.append((k * (1 << 64) - BIASED_HALF) // 10)

    hits = tested = 0
    for target in targets:
        h, t = check_boundary(raw_exp, bin_exp, dec_exp, cache, mod, s,
                              sig_min, sig_max, target, exceptions)
        hits += h
        tested += t
    return hits, tested


def find_edge_cases() -> None:
    """Sweep every binary exponent for potential misrounds."""
    if not __debug__:
        raise RuntimeError("run this verifier without -O; the large-cluster "
                           "reduction relies on proof-critical assertions")
    print("double edge-case sweep ... ", end="", flush=True)
    implicit = 1 << SIG_BITS
    normal_max = (1 << (SIG_BITS + 1)) - 1
    exceptions: Set[Tuple[int, int]] = set()
    boundary_hits = 0  # significands landing in a boundary window
    representatives_tested = 0  # significands actually run through check_value
    powers_of_two = 0

    for raw_exp in range(1, 2047):
        # Exact powers of two: exactly one significand, checked directly.
        powers_of_two += 1
        if not check_value(implicit, raw_exp):
            exceptions.add((raw_exp, implicit))

        # Regular significands (exclude the power of two at `implicit`).
        h, t = find_regular_edge_cases(raw_exp, implicit + 1, normal_max,
                                       exceptions)
        boundary_hits += h
        representatives_tested += t
        if raw_exp == 1:  # subnormals share raw_exp 1 and use the regular path
            h, t = find_regular_edge_cases(raw_exp, 1, implicit - 1, exceptions)
            boundary_hits += h
            representatives_tested += t

    if exceptions:
        print("FAILED")
        print(f"  {len(exceptions)} misrounds:")
        for raw_exp, sig in sorted(exceptions):
            value = double_from_fields(sig, raw_exp)
            print(f"  raw_exp={raw_exp} sig={sig} value={value!r} "
                  f"to_decimal={to_decimal(sig, raw_exp)}")
        raise SystemExit(1)

    print("ok")
    print(f"  {boundary_hits:,} boundary-window hits, "
          f"{representatives_tested:,} tested directly, "
          f"{powers_of_two:,} powers of two; no misrounds")


if __name__ == "__main__":
    find_edge_cases()
