#!/usr/bin/env python3
"""
A script to verify the correctness of the Żmij FP-to-string conversion for
the extended-precision long double formats: x87 80-bit and IEEE binary128.

Copyright (c) 2025 - present, Victor Zverovich
Portions Copyright (c) 2020 YaoYuan
Distributed under the MIT license.
https://github.com/vitaut/zmij/

Companion to verify.py (double), reusing its floor_sum machinery. Żmij's
long-double path adapts YaoYuan's (yy) method, the same code for both formats.

Overview
--------

Żmij converts v = bin_sig * 2**bin_exp (a p-bit significand, p = 64 for x87,
113 for binary128) to the shortest decimal dec_sig * 10**dec_exp. Following
Schubfach it scales v by a power of ten:

    w = v * 10**(-dec_exp),   dec_exp = floor(bin_exp * log10(2)),

using a precomputed constant pow10 * 2**pow10_exp ~= 10**(-dec_exp), where
pow10 is a normalized POW10_BITS-bit integer (top bit set). With
shift = -(bin_exp + pow10_exp) the scaling is a single multiply and shift:

    w        ~= (bin_sig * pow10) * 2**(bin_exp + pow10_exp)
    floor(w) ~= (bin_sig * pow10) >> shift

The product is kept to 256 bits: a 128-bit integral part and a 128-bit
fraction (`integral` and `fractional` in to_decimal),

    scaled     = (bin_sig * pow10) >> (shift - 128)
    integral   = scaled >> 128        # floor(w)
    fractional = scaled mod 2**128    # the bits past the decimal point

Long multiply bin_sig * pow10, most significant bit on the left:

    ------------------------------------------------------------
                     |HHHHHHHHHHHHHHHH|LLLLLLLLLLLLLLLL|........  pow10
                     |        XXXXXXXX|                           bin_sig
    ------------------------------------------------------------
    |   XHXHXHXHXHXHX|HXHXHXHXHXHXHXHX|                           H * bin_sig
                     |   XLXLXLXLXLXLX|LXLXLXLXLXLXLXLX|          L * bin_sig
                                      |   .............|........  tail * bin_sig
    ------------------------------------------------------------
    |    integral    |   fractional   |....dropped.....|

H is the high 128 bits of pow10 and L the next 128; the low tail beyond
POW10_BITS is dropped. The decimal point sits between integral and fractional.

pow10 is only POW10_BITS bits (256), so it, and hence the product, is rounded
down; every bit below the kept fraction is dropped. The retained value is
therefore slightly low, and a carry out of the discarded tail could nudge it up
across a rounding boundary creating critical boundary conditions.

Focusing on the decimal result, consider this example:

bin_prev: 0x800000000000009F * 2^23
bin:      0x80000000000000A0 * 2^23
bin_next: 0x80000000000000A1 * 2^23

ulp = 8.388608 * 10^6
dec_prev: 77371252455336268514.983936 * 10^6
dec:      77371252455336268523.372544 * 10^6
dec_next: 77371252455336268531.761152 * 10^6

w = 77371252455336268523.372544, dec's coefficient before the * 10^6.

            d0             d1 w  u1                            u0    next
    ────┬────┼────┬────┬────┼─*──┼────┬────┬────┬────┬────┬────┼────┬──*─┬────
        9    0    1    2    3    4    5    6    7    8    9    0    1    2
          └───────────────────┬───────────────────┘
                             1ulp

d0/d1/u1/u0 are w's rounding candidates, defined by the cases below.

1ulp falls in [1.0, 10.0), so neighbors differ only in the last digit and
trimming targets a multiple of 10. dec's rounding interval is dec +/- 0.5ulp:
dec - 0.5ulp: 77371252455336268519.178240 * 10^6
dec + 0.5ulp: 77371252455336268527.566848 * 10^6

Rounding w to an integer involves these cases:

1. Fractional part < 0.5: round down to the nearest integer d1 (as shown)
2. Fractional part > 0.5: round up to the nearest integer u1
3. w - 0.5ulp crosses a multiple of 10 (d0): round down to it (as shown)
4. w + 0.5ulp crosses a multiple of 10 (u0): round up to it

Due to approximation errors, critical boundary conditions arise:

1. When the fractional part equals exactly 0.5, `fractional` equals 2**127
    and the dropped tail is zero. We must identify all cases where the
    approximation approaches this threshold to prevent misrounding.
2. When w - 0.5ulp lies exactly on a multiple of 10, special handling is
    required.
3. When w + 0.5ulp lies exactly on a multiple of 10, special handling is
    required. We must ensure correct detection without false positives or
    false negatives.

For performance, the algorithm packs the last decimal digit of `integral` (top
4 bits) with the high 124 bits of `fractional` into a single 128-bit integer
compared against 0.5ulp. Dropping `fractional`'s low 4 bits costs a little
precision, so the boundary conditions above need extra care.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Set, Tuple

from verify import (count_mod_mul_solutions, enumerate_mod_mul_solutions,
                    pow10_hi)

# Bits kept in each floored power-of-ten constant. The verifier needs a bit
# over digits + 128 (the tie-comparison window) so the near-boundary residues
# stay enumerable; 256 covers x87 (64-bit) and binary128 (113-bit) with room
# to spare.
POW10_BITS = 256


@dataclass
class Format:
    """Parameters of a binary floating-point format."""
    name: str         # human-readable label for the sweep output
    digits: int       # precision p: significand bits incl. the leading 1,
                      # whether that bit is stored (x87) or implicit (binary*)
    exp_bits: int     # exponent field width

    def __post_init__(self) -> None:
        sig_bits = self.digits - 1
        exp_offset = (1 << (self.exp_bits - 1)) - 1 + sig_bits
        self.sig_min = 1 << sig_bits              # smallest normal significand
        self.sig_max = (1 << self.digits) - 1
        self.bin_exp_min = 1 - exp_offset         # min unbiased exponent
        self.bin_exp_max = (1 << self.exp_bits) - 2 - exp_offset


BINARY80 = Format("x87 80-bit", digits=64, exp_bits=15)
BINARY128 = Format("IEEE binary128", digits=113, exp_bits=15)


def strip_zeros(dec_sig: int, dec_exp: int) -> Tuple[int, int]:
    """
    Drop trailing zeros from `dec_sig`, bumping `dec_exp` so that the value
    dec_sig * 10**dec_exp is unchanged and equal values compare equal.
    """
    while dec_sig and dec_sig % 10 == 0:
        dec_sig //= 10
        dec_exp += 1
    return dec_sig, dec_exp


def to_decimal(bin_sig: int, bin_exp: int, fmt: Format) -> Tuple[int, int]:
    """
    Bit-exact port of Żmij's long-double shortest path.

    Return the shortest decimal of bin_sig * 2**bin_exp as (significand, exp),
    the value significand * 10**exp with no trailing zeros.
    """
    regular = (bin_sig != fmt.sig_min)  # power of two: fraction bits are zero

    log10_2_sig = 20_201_781
    log10_3_4_sig = 8_384_497
    dec_exp = (bin_exp * log10_2_sig - (0 if regular else log10_3_4_sig)) >> 26

    pow10, pow10_exp = pow10_hi(-dec_exp, POW10_BITS)  # floored power of ten
    shift = -(bin_exp + pow10_exp)

    mask128 = (1 << 128) - 1
    product = bin_sig * pow10
    scaled = (product >> (shift - 128)) & ((1 << 256) - 1)
    integral = scaled >> 128
    fractional = scaled & mask128
    last_digit = integral % 10

    half_ulp = (pow10 >> (shift - 123)) & mask128
    c = ((last_digit << 124) | (fractional >> 4)) & mask128
    half = 1 << 127
    ten = 10 << 124
    even = (bin_sig & 1) == 0

    if regular:
        round_up = fractional >= half
        if fractional == half:
            round_up = (integral & 1) != 0
        trim_down = c <= half_ulp
        if c == half_ulp:
            # A 124-bit tie: c == half_ulp fixes the high bits, so the low 64
            # bits (pow10 one nibble finer) refine it, to even on an exact match.
            frac_lo = fractional & ((1 << 64) - 1)
            ulp_lo = (pow10 >> (shift - 127)) & ((1 << 64) - 1)
            trim_down = even if frac_lo == ulp_lo else frac_lo < ulp_lo
    else:
        round_up = fractional > half
        quarter_ulp = half_ulp >> 1
        if (fractional >> 4) > quarter_ulp:
            round_up = True
        trim_down = c <= quarter_ulp

    trim_up = c >= ten - half_ulp
    gap = (ten - half_ulp - c) & mask128
    if gap <= 1 and (dec_exp == 0 or gap == 1):
        trim_up = even

    if trim_down or trim_up:
        dec_sig = integral - last_digit + (10 if trim_up else 0)
    else:
        dec_sig = integral + round_up

    return strip_zeros(dec_sig, dec_exp)


def log10_floor(f: Fraction) -> int:
    """floor(log10(f)) for f > 0."""
    n, d = f.numerator, f.denominator

    def ge_pow(k: int) -> bool:  # n/d >= 10**k, using only non-negative powers
        return n >= d * 10 ** k if k >= 0 else n * 10 ** (-k) >= d

    # Estimate from bit lengths (log10(2) ~ 0.30103); the loops below correct
    # it. str(n) would trip Python's 4300-digit int-to-str limit for extremes.
    k = int((n.bit_length() - d.bit_length()) * 0.30103)
    while ge_pow(k + 1):
        k += 1
    while not ge_pow(k):
        k -= 1
    return k


def to_decimal_exact(bin_sig: int, bin_exp: int, fmt: Format
                     ) -> Tuple[int, int]:
    """
    The true shortest correctly-rounded decimal of bin_sig * 2**bin_exp,
    computed with exact Fraction arithmetic; the reference for to_decimal.
    """
    two = Fraction(2)
    v = Fraction(bin_sig) * two ** bin_exp

    # Nearest representable neighbors. Only a power of two above the minimum
    # exponent is irregular (its lower gap is half an ulp); subnormals and the
    # smallest normal are uniformly spaced.
    if bin_sig < fmt.sig_max:
        succ = Fraction(bin_sig + 1) * two ** bin_exp
    else:
        succ = Fraction(fmt.sig_min) * two ** (bin_exp + 1)
    if bin_sig == fmt.sig_min and bin_exp > fmt.bin_exp_min:
        pred = Fraction(fmt.sig_max) * two ** (bin_exp - 1)
    else:
        pred = Fraction(bin_sig - 1) * two ** bin_exp

    lo = (v + pred) / 2
    hi = (v + succ) / 2
    closed = (bin_sig % 2 == 0)  # endpoints round to v only under ties-to-even

    # Largest p (fewest significant digits) with a multiple of 10**p in the
    # rounding interval; then the in-interval multiple nearest v (ties to even).
    # Run the search with integer floor/ceil on the fractions' numerators and
    # denominators, avoiding gcd-heavy Fraction division in the hot loop.
    ln, ld = lo.numerator, lo.denominator
    hn, hd = hi.numerator, hi.denominator
    vn, vd = v.numerator, v.denominator
    p_hi = log10_floor(hi) + 2
    p_lo = log10_floor(hi - lo) - 2
    for p in range(p_hi, p_lo - 1, -1):
        gn, gd = (10 ** p, 1) if p >= 0 else (1, 10 ** -p)
        lnum, lden = ln * gd, ld * gn    # lo / 10**p
        hnum, hden = hn * gd, hd * gn    # hi / 10**p
        kmin = -(-lnum // lden)          # ceil
        if not closed and lnum % lden == 0:
            kmin += 1
        kmax = hnum // hden              # floor
        if not closed and hnum % hden == 0:
            kmax -= 1
        if kmin > kmax:
            continue
        # round() on a Fraction gives nearest, ties to even; clamp to interval.
        k = max(kmin, min(kmax, round(Fraction(vn * gd, vd * gn))))
        return strip_zeros(k, p)
    raise AssertionError("no shortest decimal found")


# --- verification ----------------------------------------------------------
#
# The three find_edge_case_* searches are adapted from yy_double/verify.py
# (https://github.com/ibireme/c_numconv_benchmark), which verifies binary64,
# retargeted to the extended-precision formats, and using floor_sum instead of
# continued fractions and the three-gap theorem.
#
# One search per rounding boundary. Each counts the near-boundary residues (A),
# the R = (bin_sig * pow10) mod 2**shift the algorithm reads as a tie, using
# floor_sum via count_mod_mul_solutions. For the two trim boundaries we also
# count the exact ties (B, exact_tie_progression) and the intersection A & B
# (intersection_count); asserting |A| == |B| == |A & B| proves A == B, so every
# significand the algorithm trims is a genuine tie and vice versa. Equal
# cardinality alone would not: the sets can differ yet still match in size.
# The round-to-nearest search has no closed-form count and instead oracle-checks
# each candidate.


def exact_tie_progression(bin_exp: int, dec_exp: int, sig_min: int,
                          sig_max: int, sign: int) -> Tuple[int, int, int]:
    """
    Significands in [sig_min, sig_max] whose boundary v + sign * half_ulp lands
    exactly on a multiple of 10**(dec_exp + 1), the grid a trim rounds to.

    The boundary is 2**(bin_exp - 1) * (2 * sig + sign) (v = sig * 2**bin_exp,
    half_ulp = 2**(bin_exp - 1)). 2 * sig + sign is odd, so it holds no factors
    of two, and landing on a multiple of 10**(dec_exp + 1) = 2**(dec_exp + 1) *
    5**(dec_exp + 1) first needs bin_exp >= dec_exp + 2 to supply the twos. The
    surviving power of two is then a unit mod 5**(dec_exp + 1), so the condition
    reduces to 2 * sig + sign == 0 (mod 5**(dec_exp + 1)): one residue class
    sig == first (mod period), period = 5**(dec_exp + 1). Return (first, period,
    count) with `first` the smallest solution >= sig_min, or (0, 0, 0) if none.
    """
    if bin_exp < dec_exp + 2:          # boundary lacks the twos to hit the grid
        return 0, 0, 0
    period = 5 ** max(dec_exp + 1, 0)  # 5-adic part of grid, 1 if dec_exp < 0
    # 2 is invertible mod the odd period, with inverse (period + 1) // 2.
    x0 = (-sign * ((period + 1) // 2)) % period
    first = x0 + -(-(sig_min - x0) // period) * period  # first >= sig_min
    if first > sig_max:
        return 0, 0, 0
    return first, period, (sig_max - first) // period + 1


class Params:
    """Per-exponent constants of the regular path, mirroring to_decimal."""

    def __init__(self, bin_exp: int):
        self.bin_exp = bin_exp
        self.dec_exp = (bin_exp * 20_201_781) >> 26
        pow10, pow10_exp = pow10_hi(-self.dec_exp, POW10_BITS)
        self.pow10 = pow10
        self.shift = -(bin_exp + pow10_exp)
        self.half_ulp = (pow10 >> (self.shift - 123)) & ((1 << 128) - 1)
        # pow10 is exact (drops no significant bits) iff 5**(-dec_exp) fits in
        # POW10_BITS; then scaling adds no error and the tie tests are exact.
        self.exact = (self.dec_exp <= 0
                      and (5 ** -self.dec_exp).bit_length() <= POW10_BITS)


def find_edge_case_1(p: Params, sig_min: int, sig_max: int, fmt: Format,
                     found: Set[Tuple[int, int]]) -> None:
    """
    Round to nearest (boundary condition 1): check significands whose
    fractional part lands within one LSB of the 1/2 tie (fractional == 2**127),
    where the floored pow10 could push the true value across it. Exact pow10
    adds no error, so skip it. Unlike the trim ties there is no closed-form
    count, so each candidate is compared against the exact oracle and any
    misround is recorded in `found`.
    """
    if p.exact:
        return
    den = 1 << p.shift
    lsb = 1 << (p.shift - 128)           # R spanned by one fractional unit
    tie = 1 << (p.shift - 1)             # fractional == 2**127
    lo, hi = tie - lsb, tie + lsb - 1
    if count_mod_mul_solutions(p.pow10, den, sig_min, sig_max, lo, hi) == 0:
        return
    for bin_sig, _ in enumerate_mod_mul_solutions(p.pow10, den, sig_min,
                                                  sig_max, lo, hi):
        if to_decimal(bin_sig, p.bin_exp, fmt) != \
                to_decimal_exact(bin_sig, p.bin_exp, fmt):
            found.add((p.bin_exp, bin_sig))


def trim_band(p: Params, c: int, bits: int = 124) -> Tuple[int, int, int]:
    """
    Residue window (mod 10 * 2**shift) and modulus covering the one LSB the
    algorithm reads as this c. c = last_digit * 2**bits | fractional >> (128 -
    bits); encoding res = last_digit * 2**shift + R (R = product mod 2**shift)
    pins the last digit while R ranges over one LSB.

    `bits` is how many of `fractional`'s high bits `c` retains: 124 for the
    packed 124-bit comparison, 128 for the full-precision refinement, which
    narrows the window to a single `fractional` unit.
    """
    lsb = 1 << (p.shift - bits)
    base = (c >> bits) * (1 << p.shift) + (c & ((1 << bits) - 1)) * lsb
    return 10 << p.shift, base, base + lsb - 1


def assert_trim(p: Params, sig_min: int, sig_max: int, c: int, sign: int,
                label: str, bits: int = 124) -> None:
    """
    Assert the algorithm's tie set A (near-boundary residues c reads as a tie)
    equals the exact tie set B (boundaries on the decimal grid). We check
    |A| == |B| == |A & B|: since A & B is contained in both, equal counts force
    A & B == A == B, so no false ties (misrounds) and no missed ties.

    `bits` selects the comparison precision (see trim_band): 124 for the packed
    comparison, 128 for the full-precision refinement.
    """
    den, lo, hi = trim_band(p, c, bits)
    approx = count_mod_mul_solutions(p.pow10, den, sig_min, sig_max, lo, hi)
    first, period, exact = exact_tie_progression(p.bin_exp, p.dec_exp,
                                                 sig_min, sig_max, sign)
    # Ties along first, first + period, ... that also fall in the trim band,
    # walked with count_mod_mul_solutions' affine `add` so it stays O(log) even
    # when there are astronomically many ties.
    both = 0 if exact == 0 else count_mod_mul_solutions(
        p.pow10 * period, den, 0, exact - 1, lo, hi, add=p.pow10 * first)
    assert approx == exact == both, \
        f"{label} bin_exp={p.bin_exp} dec_exp={p.dec_exp} " \
        f"approx={approx} exact={exact} both={both}"


def find_edge_case_2(p: Params, sig_min: int, sig_max: int) -> None:
    """
    Trim down to a multiple of 10 (boundary condition 2): v - half_ulp on that
    multiple.

    c == half_ulp is a 124-bit tie the algorithm refines with fractional's 4
    dropped bits, so the true tie is the full 128-bit equality c_fine ==
    (pow10 >> shift - 127), where c_fine = last_digit << 128 | fractional. We
    search that 128-bit boundary and the set-equality assertion confirms it.
    """
    if p.exact:
        return
    boundary = p.pow10 >> (p.shift - 127)
    assert_trim(p, sig_min, sig_max, boundary, -1, "trim_down", bits=128)


def find_edge_case_3(p: Params, sig_min: int, sig_max: int) -> None:
    """
    Trim up to a multiple of 10 (boundary condition 3): v + half_ulp on that
    multiple.

    Flooring pow10 (hence also half_ulp) can only lower the algorithm's c, so a
    genuine tie is expected one LSB below the true threshold ten - half_ulp, at
    gap == 1, the position the even override treats as the tie. We search at
    c == ten - half_ulp - 1, and the set-equality assertion confirms it.
    """
    if p.exact:
        return
    assert_trim(p, sig_min, sig_max, (10 << 124) - p.half_ulp - 1, +1,
                "trim_up")


def find_edge_cases(fmt: Format) -> None:
    """Run the three edge-case searches over every binary exponent."""
    if not __debug__:
        raise RuntimeError("run this verifier without -O; the trim-tie set "
                           "equality checks rely on proof-critical assertions")
    print(f"{fmt.name} edge-case sweep ... ", end="", flush=True)
    found: Set[Tuple[int, int]] = set()
    for bin_exp in range(fmt.bin_exp_min, fmt.bin_exp_max + 1):
        p = Params(bin_exp)
        # Regular significands (the power of two at sig_min is irregular and
        # not covered here); subnormals share bin_exp_min, use the regular path.
        ranges = [(fmt.sig_min + 1, fmt.sig_max)]
        if bin_exp == fmt.bin_exp_min:
            ranges.append((1, fmt.sig_min - 1))
        for sig_min, sig_max in ranges:
            find_edge_case_1(p, sig_min, sig_max, fmt, found)
            find_edge_case_2(p, sig_min, sig_max)
            find_edge_case_3(p, sig_min, sig_max)

    print("ok")
    if found:
        print(f"  {len(found)} round-to-nearest misround(s):")
        for bin_exp, bin_sig in sorted(found):
            print(f"    bin_sig=0x{bin_sig:X} bin_exp={bin_exp}: "
                  f"actual={to_decimal(bin_sig, bin_exp, fmt)} "
                  f"expected={to_decimal_exact(bin_sig, bin_exp, fmt)}")
    assert not found, f"{fmt.name}: {len(found)} round-to-nearest misround(s)"


if __name__ == "__main__":
    find_edge_cases(BINARY80)
    find_edge_cases(BINARY128)
