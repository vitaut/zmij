#!/usr/bin/env python3
"""
A script to verify the correctness of the yy_double algorithm.
This script requires Python 3.8 or higher.

Author: YaoYuan (ibireme@gmail.com)
Date: 2024-11-10

Modified for xjb/zmij's optimized double-to-string path which uses dec_exp+1
scaling (e10 = -k-1), extra_shift=6, and extracts the last digit from the
fractional part rather than the integral part.
"""

import math, decimal, struct
from decimal import Decimal
from typing import List, Tuple, Dict, Callable


"""
Overview
------------------

Every float64 number can be precisely converted to a finite decimal representation.
For a binary representation sig2 * 2^exp2, the exact decimal conversion 
is sig10 * 10^exp10, where:
    sig10 = (sig2 * 2^exp2) * (10^-exp10)

Following the Schubfach algorithm:
    k = floor(exp2 * log10(2))           // k is exp10
    h = exp2 - k * floor(log2(10))       // h in [0, 3]

For 10^exp10, we precompute a binary cache table in the form cache * 2^cache_exp,
where cache in [0.5, 1.0) (ensuring the MSB is 1), and:
    cache_exp = floor(exp10 * log2(10)) + 1

Thus:
    sig10 = (sig2 * 2^exp2) * (10^-k)
            = (sig2 * 2^exp2) * (cache * 2^(h - exp2 + 1))
            = (sig2 * cache) * 2^(h + 1)
            = (sig2 * cache) << (h + 1)

Both sig2 and cache are represented as 64-bit integers, with sig2 occupying the high 53 bits. 
The exact cache value may be infinitely recurring. 
The multiplication sig2 * cache is illustrated as:

----------------------------------------------------------------------------
cache            |AAAAAAAAAAAAAAAA|BBBBBBBBBBBBBBBB|................
sig2             |   XXXXXXXXXXXXX|
----------------------------------------------------------------------------
|   AXAXAXAXAXAXA|XAXAXAXAXAXAXAXA|
                 |   BXBXBXBXBXBXB|XBXBXBXBXBXBXBXB|
                                  |   .............|................
----------------------------------------------------------------------------
|   -----hi------|-------lo-------|................|    (sig2 * cache << 1/2/3/4)

Since the cache retains only the high 128 bits, all bits beyond '...' are 
truncated. Potential carry from the bit immediately following '...' creates 
critical boundary conditions. 

Focusing on the decimal result, consider this example:

bin_prev: 0x1000000000001D * 2^23
bin:      0x1000000000001E * 2^23
bin_next: 0x1000000000001F * 2^23

ulp = 8.388608 * 10^6
dec_prev: 37778931862957404.979200 * 10^6
dec:      37778931862957413.367808 * 10^6
dec_next: 37778931862957421.756416 * 10^6

            d0             d1 c  u1                            u0    next      
    ----|----|----|----|----|-*--|----|----|----|----|----|----|----|--*-|----
        9    0    1    2    3    4    5    6    7    8    9    0    1    2
          |___________________|___________________|
                             1ulp

The 1ulp falls in [1.0, 10.0), the acceptable range for dec is +/-0.5ulp:
dec - 0.5ulp: 37778931862957409.173504 * 10^6
dec + 0.5ulp: 37778931862957417.562112 * 10^6

Rounding dec to an integer involves these cases:
1. Fractional part < x.5: round down by 1 (as shown)
2. Fractional part > x.5: round up by 1
3. dec - 0.5ulp crosses a multiple of 10: round down by 10 (as shown)
4. dec + 0.5ulp crosses a multiple of 10: round up by 10

Due to approximation errors, critical boundary conditions arise:
1. When the fractional part equals exactly 0.5, 
    the computed 'lo' part equals 0x8000000000000000 with all trailing zeros. 
    We must identify all cases where the approximation approaches this threshold 
    to prevent misrounding.
2. When the midpoint between `dec` and `dec_prev` lies exactly on a multiple of 10, 
    special handling is required.
3. When the midpoint between `dec` and `dec_next` lies exactly on a multiple of 10, 
    special handling is required. 
    We must ensure correct detection without false positives or false negatives.

For performance considerations, the actual algorithm merges the last decimal digit 
of `hi` (4 bits) with the high bits of `lo` (60 bits) into a single 64-bit integer 
when comparing against 0.5ulp. This results in additional precision loss and requires 
more attention.


Results
--------------------
For float64, this code identifies a single exceptional value: 1.3076622631878654e65.
This value enters an incorrect branch in yy_double, but coincidentally produces 
the correct result, so no special handling is required.

For float32, all binary representations can be exhaustively tested for correctness 
within minutes, so verification is not performed here.
"""


# ==============================================================================
# Continued fraction
# ==============================================================================

def mod_inverse(x: int, m: int) -> int:
    """
    Calculate the modular inverse of 'x' under modulo 'm'.
    """
    return pow(x, -1, m)


def calc_continued_fraction(num: int, den: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Calculate the continued fraction representation of a rational number
    using the Euclidean algorithm.
    
    Example:
    Input: num/den = 7/23
    Returns: (a_arr, p_arr, q_arr)
        a0: 0, p0/q0: 0/1
        a1: 3, p1/q1: 1/3
        a2: 3, p2/q2: 3/10
        a3: 2, p3/q3: 7/23
    """
    a_arr, p_arr, q_arr = [], [], []
    h1, h2, k1, k2 = 1, 0, 0, 1
    
    while den:
        q = num // den
        r = num % den
        h = q * h1 + h2
        k = q * k1 + k2
        a_arr.append(q)
        p_arr.append(h)
        q_arr.append(k)
        h2, h1 = h1, h
        k2, k1 = k1, k
        num, den = den, r
    
    return a_arr, p_arr, q_arr


# ==============================================================================
# Gap calculation
# ==============================================================================

class Gap:
    """
    A class to store gap information.
    """
    value = 0 # gap value (length of the gap)
    count = 0 # gap count (number of occurrences)
    def __init__(self, value: int = 0, count: int = 0):
        self.value = value
        self.count = count
    
    def __lt__(self, other: "Gap") -> bool:
        return self.value < other.value
    
    def __eq__(self, other: "Gap") -> bool:
        return self.value == other.value and self.count == other.count
    
    def __repr__(self) -> str:
        return f"Gap(value={self.value}, count={self.count})"
    
    def __str__(self) -> str:
        return f"({self.value}:{self.count})"


def sort_and_merge_gaps(gaps: List[Gap]) -> List[Gap]:
    """
    Normalize gaps:
    - Sort the gaps by their value.
    - Merge gaps with the same value by summing their counts.
    - Remove gaps with non-positive values or counts.
    """
    gaps = [gap for gap in gaps if gap.value > 0 and gap.count > 0]
    gaps.sort()
    
    merged: List[Gap] = []
    for gap in gaps:
        if merged and merged[-1].value == gap.value:
            merged[-1].count += gap.count
        else:
            merged.append(gap)
    return merged


def calc_gaps_fast(p: int, q: int, N: int, sort: bool = True) -> List[Gap]:
    """
    Calculate gaps between consecutive values in the sorted set {0, {px % q}, q}
    where `0 < p < q`, `0 < N < q`, `p` and `q` are coprime, and x in the range [1, N].

    This function uses the Three-gap theorem to calculate the gaps, see:
    https://en.wikipedia.org/wiki/Three-gap_theorem
    https://mathsanew.com/articles/three_distance_theorem.pdf
    https://arxiv.org/abs/1712.03758
    """
    assert 0 < p < q, f"Invalid p({p}) and q({q}), require 0 < p < q"
    assert 0 < N < q, f"Invalid N({N}), require 0 < N < q"
    assert math.gcd(p, q) == 1, f"p({p}) and q({q}) are not coprime"

    # fill continued fraction
    a_arr, p_arr, q_arr = calc_continued_fraction(p, q)
    a_arr.append(0) # let a[-1] = 0
    p_arr.append(1) # let p[-1] = 1
    q_arr.append(0) # let q[-1] = 0
    
    # find t, r, s values
    t = len(a_arr) - 1
    for i in range(len(a_arr)):
        if q_arr[i - 1] + q_arr[i] <= N < q_arr[i] + q_arr[i + 1]:
            t = i
            break
    r = (N - q_arr[t - 1]) // q_arr[t]
    s = (N - q_arr[t - 1]) % q_arr[t]
    
    # calculate u1 and uN
    if t % 2 == 0:  # t is even
        u1 = q_arr[t]
        uN = q_arr[t - 1] + r * q_arr[t]
    else:  # t is odd
        u1 = q_arr[t - 1] + r * q_arr[t]
        uN = q_arr[t]
    
    # calculate gap lengths
    L1 = (u1 * p) % q
    L2 = q - ((uN * p) % q)
    L3 = (L1 + L2) % q

    # calculate counts of each gap length
    count_L1 = N + 1 - u1
    count_L2 = N + 1 - uN
    count_L3 = (u1 + uN - (N + 1)) if s < q_arr[t] - 1 else 0
    
    # fill gaps info
    gaps = [Gap(L1, count_L1), Gap(L2, count_L2), Gap(L3, count_L3)]
    if sort: 
        gaps = sort_and_merge_gaps(gaps)
    return gaps


def calc_gaps_range_fast(p: int, q: int, M: int, N: int, sort: bool = True) -> List[Gap]:
    """
    Calculate gaps between consecutive values in the sorted set {0, {px % q}, q}
    where `0 < p < q`, `0 < N < q`, `p` and `q` are coprime, and x is in the range [M, N].

    The Three-gap theorem states that if you place points on a circle at angles 
    {0a, 1a, 2a, ..., Na}, the circle will be divided into arcs of at most 3 distinct lengths.

    Here, we consider points at angles from `Ma` to `Na`. By shifting the rotation 
    by `Ma`, this is equivalent to the set {-Ma, 0a, 1a, ..., (N-M)a}.

    Using the Three-gap theorem, we can calculate the gaps of {0a, 1a, ..., (N-M)a}. 
    Adding the extra point `-Ma` splits one of these existing gaps, resulting in 
    at most 5 distinct gap lengths.

    If the `sort` is set to `False`, the returned gaps will be:
    { L1, L2, L1+L2, LastGap, FirstGap }.
    """
    assert 0 < p < q, f"Invalid p({p}) and q({q}), require 0 < p < q"
    assert 0 < M < N < q, f"Invalid M({M}) and N({N}), require 0 < M < N < q"
    assert math.gcd(p, q) == 1, f"p({p}) and q({q}) are not coprime"
    inv = mod_inverse(p, q)

    # this y value will split an existing gap in calc_gaps_fast(p, q, M - N)
    y = q - (p * M % q)
    x = y * inv % q
    assert x in range(N - M + 1, q)
    
    N -= M
    cur_y = y
    while True:
        # calculate the higher and lower y values near the current y
        cur_x = cur_y * inv % q
        sub_gaps = calc_gaps_fast(p, q, cur_x, False)
        y_hi = cur_y + sub_gaps[1].value # L2
        y_lo = cur_y - sub_gaps[0].value # L1
        
        # calculate the corresponding x values
        x_hi = y_hi * inv % q
        x_lo = y_lo * inv % q
        x_min = min(x_hi, x_lo)

        # stop if one of the x value is smaller than N
        if y_hi == q:
            cur_y = q
        else:
            cur_y = x_min * p % q
        if x_min <= N:
            break
    
    gaps = calc_gaps_fast(p, q, N, False)
    if cur_y < y: # cur_y is the lower bound of the target gap
        sep_gap = next((gap for gap in gaps if (cur_y + gap.value) * inv % q <= N))
        sep_lo = y - cur_y
        sep_hi = sep_gap.value - sep_lo
    else: # cur_y is the higher bound of the target gap
        sep_gap = next((gap for gap in gaps if (cur_y - gap.value) * inv % q <= N))
        sep_hi = cur_y - y
        sep_lo = sep_gap.value - sep_hi
    
    # split the target gap into two gaps
    sep_gap.count -= 1
    gaps.append(Gap(sep_lo, 1))
    gaps.append(Gap(sep_hi, 1))
    if sort:
        gaps = sort_and_merge_gaps(gaps)
    return gaps


def calc_gaps_naive(p: int, q: int, N: int) -> List[Gap]:
    """
    Calculate gaps between consecutive values in the sorted set {0, {px % q}, q}
    where x is in the range [1, N].
    
    This function generates all values in the set, sorts them, and calculates
    the gaps between consecutive values. It is used as a correctness checker for
    the optimized `calc_gaps_fast()` function.
    
    Example:
    Input p = 7, q = 23, N = 8
        Values: { 0, 23 } + {px % q (x in range 1 to N) }
        Values: { 0,  7, 14, 21,  5, 12, 19,  3, 10, 23 }
        Sorted: { 0,  3,  5,  7, 10, 12, 14, 19, 21, 23 }
        Gaps:   { 3,  2,  2,  3,  2,  2,  5,  2,  2 }
    Returns: [Gap(gap=2, num=6), Gap(gap=3, num=2), Gap(gap=5, num=1)]
    """
    arr = [p * x % q for x in range(1, N + 1)]
    arr.append(0)
    arr.append(q)
    arr.sort()

    set = {}
    for i in range(1, len(arr)):
        gap = arr[i] - arr[i - 1]
        if gap not in set: set[gap] = 0
        set[gap] += 1

    gaps = [Gap(gap, set[gap]) for gap in set]
    return sort_and_merge_gaps(gaps)


def calc_gaps_range_naive(p: int, q: int, M: int, N: int) -> List[Gap]:
    """
    Calculate gaps between consecutive values in the sorted set {0, {px % q}, q}
    where x is in the range [M, N].
    """
    arr = [x * p % q for x in range(M, N + 1)]
    arr.append(0)
    arr.append(q)
    arr.sort()

    set = {}
    for i in range(1, len(arr)):
        gap = arr[i] - arr[i - 1]
        if gap not in set: set[gap] = 0
        set[gap] += 1
    
    gaps = [Gap(gap, set[gap]) for gap in set]
    return sort_and_merge_gaps(gaps)


def test_gaps_calc():
    """
    Validate the calc_gaps function:
        y = px % q (x in the range [1, N])
        gaps between sorted values in the set { 0, y[1], y[2], ..., y[N], q }
    """
    print("--- test calc_gaps ---")
    for Q in range(1, 64):
        for P in range(1, 64):
            for N in range(1, 64):

                # naive method, used for validation
                gaps_naive = calc_gaps_naive(P, Q, N)
                assert 1 <= len(gaps_naive) <= 3, "Gap count error"
                assert sum(gap.value * gap.count for gap in gaps_naive) == Q, "Gap sum error"

                # make p and q coprime
                g = math.gcd(P, Q)
                p = P // g
                q = Q // g
                p = p % q

                # handle special cases
                if p == 0:
                    assert gaps_naive == [Gap(q * g, 1)]
                    continue
                elif N >= q:
                    assert gaps_naive == [Gap(g, q)]
                    continue
                
                # calculate gaps using the three-gap theorem
                gaps_fast = calc_gaps_fast(p, q, N, sort=False)
                assert len(gaps_fast) == 3
                assert math.gcd(gaps_fast[0].value, gaps_fast[1].value) == 1
                if gaps_fast[2].value != 0:
                    assert math.gcd(gaps_fast[0].value, gaps_fast[2].value) == 1
                    assert math.gcd(gaps_fast[1].value, gaps_fast[2].value) == 1
                    assert gaps_fast[0].value + gaps_fast[1].value == gaps_fast[2].value

                # compare with naive method
                gaps_fast = sort_and_merge_gaps(gaps_fast)
                for gap in gaps_fast:
                    gap.value *= g
                assert gaps_fast == gaps_naive, "Gaps not same"

    print("--- done ---")


    """
    Validate the calc_gaps_range function:
        y = px % q (x in the range [M, N])
        gaps between sorted values in the set { 0, y[M], y[M+1], ..., y[N], q }
    """
    print("--- test calc_gaps_range ---")
    for Q in range(1, 64):
        for P in range(1, 64):
            for M in range(1, Q):
                for N in range(M + 1, Q):

                    # naive method, used for validation
                    gaps_naive = calc_gaps_range_naive(P, Q, M, N)
                    assert 1 <= len(gaps_naive) <= 5, "Gap count error"
                    assert sum(gap.value * gap.count for gap in gaps_naive) == Q, "Gap sum error"

                    # make p and q coprime
                    g = math.gcd(P, Q)
                    p = P // g
                    q = Q // g
                    p = p % q

                    # handle special cases
                    if p == 0:
                        # invalid case
                        continue
                    if N >= q:
                        # range out of bounds
                        continue

                    # calculate gaps using the three-gap theorem
                    gaps_fast = calc_gaps_range_fast(p, q, M, N, False)
                    assert len(gaps_fast) == 5

                    # validate the first and last gaps
                    y_arr = sorted([x * P % Q for x in range(M, N + 1)])
                    first_y = y_arr[0]
                    last_y = y_arr[-1]

                    first_gap = gaps_fast[-1].value # sep_hi
                    last_gap = gaps_fast[-2].value # sep_lo
                    assert first_gap * g == first_y
                    assert Q - last_gap * g == last_y

                    inv = mod_inverse(p, q)
                    assert first_gap * inv % q in range(M, N + 1)
                    assert (q - last_gap) * inv % q in range(M, N + 1)

                    # compare with naive method
                    gaps_fast = sort_and_merge_gaps(gaps_fast)
                    for gap in gaps_fast:
                        gap.value *= g
                    assert gaps_fast == gaps_naive, "Gaps not same"

    print("--- done ---")


def calc_mod_mul_count_fast(NUM: int, MOD: int, 
                            X_MIN: int, X_MAX: int, 
                            Y_MIN: int, Y_MAX: int,
                            print_all: bool = False) -> int:
    """
    Given the equation `y = NUM * x % MOD`,
    where x is in the range [x_min, x_max], and y is in the range [y_min, y_max],
    calculate the number of {x, y} pairs that satisfy the equation.
    If the fast method is not available, return -1.

    --------------------------------
    Explanation:
    
    Try find x where:
        Y = (x * NUM) % DEN, Y in range [TOP1, TOP2], x in range [SIG_MIN, SIG_MAX]

    Make num and den coprime:
        y = (x * num) % den, y in range [top1, top2]

    Let inv = mod_inverse(num, den):
        x = (y * inv) % den, y in range [top1, top2]
    
    Now the gaps between sorted set { 0, x1, x2, ..., xN, den } are:
        gaps = calc_gaps_range_fast(inv, den, top1, top2)

    """
    assert 0 < NUM and 0 < MOD
    assert 0 <= X_MIN <= X_MAX
    assert 0 <= Y_MIN <= Y_MAX

    Y_MAX = min(Y_MAX, MOD - 1)

    # use naive method if the range is small
    if X_MAX - X_MIN < 100:
        x_count = 0
        for x in range(X_MIN, X_MAX + 1):
            y = NUM * x % MOD
            if Y_MIN <= y <= Y_MAX:
                x_count += 1
                if print_all:
                    print(f"    x: 0x{x:X}, y: 0x{y:X}")
        return x_count

    gcd = math.gcd(NUM, MOD)
    num = NUM // gcd
    den = MOD // gcd
    num %= den
    inv = mod_inverse(num, den)
    y_min = -(-Y_MIN // gcd)    # ceil()
    y_max = Y_MAX // gcd        # floor()
    # y_max = min(y_max, MOD - 1)

    if num == 0 or den == 1:
        # y is always 0
        if y_min == 0:
            if print_all:
                for x in range(X_MIN, X_MAX + 1):
                    print(f"    x: 0x{x:X}, y: 0x0")
            return X_MAX - X_MIN + 1
        else:
            return 0
    
    if y_min > y_max:
        # no solution
        return 0 
    
    if y_min == y_max:
        # x = x_base + n * DEN
        x_base = y_min * inv % den
        n_min = (X_MIN - x_base + MOD - 1) // MOD if x_base < X_MIN else 0
        n_max = (X_MAX - x_base) // MOD if x_base <= X_MAX else -1
        x_count = n_max - n_min + 1
        if x_count > 0:
            if print_all:
                for n in range(n_min, n_max + 1):
                    x = x_base + n * MOD
                    y = NUM * x % MOD
                    print(f"    x: 0x{x:X}, y: 0x{y:X}")
            return x_count
        else:
            return 0

    # calculate gaps between sorted set { 0, x1, x2, ..., xN, den }
    gaps = calc_gaps_range_fast(inv, den, y_min, y_max, sort=False)
    x1 = gaps[-1].value # first gap between 0 and x1
    assert x1 * NUM % MOD in range(Y_MIN, Y_MAX + 1)

    if x1 > X_MAX:
        # no solution
        return 0
    
    if X_MAX >= den:
        # calculation is a bit complex but rarely encountered, so not handled
        return -1

    # now there must be valid x in the range [X_MIN, X_MAX]
    # check if all valid x has the same gap
    gaps = sort_and_merge_gaps(gaps[:-2]) # remove edge gaps and sort
    gap = gaps[0].value # the smallest gap
    gap_count = (X_MAX - x1) // gap + 1
    xn = x1 + gap * gap_count
    if xn * NUM % MOD in range(Y_MIN, Y_MAX + 1):
        # all valid x has the same gap
        x_min = (X_MIN - x1 + gap - 1) // gap
        x_max = (X_MAX - x1) // gap
        x_count = x_max - x_min + 1
        if print_all:
            for n in range(x_min, x_max + 1):
                x = x1 + n * gap
                y = x * NUM % MOD
                assert y in range(Y_MIN, Y_MAX + 1)
                print(f"    x: 0x{x:X}, y: 0x{y:X}")
        return x_count
    else:
        # there are different gaps between valid x (rare case)
        x_count = 0
        x = x1
        while x <= X_MAX:
            if x >= X_MIN:
                x_count += 1
                if print_all:
                    y = x * NUM % MOD
                    assert y in range(Y_MIN, Y_MAX + 1)
                    print(f"    x: 0x{x:X}, y: 0x{y:X}")
            gap = next((gap.value for gap in gaps if (x + gap.value) * NUM % MOD in range(Y_MIN, Y_MAX + 1)))
            x += gap
        return x_count
    

def calc_mod_mul_count(p: int, q: int, 
                       x_min: int, x_max: int, 
                       y_min: int, y_max: int,
                       print_all: bool = False) -> int:
    """
    Given the equation `y = px % q`, where `0 < p and 0 < q`, 
    x is in the range [x_min, x_max], and y is in the range [y_min, y_max],
    calculate the number of x, y pairs that satisfy the equation.
    """
    assert 0 < p and 0 < q
    assert 0 <= x_min <= x_max
    assert 0 <= y_min <= y_max

    x_count = 0
    for x in range(x_min, x_max + 1):
        y = p * x % q
        if y_min <= y <= y_max:
            x_count += 1
            if print_all:
                print(f"x: 0x{x:X}, y: 0x{y:X}")
    return x_count


# ==============================================================================
# Modular equation count
# ==============================================================================

def count_mod_fast(k: int, r: int, m: int, x_min: int, x_max: int):
    """
    Count the number of integer solutions x in [x_min, x_max] for the linear congruence:
        kx == r (mod m)

    Notes:
    - r can be any integer (including negative). Internally it is reduced modulo m.
    - This covers both forms: (k * x - r) % m == 0 and (k * x) % m == r (for any r).
    - After reduction, if the modulus becomes m == 1, every x in the range is a solution.
    """
    assert k > 0 and m > 0

    # simplify the equation
    g = math.gcd(k, m)
    if r % g != 0:
        return 0  # no solution
    k = k // g
    r = r // g
    m = m // g

    # fast path: modulo 1 means all integers are congruent
    if m == 1:
        return x_max - x_min + 1

    # find a particular solution using mod_inverse
    k_inv = mod_inverse(k, m)
    x0 = (k_inv * r) % m

    # count all solutions in the range [x_min, x_max]
    t_min = (x_min - x0 + m - 1) // m  # ceil()
    t_max = (x_max - x0) // m          # floor()
    return max(0, t_max - t_min + 1)


def test_count_mod():
    print("--- test count_valid_sig ---")
    for k in range(1, 16):
        for r in range(0, 16):
            for m in range(1, 16):
                for x_min in range(0, 40):
                    for x_max in range(x_min, 40):
                        count_fast = count_mod_fast(k, r, m, x_min, x_max)
                        count_naive = sum(1 for x in range(x_min, x_max + 1) if (k * x - r) % m == 0)
                        assert count_fast == count_naive

                        count_fast = count_mod_fast(k, -r, m, x_min, x_max)
                        count_naive = sum(1 for x in range(x_min, x_max + 1) if (k * x + r) % m == 0)
                        assert count_fast == count_naive

                        if r < m:
                            count_fast = count_mod_fast(k, r, m, x_min, x_max)
                            count_naive = sum(1 for x in range(x_min, x_max + 1) if (k * x) % m == r)
                            assert count_fast == count_naive

    print("--- done ---")


# ==============================================================================
# Lookup table for yy_strtod() and yy_dtoa()
# ==============================================================================

POW10_SIG_TABLE_MIN_EXP = -343
POW10_SIG_TABLE_MAX_EXP =  324
POW10_SIG_TABLE_MIN_EXACT_EXP = 0
POW10_SIG_TABLE_MAX_EXACT_EXP = 55

def calc_pow10_u128(p: int) -> int:
    """
    Calculate the power of 10 and return the high 128 bits of the result.
    """
    
    # Calculate 10^p with high precision
    decimal.getcontext().prec = 5000
    sig = Decimal(10) ** p

    # Normalize the sig to range [0.5,1)
    # sig *= Decimal(2) ** -math.floor(p * math.log2(10) + 1)
    while sig < 1:
        sig *= 2
    while sig >= 1:
        sig /= 2

    # Calculate the highest 128 bits of the sig
    all = sig * (2 ** 128)
    top = int(all)
    return top


def is_pow10_u128_exact(p: int) -> bool:
    """
    Check if calc_pow10_u128(p) returns an exact value.
    """
    
    # Calculate 10^p with high precision
    decimal.getcontext().prec = 5000
    sig = Decimal(10) ** p

    # Normalize the sig to range [0.5,1)
    while sig < 1:
        sig *= 2
    while sig >= 1:
        sig /= 2

    # Calculate the highest 128 bits of the sig
    all = sig * (2 ** 128)
    top = int(all)
    return top == all


def print_pow10_u128_table():
    """
    Print the power of 10 table for yy_strtod() and yy_dtoa().
    """
    print(f"#define POW10_SIG_TABLE_MIN_EXP {POW10_SIG_TABLE_MIN_EXP}")
    print(f"#define POW10_SIG_TABLE_MAX_EXP {POW10_SIG_TABLE_MAX_EXP}")
    print(f"#define POW10_SIG_TABLE_MIN_EXACT_EXP {POW10_SIG_TABLE_MIN_EXACT_EXP}")
    print(f"#define POW10_SIG_TABLE_MAX_EXACT_EXP {POW10_SIG_TABLE_MAX_EXACT_EXP}")
    print("static const u64 pow10_sig_table[] = {")
    for p in range(POW10_SIG_TABLE_MIN_EXP, POW10_SIG_TABLE_MAX_EXP + 1):
        is_exact = is_pow10_u128_exact(p)
        assert is_exact == (p in range(POW10_SIG_TABLE_MIN_EXACT_EXP, POW10_SIG_TABLE_MAX_EXACT_EXP + 1))

        c = calc_pow10_u128(p)
        s = f"{c:X}"
        line = f"    U64(0x{s[0:8]}, 0x{s[8:16]}), U64(0x{s[16:24]}, 0x{s[24:32]})"
        if is_exact:
            line += f", /* == 10^{p} */"
        elif p == POW10_SIG_TABLE_MAX_EXP:
            line += f"  /* ~= 10^{p} */"
        else:
            line += f", /* ~= 10^{p} */"
        print(line)
    print("};")


# ==============================================================================
# Float64 encoding and decoding
# ==============================================================================

# float64 number bits
F64_BITS = 64 
# float64 number exponent part bits
F64_EXP_BITS = 11 
# float64 number significand part bits
F64_SIG_BITS = 52
# float64 number significand part bits (with 1 hidden bit)
F64_SIG_FULL_BITS = 53

# maximum binary power of float64 number
F64_MAX_BIN_EXP = 1023
# minimum binary power of float64 number
F64_MIN_BIN_EXP = -1022

F64_SIG_RAW_MIN = 0
F64_SIG_RAW_MAX = (1 << F64_SIG_BITS) - 1 # 0xFFFFFFFFFFFFF
F64_EXP_RAW_MIN = 0                       #        0 (min subnormal)
F64_EXP_RAW_MAX = (1 << F64_EXP_BITS) - 2 # 0x7FE, 2046 (max normal)

SIG_BIN_MIN = (1 << F64_SIG_BITS) | F64_SIG_RAW_MIN # 0x10000000000000
SIG_BIN_MAX = (1 << F64_SIG_BITS) | F64_SIG_RAW_MAX # 0x1FFFFFFFFFFFFF
EXP_BIN_MIN = F64_MIN_BIN_EXP - F64_SIG_BITS # -1022 - 52 = -1074 (subnormal)
EXP_BIN_MAX = F64_MAX_BIN_EXP - F64_SIG_BITS #  1023 - 52 = 971 (max normal)

EXTRA_SHIFT = 6
TRIM_BITS = 64 + EXTRA_SHIFT


# ==============================================================================
# Double to string conversion edge cases
# ==============================================================================

def find_d2s_edge_case_1(e2, e10, h, p10, p10_exact, SIG_MIN, SIG_MAX):

    NUM = p10 << (h + 1)

    if p10_exact and (NUM & ((1 << TRIM_BITS) - 1)) == 0:
        return

    NUM *= 10
    DEN = 1 << (128 + EXTRA_SHIFT)

    BIAS = 6 # Bias used to correctly handle boundary cases.

    # The * 10 in NUM folds digit extraction into the modular product, but the
    # lower TRIM_BITS bits of the original product contribute a carry of up to
    # floor((2**TRIM_BITS - 1) * 10 / 2**TRIM_BITS) = 9 into the digit_frac
    # position. The 9 in TOP2 accounts for this.
    TOP1 = ((0x7FFFFFFFFFFFFFFF - BIAS) << TRIM_BITS)
    TOP2 = ((0x8000000000000009 - BIAS) << TRIM_BITS) | ((1 << TRIM_BITS) - 1)
    
    count = calc_mod_mul_count_fast(NUM, DEN, SIG_MIN, SIG_MAX, TOP1, TOP2, print_all=True)
    assert count >= 0
    if count > 0:
        print(f"found: {count}, e2: {e2}, e10: {e10}")


def find_d2s_edge_case_2(e2, e10, h, p10, p10_exact, SIG_MIN, SIG_MAX):
    # Find: (sig * cache) % den == half_ulp
    if e10 == -1:
        return # special case, no need to check, ulp is 1/2/4/8

    # when e10 > -1, ulp is 2.5, 1.25, 6.25, 3.125, ...
    #   sig is multiple of ulp, so last digit + frac == half_ulp never occurs
    known_count = 0
    if e10 < 0:
        ulp = 2 ** e2
        half_ulp = ulp // 2
        den = 10 ** len(str(ulp))
        known_count = count_mod_fast(ulp, half_ulp, den, SIG_MIN, SIG_MAX)
    
    NUM =  p10 << (h + 1)
    DEN =  1 << (128 + EXTRA_SHIFT)
    TOP1 = (((p10 << h) >> TRIM_BITS) << TRIM_BITS)
    TOP2 = (((p10 << h) >> TRIM_BITS) << TRIM_BITS) | ((1 << TRIM_BITS) - 1)

    count = calc_mod_mul_count_fast(NUM, DEN, SIG_MIN, SIG_MAX, TOP1, TOP2)
    assert count == known_count
    

def find_d2s_edge_case_3(e2, e10, h, p10, p10_exact, SIG_MIN, SIG_MAX):
    # Find: (sig * cache) % den == half_ulp
    if e10 == -1:
        return # special case, no need to check

    known_count = 0
    if e10 < 0:
        ulp = 2 ** e2
        half_ulp = ulp // 2
        den = 10 ** len(str(ulp))
        known_count = count_mod_fast(ulp, -half_ulp, den, SIG_MIN, SIG_MAX)
    
    NUM =  p10 << (h + 1)
    DEN =  1 << (128 + EXTRA_SHIFT)
    scaled_half_ulp = p10 >> (64 + EXTRA_SHIFT - h)
    target = (1 << 64) - scaled_half_ulp - 1
    TOP1 = (target << TRIM_BITS)
    TOP2 = (target << TRIM_BITS) | ((1 << TRIM_BITS) - 1)

    count = calc_mod_mul_count_fast(NUM, DEN, SIG_MIN, SIG_MAX, TOP1, TOP2)
    assert count == known_count
    

def find_d2s_edge_cases():
    """
    Find edge cases for double to string conversion.
    """

    # Generate the list of parameters
    list = []
    for e2 in range(EXP_BIN_MIN, EXP_BIN_MAX + 1):
        k = (e2 * 315653) >> 20
        h = e2 + (((-k - 1) * 217707) >> 16) + EXTRA_SHIFT; # h = [2, 3, 4, 5]

        # Calculate the power of 10
        e10 = -k - 1
        p10 = calc_pow10_u128(e10)

        # The power of 10 is exact or rounded down
        p10_exact = e10 in range(POW10_SIG_TABLE_MIN_EXACT_EXP, POW10_SIG_TABLE_MAX_EXACT_EXP + 1)

        # The range of `sig` values
        SIG_MIN = SIG_BIN_MIN # 0x10000000000000
        SIG_MAX = SIG_BIN_MAX # 0x1FFFFFFFFFFFFF
        if e2 == EXP_BIN_MIN: # subnormal
            SIG_MIN = 1
            SIG_MAX = 0xFFFFFFFFFFFFF

        list.append((e2, e10, h, p10, p10_exact, SIG_MIN, SIG_MAX))

    print("--- Edge case 1 ---")
    for params in list: find_d2s_edge_case_1(*params)

    print("--- Edge case 2 ---")
    for params in list: find_d2s_edge_case_2(*params)
    
    print("--- Edge case 3 ---")
    for params in list: find_d2s_edge_case_3(*params)

    print("==========================================================")
    print("Finished")


# ==============================================================================

if __name__ == "__main__":
    decimal.getcontext().prec = 5000

    # print lookup table for yy_strtod and yy_dtoa:
    # print_pow10_u128_table()

    # test_gaps_calc()
    # test_count_mod()

    find_d2s_edge_cases()
