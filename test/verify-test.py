#!/usr/bin/env python3
"""
Sanity tests for verify.py.

These validate count_mod_mul_solutions / enumerate_mod_mul_solutions against a
naive brute-force reference over small inputs, exercise count_mod_mul_solutions
in the large-integer regime it is actually used in (via oracle-free invariants
and a full-period closed form), confirm the truncated `fractional` stays within
ERROR_MARGIN of the exact value, and spot-check the Żmij port against
Python's repr on random doubles.
"""

import importlib.util
import math
import pathlib
import random
import struct

_spec = importlib.util.spec_from_file_location(
    "verify_zmij", pathlib.Path(__file__).with_name("verify.py"))
verify_zmij = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_zmij)

count_mod_mul_solutions = verify_zmij.count_mod_mul_solutions
enumerate_mod_mul_solutions = verify_zmij.enumerate_mod_mul_solutions
NUM_SIG_BITS = verify_zmij.NUM_SIG_BITS
ERROR_MARGIN = verify_zmij.ERROR_MARGIN
check_value = verify_zmij.check_value
to_decimal = verify_zmij.to_decimal
exp_params = verify_zmij.exp_params
exact_fractional = verify_zmij.exact_fractional


def count_mod_mul_solutions_naive(p: int, q: int,
                                  x_min: int, x_max: int,
                                  y_min: int, y_max: int) -> int:
    """Naive reference for `count_mod_mul_solutions`."""
    assert 0 < p and 0 < q
    assert 0 <= x_min <= x_max
    assert 0 <= y_min <= y_max
    x_count = 0
    for x in range(x_min, x_max + 1):
        y = p * x % q
        if y_min <= y <= y_max:
            x_count += 1
    return x_count


def test_count_mod_mul_solutions() -> None:
    """Validate count_mod_mul_solutions against the naive reference."""
    print("count_mod_mul_solutions ... ", end="", flush=True)
    for mod in range(1, 17):
        for num in range(1, 17):
            for x_min in range(0, 8):
                for x_max in range(x_min, x_min + 8):
                    for y_min in range(0, mod + 2):
                        for y_max in range(y_min, mod + 2):
                            args = (num, mod, x_min, x_max, y_min, y_max)
                            fast = count_mod_mul_solutions(*args)
                            naive = count_mod_mul_solutions_naive(*args)
                            assert fast == naive, (*args, fast, naive)
    print("ok")


def test_count_full_period(trials: int = 10000) -> None:
    """
    Full-period closed form, an exact oracle at arbitrary scale.

    Over x in [0, mod-1] the value num*x % mod hits each multiple of
    g = gcd(num, mod) exactly g times, so the count equals
    g * (#multiples of g in [y_min, y_max]). Exercises inputs far beyond the
    reach of the brute-force reference.
    """
    print("count full period ... ", end="", flush=True)
    rng = random.Random(3)
    for _ in range(trials):
        mod = rng.randint(1, 1 << 60)
        num = rng.randint(1, 1 << 60)
        if rng.random() < 0.5:  # force a nontrivial common factor sometimes
            g = rng.randint(2, 1000)
            num, mod = num * g, mod * g
        y_min = rng.randint(0, mod - 1)
        y_max = rng.randint(y_min, mod - 1 + (1 << 20))
        got = count_mod_mul_solutions(num, mod, 0, mod - 1, y_min, y_max)
        g = math.gcd(num, mod)
        hi = min(y_max, mod - 1)
        want = g * (hi // g - (y_min - 1) // g) if y_min <= hi else 0
        assert got == want, (num, mod, y_min, y_max, got, want)
    print(f"ok ({trials:,} trials)")


def test_count_metamorphic(trials: int = 10000) -> None:
    """
    Oracle-free invariants at large scale: additivity over the x-range and the
    y-interval, plus full coverage of the residues [0, mod-1].
    """
    print("count metamorphic ... ", end="", flush=True)
    rng = random.Random(4)
    for _ in range(trials):
        mod = rng.randint(1, 1 << 60)
        num = rng.randint(1, 1 << 60)
        x_min = rng.randint(0, 1 << 50)
        x_max = x_min + rng.randint(0, 1 << 40)
        y_min = rng.randint(0, mod - 1)
        y_max = rng.randint(y_min, mod - 1)
        whole = count_mod_mul_solutions(num, mod, x_min, x_max, y_min, y_max)

        if x_max > x_min:  # additivity over the x-range
            k = rng.randint(x_min, x_max - 1)
            left = count_mod_mul_solutions(num, mod, x_min, k, y_min, y_max)
            right = count_mod_mul_solutions(num, mod, k + 1, x_max,
                                            y_min, y_max)
            assert whole == left + right, ("x", num, mod, x_min, x_max, k)

        if y_max > y_min:  # additivity over the y-interval
            t = rng.randint(y_min, y_max - 1)
            lo = count_mod_mul_solutions(num, mod, x_min, x_max, y_min, t)
            hi = count_mod_mul_solutions(num, mod, x_min, x_max, t + 1, y_max)
            assert whole == lo + hi, ("y", num, mod, y_min, y_max, t)

        # every x lands in exactly one residue of [0, mod-1]
        cover = count_mod_mul_solutions(num, mod, x_min, x_max, 0, mod - 1)
        assert cover == x_max - x_min + 1, ("cover", num, mod, x_min, x_max)
    print(f"ok ({trials:,} trials)")


def test_enumerate_mod_mul_solutions() -> None:
    """Validate enumerate_mod_mul_solutions against the naive scan."""
    print("enumerate_mod_mul_solutions ... ", end="", flush=True)
    for mod in range(1, 13):
        for num in range(1, 13):
            for x_min in range(0, 6):
                for x_max in range(x_min, x_min + 6):
                    residues = [(x, num * x % mod)
                                for x in range(x_min, x_max + 1)]
                    for y_min in range(0, mod + 2):
                        for y_max in range(y_min, mod + 2):
                            args = (num, mod, x_min, x_max, y_min, y_max)
                            got = list(enumerate_mod_mul_solutions(*args))
                            hi = min(y_max, mod - 1)
                            want = [(x, r) for x, r in residues
                                    if y_min <= r <= hi]
                            assert got == want, (*args, got, want)
    print("ok")


def test_fractional_error_bound(samples: int = 100000) -> None:
    """
    Confirm the truncated `fractional` equals floor(2^64 * true_fraction) up to
    a small slack, justifying ERROR_MARGIN. Uses exact rational arithmetic.
    """
    print("fractional error bound ... ", end="", flush=True)
    rng = random.Random(1)
    implicit = 1 << NUM_SIG_BITS
    mask64 = (1 << 64) - 1
    max_slack = 0
    for _ in range(samples):
        raw_exp = rng.randint(1, 2046)
        bin_sig = rng.randint(implicit, (1 << (NUM_SIG_BITS + 1)) - 1)
        bin_exp, dec_exp, shift, pow10 = exp_params(raw_exp)
        s = 70 - shift
        fast = (pow10 * bin_sig // (1 << s)) & mask64
        exact = exact_fractional(bin_sig, bin_exp, dec_exp)
        slack = (fast - exact) & mask64
        slack = min(slack, (1 << 64) - slack)  # distance on the ring
        max_slack = max(max_slack, slack)
    assert max_slack < ERROR_MARGIN, max_slack
    print(f"ok (max slack {max_slack} over {samples:,} samples)")


def test_sample(samples: int = 100000) -> None:
    """Spot-check the Żmij port against Python's repr on random doubles."""
    print("sample ... ", end="", flush=True)
    rng = random.Random(2)
    for _ in range(samples):
        bits = rng.getrandbits(64) & ((1 << 63) - 1)  # non-negative finite-ish
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if value != value or value in (float("inf"),) or value == 0.0:
            continue
        raw_exp = (bits >> NUM_SIG_BITS) & 0x7FF
        frac = bits & ((1 << NUM_SIG_BITS) - 1)
        if raw_exp == 0:
            bin_sig, raw_exp = frac, 1
        else:
            bin_sig = frac | (1 << NUM_SIG_BITS)
        assert check_value(bin_sig, raw_exp), \
            (raw_exp, bin_sig, value, to_decimal(bin_sig, raw_exp))
    print(f"ok ({samples:,} samples)")


if __name__ == "__main__":
    test_count_mod_mul_solutions()
    test_count_full_period()
    test_count_metamorphic()
    test_enumerate_mod_mul_solutions()
    test_fractional_error_bound()
    test_sample()
