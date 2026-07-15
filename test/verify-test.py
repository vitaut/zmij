#!/usr/bin/env python3
"""
Sanity tests for verify.py.

These validate count_mod_mul_solutions / enumerate_mod_mul_solutions against a
naive brute-force reference over small inputs, confirm the finite `fractional`
stays within BOUNDARY_WINDOW of the exact value, and spot-check the zmij port
against Python's repr on random doubles.
"""

import importlib.util
import pathlib
import random
import struct

_spec = importlib.util.spec_from_file_location(
    "verify_zmij", pathlib.Path(__file__).with_name("verify.py"))
verify_zmij = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_zmij)

count_mod_mul_solutions = verify_zmij.count_mod_mul_solutions
enumerate_mod_mul_solutions = verify_zmij.enumerate_mod_mul_solutions
SIG_BITS = verify_zmij.SIG_BITS
MASK64 = verify_zmij.MASK64
BOUNDARY_WINDOW = verify_zmij.BOUNDARY_WINDOW
check_value = verify_zmij.check_value
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
    Confirm the finite `fractional` equals floor(2^64 * true_fraction) up to a
    small slack, justifying BOUNDARY_WINDOW. Uses exact rational arithmetic.
    """
    print("fractional error bound ... ", end="", flush=True)
    rng = random.Random(1)
    implicit = 1 << SIG_BITS
    max_slack = 0
    for _ in range(samples):
        raw_exp = rng.randint(1, 2046)
        sig = rng.randint(implicit, (1 << (SIG_BITS + 1)) - 1)
        bin_exp, dec_exp, shift, cache = exp_params(raw_exp)
        s = 70 - shift
        fast = (cache * sig // (1 << s)) & MASK64
        exact = exact_fractional(sig, bin_exp, dec_exp)
        slack = (fast - exact) & MASK64
        slack = min(slack, (1 << 64) - slack)  # distance on the ring
        max_slack = max(max_slack, slack)
    assert max_slack < BOUNDARY_WINDOW, max_slack
    print(f"ok (max slack {max_slack} over {samples:,} samples)")


def test_sample(samples: int = 100000) -> None:
    """Spot-check the zmij port against Python's repr on random doubles."""
    print("sample ... ", end="", flush=True)
    rng = random.Random(2)
    for _ in range(samples):
        bits = rng.getrandbits(64) & ((1 << 63) - 1)  # non-negative finite-ish
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if value != value or value in (float("inf"),) or value == 0.0:
            continue
        raw_exp = (bits >> SIG_BITS) & 0x7FF
        frac = bits & ((1 << SIG_BITS) - 1)
        if raw_exp == 0:
            sig, raw_exp = frac, 1
        else:
            sig = frac | (1 << SIG_BITS)
        ok, got, want = check_value(sig, raw_exp)
        assert ok, (raw_exp, sig, value, got, want)
    print(f"ok ({samples:,} samples)")


if __name__ == "__main__":
    test_count_mod_mul_solutions()
    test_enumerate_mod_mul_solutions()
    test_fractional_error_bound()
    test_sample()
