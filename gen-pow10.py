#!/usr/bin/env python3
# Power of 10 significand generator for Żmij.
# Copyright (c) 2025 - present, Victor Zverovich

import math
import sys

# Range of decimal exponents [K_min, K_max] from the paper.
dec_exp_min = -324
dec_exp_max =  292

num_bits = 128

def get_pow10():
    # Negate dec_pow_min and dec_pow_max because we need negative powers 10^-k.
    for dec_exp in range(-dec_exp_max, -dec_exp_min + 1, 1):
        # dec_exp is -k in the paper.
        bin_exp = math.floor(dec_exp * math.log2(10)) - (num_bits - 1)
        bin_pow = 2**abs(bin_exp)
        dec_pow = 10**abs(dec_exp)
        if dec_exp < 0:
            result = bin_pow // dec_pow
        elif bin_exp < 0:
            result = dec_pow * bin_pow
        else:
            result = dec_pow // bin_pow
        hi, lo = result >> 64, (result & (2**64 - 1))
        print(f"{{{hi:#x}, {lo:#018x}}}, // {dec_exp:4}")

def gen_shifts():
    i = 0
    print('"', end="")
    for raw_exp in range(1 << 11):
        if i % 37 == 0 and i != 0:
            print('"\n"', end="")
        i += 1
        if raw_exp == 0:
            raw_exp = 1
        bin_exp = raw_exp - (52 + 1023)

        # log10_2_sig = round(log10(2) * 2**log10_2_exp)
        log10_2_sig = 315_653
        log10_2_exp = 20
        dec_exp = (bin_exp * log10_2_sig) >> log10_2_exp

        # log2_pow10_sig = round(log2(10) * 2**log2_pow10_exp) + 1
        log2_pow10_sig = 217_707
        log2_pow10_exp = 16
        # pow10_bin_exp = floor(log2(10**-dec_exp))
        pow10_bin_exp = -dec_exp * log2_pow10_sig >> log2_pow10_exp
        exp_shift = bin_exp + pow10_bin_exp + 1

        print(f"\\{exp_shift}", end="")
    print('"')

if sys.argv[1] == 'shifts':
    gen_shifts()
else:
    get_pow10()