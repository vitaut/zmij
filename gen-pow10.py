#!/usr/bin/env python3
# Power of 10 overestimates generator for the Schubfach algorithm:
# https://fmt.dev/papers/Schubfach4.pdf.
# Copyright (c) 2025 - present, Victor Zverovich

import math

# Range of decimal exponents [K_min, K_max] from the paper.
dec_exp_min = -324
dec_exp_max =  292

num_bits = 128

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
    result = result + 1
    hi, lo = result >> 64, (result & (2**64 - 1))
    print(f"{{{hi:#x}, {lo:#018x}}}, // {dec_exp:4}")
