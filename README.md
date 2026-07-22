# 🐉 Żmij

[![CI](https://github.com/vitaut/zmij/actions/workflows/ci.yml/badge.svg)](
https://github.com/vitaut/zmij/actions/workflows/ci.yml)

A double-to-string conversion algorithm based on [Schubfach](
https://fmt.dev/papers/Schubfach4.pdf) and [xjb](
https://github.com/xjb714/xjb/blob/main/xjb.pdf)
with implementations in C and C++

## Features

* Round trip guarantee
* Shortest decimal representation
* Correct rounding
* High [performance](#performance)
* Small binary size
* Fast [compile time](#compile-time)
* IEEE 754 `double` and `float` support
* Safer API than classic `dtoa`
* User-friendly output format similar to Python's default representation
* Negative zero dependencies
* Small, clean codebase consisting of one
  [source file](https://github.com/vitaut/zmij/blob/main/zmij.cc) and an
  optional [header](https://github.com/vitaut/zmij/blob/main/zmij.h)
* Permissive [license](https://github.com/vitaut/zmij/blob/main/LICENSE)

## Usage

```c++
#include "zmij.h"
#include <stdio.h>

int main() {
  char buf[zmij::double_buffer_size + 1];
  auto end = zmij::write(buf, sizeof(buf), 5.0507837461e-27);
  *end = '\0';
  puts(buf);
}
```

## Performance

On an Apple M5 Max running macOS, compiled with Clang 21.0, Żmij is more than
7x faster than [Ryū](https://github.com/ulfjack/ryu), used by multiple C++
standard library implementations, ~13x faster than
[double-conversion](https://github.com/google/double-conversion) and ~100x
faster than `sprintf` on
[dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark).

**Conversion time (smaller is better):**

<a href="https://fmtlib.github.io/dtoa-benchmark/results/apple-m5-max_macos_clang21.0_ab145b9.html">
  <img width="820" height="370" alt="Mean conversion time on Apple M5 Max"
       src="test/charts/apple-m5-max-mean.svg" />
</a>

`ostringstream` and `sprintf` are left out of the charts to keep the faster
methods readable.

<a href="https://fmtlib.github.io/dtoa-benchmark/results/apple-m5-max_macos_clang21.0_ab145b9.html">
  <img width="820" height="650" alt="Time vs. digit count on Apple M5 Max"
       src="test/charts/apple-m5-max-by-digits.svg" />
</a>

On an AMD EPYC 7C13 (Milan) running Linux, compiled with GCC 13.3, Żmij is
~3.8x faster than Ryū, ~7x faster than double-conversion and ~34x faster than
`sprintf`.

**Conversion time (smaller is better):**

<a href="https://fmtlib.github.io/dtoa-benchmark/results/epyc-7c13_linux_gcc13.3_ee50fc8.html">
  <img width="820" height="370" alt="Mean conversion time on AMD EPYC 7C13"
       src="test/charts/epyc-7c13-mean.svg" />
</a>

`ostringstream` and `sprintf` are left out of the charts to keep the faster
methods readable.

<a href="https://fmtlib.github.io/dtoa-benchmark/results/epyc-7c13_linux_gcc13.3_ee50fc8.html">
  <img width="820" height="650" alt="Time vs. digit count on AMD EPYC 7C13"
       src="test/charts/epyc-7c13-by-digits.svg" />
</a>

## Compile time

Compile time is ~135ms by default and ~180ms with optimizations enabled as
measured by

```
% time c++ -c zmij.cc [-O2]
```

taking the best of 3 runs.

## Languages

* C++: https://github.com/vitaut/zmij/blob/main/zmij.cc
  (reference implementation)
* C: https://github.com/vitaut/zmij/blob/main/zmij.c
* Rust: https://github.com/dtolnay/zmij
* Zig: https://github.com/de-sh/zmij

## Differences from Schubfach

* 1 instead of 3 multiplications by a power of 10 in the common case
* Faster logarithm approximations
* Faster division and modulo
* Fewer conditional branches
* More efficient significand and exponent output
* Improved storage of powers of 10
* SIMD support

## Name

Żmij (pronounced roughly zhmeey or more precisely /ʐmij/) is a Polish word that
refers to a mythical dragon- or serpent-like creature, continuing the dragon
theme [started by Steele and White](https://fmt.dev/papers/p372-steele.pdf).

A nice bonus is that the name even contains a "floating point" in its first
letter. And to quote Aras Pranckevičius, "Żmij is also literally a beast."

## Acknowledgements

We would like to express our gratitude to the individuals who have made
**Żmij** possible:

* Victor Zverovich ([@vitaut](https://github.com/vitaut)) - Original author and
  maintainer of Żmij.

* Tobias Schlüter ([@TobiSchluter](https://github.com/TobiSchluter)) -
  Contributed significant performance and portability improvements, including
  SIMD/SSE support and core algorithm refinements that enhance execution speed
  and cross-platform compatibility.

* Dougall Johnson ([@dougallj](https://github.com/dougallj)) – Authored the
  NEON implementation and contributed many optimization ideas, substantially
  improving performance on ARM platforms.

* Alex Guteniev ([@AlexGuteniev](https://github.com/AlexGuteniev)) -
  Contributed multiple fixes and improvements across build systems, platform
  compatibility, and testing infrastructure.

* Xiang JunBo ([@xjb714](https://github.com/xjb714)) - Contributed
  high-performance BCD digit extraction algorithm and additional optimization
  ideas used across scalar and SIMD code paths. The double path uses xjb's
  $10^{-k-1}$ scaling to eliminate a division from the critical path.

* David Tolnay ([@dtolnay](https://github.com/dtolnay)) - Created and maintains
  the [Rust port of Żmij](https://github.com/dtolnay/zmij), expanding the
  algorithm's reach and adoption in the Rust ecosystem.

* Raffaello Giulietti - Author of the Schubfach algorithm, whose work forms a
  foundational basis for Żmij.

* Yaoyuan Guo ([@ibireme](https://github.com/ibireme)) - Author of the yy
  algorithm, whose ideas influenced key optimizations used in Żmij.

* Junekey Jeon ([@jk-jeon](https://github.com/jk-jeon)) - Author of the
  Dragonbox algorithm, which informed design and benchmarking comparisons for
  Żmij, as well as the `to_decimal` API.

* Community contributors who provided feedback, issues, suggestions, and
  occasional commits, helping improve the robustness and performance of Żmij.
