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

Żmij v1 is more than 7x faster than [Ryū](https://github.com/ulfjack/ryu)
used by multiple C++ standard library implementations, ~13x faster than
[double-conversion](https://github.com/google/double-conversion) and ~9x
faster than [Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M5
Max.

| Function          | Time (ns) | Speedup |
|-------------------|----------:|--------:|
| zmij              | 3.931     | 120.00x |
| xjb64             | 4.168     | 113.18x |
| yy                | 16.609    | 28.40x  |
| dragonbox         | 20.262    | 23.28x  |
| fmt               | 24.397    | 19.34x  |
| uscale            | 28.830    | 16.36x  |
| ryu               | 30.449    | 15.49x  |
| to_chars          | 34.225    | 13.78x  |
| schubfach         | 34.576    | 13.64x  |
| double-conversion | 50.896    | 9.27x   |
| sprintf           | 399.748   | 1.18x   |
| ostringstream     | 471.702   | 1.00x   |

**Conversion time (smaller is better):**

<img width="726" height="313" alt="image"
     src="https://github.com/user-attachments/assets/c36c95c6-52a1-42a9-880e-d7621112f7d9" />

`ostringstream` and `sprintf` are excluded due to their significantly slower
performance.

<img width="739" height="605" alt="image"
     src="https://github.com/user-attachments/assets/e6452189-5a4c-4ba2-9e17-f720e263dd5d" />

On an AMD EPYC 7C13 (Milan) running Linux, Żmij is approximately **3.8× faster
than Ryū** and **7× faster than double-conversion** when compiled with GCC 13.3.

| Function            | Time (ns) | Speedup |
|---------------------|----------:|--------:|
| zmij                | 14.031    | 50.05x  |
| xjb64               | 15.212    | 46.16x  |
| yy                  | 26.949    | 26.06x  |
| dragonbox           | 30.884    | 22.74x  |
| uscale              | 44.457    | 15.80x  |
| fmt                 | 45.289    | 15.51x  |
| schubfach           | 47.045    | 14.93x  |
| ryu                 | 53.305    | 13.17x  |
| to_chars            | 62.819    | 11.18x  |
| double-conversion   | 101.097   | 6.95x   |
| sprintf             | 477.333   | 1.47x   |
| ostringstream       | 702.277   | 1.00x   |

<img width="741" height="327" alt="image"
     src="https://github.com/user-attachments/assets/e7f0ec3f-7317-4c60-b33d-c106215f1ee6" />

<img width="761" height="616" alt="image"
     src="https://github.com/user-attachments/assets/cece5726-006c-4712-bf09-6ca62c27ef29" />

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
