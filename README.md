# 🐉 Żmij

A double-to-string conversion algorithm based on [Schubfach](
https://fmt.dev/papers/Schubfach4.pdf) and [yy](
https://github.com/ibireme/c_numconv_benchmark/blob/master/vendor/yy_double)
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
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 5.0507837461e-27);
  puts(buf);
}
```

## Performance

Żmij v1 is more than 4x faster than [Ryū](https://github.com/ulfjack/ryu)
used by multiple C++ standard library implementations, 9x faster than
[double-conversion](https://github.com/google/double-conversion) and ~2.5x
faster than [Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M1.

| Function          | Time (ns) | Speedup |
|-------------------|----------:|--------:|
| ostringstream     | 871.431   | 1.00x   |
| sprintf           | 735.292   | 1.19x   |
| double-conversion | 83.332    | 10.46x  |
| to_chars          | 42.808    | 20.36x  |
| ryu               | 36.809    | 23.67x  |
| schubfach         | 24.721    | 35.25x  |
| fmt               | 22.224    | 39.21x  |
| dragonbox         | 20.532    | 42.44x  |
| yy                | 14.006    | 62.22x  |
| xjb64             | 10.542    | 82.66x  |
| zmij              | 8.661     | 100.62x |
| null              | 0.946     | 921.13x |

**Conversion time (smaller is better):**

<img width="726" height="313" alt="image"
     src="https://github.com/user-attachments/assets/c36c95c6-52a1-42a9-880e-d7621112f7d9" />

`ostringstream` and `sprintf` are excluded due to their significantly slower
performance.

<img width="739" height="605" alt="image"
     src="https://github.com/user-attachments/assets/e6452189-5a4c-4ba2-9e17-f720e263dd5d" />

On EPYC Milan (AMD64) running Linux, Żmij is approximately **2.8× faster than
Ryū** and **5× faster than double-conversion** when compiled with GCC 11.5.

| Function            | Time (ns) | Speedup |
|---------------------|----------:|--------:|
| ostringstream       | 958.889   | 1.00x   |
| sprintf             | 563.022   | 1.70x   |
| double-conversion   | 95.706    | 10.02x  |
| to_chars            | 67.115    | 14.29x  |
| ryu                 | 54.144    | 17.71x  |
| schubfach           | 44.435    | 21.58x  |
| fmt                 | 40.098    | 23.91x  |
| dragonbox           | 30.896    | 31.04x  |
| yy                  | 26.959    | 35.57x  |
| xjb64               | 19.275    | 49.75x  |
| zmij                | 19.194    | 49.96x  |
| null                | 2.766     | 346.72x |

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
  ideas used across scalar and SIMD code paths.

* David Tolnay ([@dtolnay](https://github.com/dtolnay)) - Created and maintains
  the [Rust port of Żmij](https://github.com/dtolnay/zmij), expanding the
  algorithm's reach and adoption in the Rust ecosystem.

* Raffaello Giulietti - Author of the Schubfach algorithm, whose work forms a
  foundational basis for Żmij.

* Yaoyuan Guo ([@ibireme](https://github.com/ibireme)) - Author of the yy
  algorithm, whose ideas influenced key optimizations used in Żmij.

* Cassio Neri ([@cassioneri](https://github.com/cassioneri)) - Proposed the
  single-candidate rounding strategy used in Żmij.

* Junekey Jeon ([@jk-jeon](https://github.com/jk-jeon)) - Author of the
  Dragonbox algorithm, which informed design and benchmarking comparisons for
  Żmij, as well as the `to_decimal` API.

* Community contributors who provided feedback, issues, suggestions, and
  occasional commits, helping improve the robustness and performance of Żmij.
