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
* Fast [compile time](#compile-time)
* IEEE 754 `double` and `float` support
* Safer API than classic `dtoa`
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

More than 4x faster than [Ryū](https://github.com/ulfjack/ryu) used by multiple
C++ standard library implementations, 9x faster than [double-conversion](
https://github.com/google/double-conversion) and ~2.5x faster than
[Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M1.


| Function           | Time (ns) | Speedup |
|--------------------|----------:|--------:|
| ostringstream      | 870.478   | 1.00x   |
| sprintf            | 734.033   | 1.19x   |
| double-conversion  | 82.903    | 10.50x  |
| to_chars           | 42.537    | 20.46x  |
| ryu                | 36.805    | 23.65x  |
| schubfach          | 24.653    | 35.31x  |
| fmt                | 22.201    | 39.21x  |
| dragonbox          | 20.544    | 42.37x  |
| yy                 | 13.963    | 62.34x  |
| xjb64              | 10.500    | 82.90x  |
| zmij               | 8.895     | 97.87x  |
| null               | 0.929     | 936.55x |

**Conversion time (smaller is better):**

<img width="816" height="358" alt="image"
  src="https://github.com/user-attachments/assets/c6eea19d-f824-4069-bc26-d701a419916e" />

`ostringstream` and `sprintf` are excluded due to their significantly slower
performance.

<img width="857" height="687" alt="image"
  src="https://github.com/user-attachments/assets/13cb86d3-4d76-4903-a13e-d4845a4388b4" />

On EPYC Milan (AMD64) Żmij is ~2.8x faster than Ryū and ~5x faster than double-conversion.

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

<img width="741" height="327" alt="image" src="https://github.com/user-attachments/assets/e7f0ec3f-7317-4c60-b33d-c106215f1ee6" />

<img width="761" height="616" alt="image" src="https://github.com/user-attachments/assets/cece5726-006c-4712-bf09-6ca62c27ef29" />

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
  Żmij.

* Community contributors who provided feedback, issues, suggestions, and
  occasional commits, helping improve the robustness and performance of Żmij.
