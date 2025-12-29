# 🐉 Żmij

A double-to-string conversion algorithm based on [Schubfach](https://fmt.dev/papers/Schubfach4.pdf)
and [yy](https://github.com/ibireme/c_numconv_benchmark/blob/master/vendor/yy_double/yy_double.c)
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
* Small, clean codebase consisting of one [source file](
  https://github.com/vitaut/zmij/blob/main/zmij.cc) and one [header](https://github.com/vitaut/zmij/blob/main/zmij.h)
* Permissive [license](https://github.com/vitaut/zmij/blob/main/LICENSE)

## Usage

```c++
#include "zmij.h"
#include <stdio.h>

int main() {
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 6.62607015e-34);
  puts(buf);
}
```

## Performance

More than 3x faster than [Ryu](https://github.com/ulfjack/ryu) used by multiple
C++ standard library implementations and ~2x faster than
[Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M1.

| Function            | Time (ns) | Speedup |
|---------------------|----------:|--------:|
| ostringstream       | 876.371   | 1.00x   |
| sprintf             | 735.924   | 1.19x   |
| double-conversion   | 85.654    | 10.23x  |
| asteria             | 71.738    | 12.22x  |
| to_chars            | 42.857    | 20.45x  |
| ryu                 | 37.821    | 23.17x  |
| schubfach           | 24.809    | 35.32x  |
| fmt                 | 22.316    | 39.27x  |
| dragonbox           | 20.724    | 42.29x  |
| yy                  | 14.095    | 62.18x  |
| zmij                | 10.546    | 83.10x  |
| null                | 0.939     | 933.48x |

<img width="802" height="348" alt="image" src="https://github.com/user-attachments/assets/96858bc0-ef8a-4201-950e-f666b69b83a8" />

<img width="813" height="657" alt="image" src="https://github.com/user-attachments/assets/558fe101-7138-442b-a7ff-7710fd61f2d6" />

## Compile time

Compile time is ~135ms by default and ~155ms with optimizations enabled as measured by

```
% time c++ -c -std=c++20 zmij.cc [-O2]
```

taking the best of 3 runs.

## Languages

* C++: https://github.com/vitaut/zmij/blob/main/zmij.cc (reference implementation)
* C: https://github.com/vitaut/zmij/blob/main/zmij.c
* Rust: https://github.com/dtolnay/zmij
* Zig: https://github.com/de-sh/zmij

## Differences from Schubfach

* 1 instead of 3 multiplications by powers of 10 in the common case
* Faster logarithm approximations
* Faster division and modulo
* Fewer conditional branches
* More efficient significand and exponent output
* Improved storage of powers of 10

## Name

Żmij (pronounced roughly zhmeey or more precisely /ʐmij/) is a Polish word that refers
to a mythical dragon- or serpent-like creature. This continues the dragon theme [started
by Steele and White](https://fmt.dev/papers/p372-steele.pdf). Nice feature of this name
is that it has a floating point in the first letter.
