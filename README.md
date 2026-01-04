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
  https://github.com/vitaut/zmij/blob/main/zmij.cc) and an optional [header](https://github.com/vitaut/zmij/blob/main/zmij.h)
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

More than 3.5x faster than [Ryu](https://github.com/ulfjack/ryu) used by multiple
C++ standard library implementations and ~2.5x faster than
[Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M1.

| Function            | Time (ns) | Speedup |
|---------------------|----------:|--------:|
| ostringstream       |   874.884 |   1.00× |
| sprintf             |   743.801 |   1.18× |
| double-conversion   |    83.519 |  10.48× |
| to_chars            |    43.672 |  20.03× |
| ryu                 |    36.865 |  23.73× |
| schubfach           |    24.879 |  35.16× |
| fmt                 |    22.338 |  39.17× |
| dragonbox           |    20.641 |  42.39× |
| yy                  |    14.335 |  61.03× |
| xjb64               |    10.724 |  81.58× |
| zmij                |    10.087 |  86.73× |
| null                |     0.930 | 940.73× |

<img width="804" height="350" alt="image" src="https://github.com/user-attachments/assets/389d7e77-1ed2-4988-9521-1f6dbffbc77f" />

<img width="835" height="672" alt="image" src="https://github.com/user-attachments/assets/3d1224d8-1efa-47ee-b5b4-4ed3179bc799" />

## Compile time

Compile time is ~135ms by default and ~155ms with optimizations enabled as measured by

```
% time c++ -c zmij.cc [-O2]
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
