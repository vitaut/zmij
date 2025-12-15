# Żmij

A double-to-string conversion algorithm based on [Schubfach](https://fmt.dev/papers/Schubfach4.pdf)

## Features

* Round trip guarantee
* Shortest decimal representation
* Correct rounding
* High [performance](#performance)
* Fast [compile time](#compile-time)
* Zero dependencies
* Small, clean codebase consisting of one [source file](
  https://github.com/vitaut/zmij/blob/main/zmij.cc) and one [header](https://github.com/vitaut/zmij/blob/main/zmij.h)
* Permissive [license](https://github.com/vitaut/zmij/blob/main/LICENSE)

## Usage

```c++
#include "zmij.h"
#include <stdio.h>

int main() {
  char buf[zmij::buffer_size];
  zmij::dtoa(6.62607015e-34, buf);
  puts(buf);
}
```

## Performance

More than 3x faster than [Ryu](https://github.com/ulfjack/ryu) used by multiple
C++ standard library implementations and ~2x faster than
[Schubfach](https://github.com/vitaut/schubfach)
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark) run on Apple M1.

| Function      | Time (ns) | Speedup |
|---------------|----------:|--------:|
| ostringstream | 888.086   | 1.00x   |
| sprintf       | 735.842   | 1.21x   |
| doubleconv    | 83.281    | 10.66x  |
| to_chars      | 42.965    | 20.67x  |
| ryu           | 37.733    | 23.54x  |
| schubfach     | 25.338    | 35.05x  |
| fmt           | 22.968    | 38.67x  |
| dragonbox     | 20.925    | 42.44x  |
| zmij          | 12.253    | 72.48x  |
| null          | 0.948     | 937.15x |

<img width="787" height="350" alt="image"
     src="https://github.com/user-attachments/assets/91016b7a-fed6-4d5b-a62f-493f7d5d5310" />

<img width="873" height="668" alt="image"
     src="https://github.com/user-attachments/assets/6ad693b9-1a8d-4fe4-ba45-26dafd8e3e13" />

## Compile time

Compile time is ~60ms by default and ~68ms with optimizations enabled as measured by

```
% time c++ -c -std=c++20 zmij.cc [-O2]
```

taking the best of 3 runs.


## Differences from Schubfach

* Selection from 1-3 candidates instead of 2-4
* Fewer integer multiplications in the shorter case
* Faster logarithm approximations
* Faster division and modulo
* Fewer conditional branches
* More efficient significand and exponent output
* Simpler storage of powers of 10 significands
