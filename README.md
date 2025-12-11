# Żmij

A double-to-string conversion algorithm based on [Schubfach](https://fmt.dev/papers/Schubfach4.pdf)

## Features

* Round trip guarantee
* Shortest decimal representation
* Correct rounding
* High [performance](#performance)
* Fast compile time
* Zero dependencies
* Small, clean codebase consisting of one source file and one header
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
on [dtoa-benchmark](https://github.com/fmtlib/dtoa-benchmark).

| Function      | Time (ns) | Speedup |
|---------------|----------:|--------:|
| ostringstream |   902.796 |   1.00x |
| sprintf       |   737.769 |   1.22x |
| doubleconv    |    85.790 |  10.52x |
| to_chars      |    42.779 |  21.10x |
| ryu           |    38.631 |  23.37x |
| schubfach     |    25.292 |  35.69x |
| fmt           |    22.384 |  40.33x |
| dragonbox     |    21.162 |  42.66x |
| zmij          |    12.360 |  73.04x |
| null          |     0.957 | 943.30x |

<img width="772" height="334" alt="image"
     src="https://github.com/user-attachments/assets/64fe6e67-d921-4d96-8702-21fa8de909d9" />

<img width="862" height="655" alt="image"
     src="https://github.com/user-attachments/assets/2efc55cd-d93c-45d5-93a6-fa6846273ade" />

Differences from Schubfach:
* Selection from 1-3 candidates instead of 2-4
* Faster logarithm approximations
* Faster division and modulo
* Fewer conditional branches
* More efficient significand and exponent output
