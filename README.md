# Żmij

A double-to-string conversion algorithm based on Schubfach

Improvements:
* Faster logarithm approximations
* Fewer branches
* Faster division and modulo
* More efficient digit generation

More than 2x faster than Ryu and 50% faster than [Shubfach](https://github.com/vitaut/schubfach)
on dtoa-benchmark.

| Function       | Time (ns) | Speedup  |
|----------------|----------:|---------:|
| ostringstream  | 875.978   | 1.00x    |
| sprintf        | 746.631   | 1.17x    |
| doubleconv     | 89.011    | 9.84x    |
| to_chars       | 43.916    | 19.95x   |
| ryu            | 37.249    | 23.52x   |
| schubfach      | 24.822    | 35.29x   |
| fmt            | 22.302    | 39.28x   |
| zmij           | 16.648    | 52.62x   |
| dragonbox      | 13.843    | 63.28x   |
| null           | 0.931     | 941.32x  |
