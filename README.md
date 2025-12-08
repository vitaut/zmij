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
| ostringstream  | 892.448   | 1.00x    |
| sprintf        | 751.571   | 1.19x    |
| doubleconv     | 84.961    | 10.50x   |
| to_chars       | 42.921    | 20.79x   |
| ryu            | 37.102    | 24.05x   |
| schubfach      | 25.346    | 35.21x   |
| fmt            | 22.374    | 39.89x   |
| dragonbox      | 20.681    | 43.15x   |
| zmij           | 16.219    | 55.02x   |
| null           | 0.932     | 957.80x  |

<img width="769" height="347" alt="image" src="https://github.com/user-attachments/assets/d3cc3e54-a70e-45a1-997e-ed86181dde87" />
