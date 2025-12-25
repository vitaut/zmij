// A verifier for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include <stdint.h>  // uint32_t
#include <stdio.h>   // printf
#include <string.h>  // memcpy

#include "dragonbox/dragonbox_to_chars.cpp"

#include "zmij.h"

int main() {
  char actual[zmij::float_buffer_size] = {};
  char expected[32] = {};
  uint32_t i = 0;
  double percent = 100.0 / (1LL << 32);
  do {
    if (i % 50'000'000 == 0) printf("Progress: %.2f%%\n", i * percent);

    uint32_t bits = i++;
    float value = 0;
    memcpy(&value, &bits, sizeof(float));

    zmij::write(actual, sizeof(actual), value);
    *jkj::dragonbox::to_chars(value, expected) = '\0';

    if (strcmp(actual, expected) == 0) continue;
    if (strcmp(actual, "0") == 0 && strcmp(expected, "0e0") == 0) continue;
    if (strcmp(actual, "-0") == 0 && strcmp(expected, "-0e0") == 0) continue;

    printf("Output mismatch: %s != %s\n", actual, expected);
    return 1;
  } while (i != 0);
  printf("Tested %u values\n", unsigned(i));
}
