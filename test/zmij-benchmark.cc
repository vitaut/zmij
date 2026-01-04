// Benchmark for https://github.com/vitaut/zmij/.
// Copyright (c) 2025 - present, Victor Zverovich
// Distributed under the MIT license (see LICENSE).

#include "benchmark.h"
#include "dragonbox/dragonbox_to_chars.h"
#include "zmij.h"

void dtoa_zmij(double value, char* buffer) {
  zmij::write(buffer, zmij::double_buffer_size, value);
}

REGISTER_METHOD(zmij);

void dtoa_dragonbox(double value, char* buffer) {
  jkj::dragonbox::to_chars(value, buffer, jkj::dragonbox::policy::cache::full);
}

REGISTER_METHOD(dragonbox);
