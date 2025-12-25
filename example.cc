#include <stdio.h>

#include "zmij.h"

int main() {
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 6.62607015e-34);
  puts(buf);
}
