#include <stdio.h>

#include "zmij.h"

int main() {
  char buf[zmij::double_buffer_size + 1];
  auto size = zmij::write(buf, sizeof(buf), 5.0507837461e-27);
  buf[size] = '\0';
  puts(buf);
}
