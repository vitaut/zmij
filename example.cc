#include <stdio.h>

#include "zmij.h"

int main() {
  char buf[zmij::double_buffer_size];
  zmij::write(buf, sizeof(buf), 5.0507837461e-27);
  puts(buf);
}
