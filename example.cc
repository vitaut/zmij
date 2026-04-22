#include <stdio.h>
#include <stdfloat>

#include "zmij.h"

int main() {
  size_t size = zmij::double_buffer_size;
  char buf[size + 1];
  char* end;
  
  float const f = 1;
  end = zmij::write(buf, sizeof(buf), f);
  *end = '\0';
  puts(buf);
  
  std::float16_t const h = 1;
  end = zmij::write(buf, sizeof(buf), h);
  *end = '\0';
  puts(buf);
}
