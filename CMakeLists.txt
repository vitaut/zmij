cmake_minimum_required(VERSION 3.16)
project(zmij CXX)

set(ZMIJ_STANDARD cxx_std_14)

option(ZMIJ_USE_SIMD "Use SIMD instructions" ON)

add_library(zmij zmij.cc zmij.h)
target_include_directories(zmij PUBLIC .)
target_compile_features(zmij PRIVATE ${ZMIJ_STANDARD})
if (NOT ZMIJ_USE_SIMD)
  target_compile_definitions(zmij PRIVATE "ZMIJ_USE_SIMD=0")
endif ()
add_executable(example example.cc)
target_link_libraries(example zmij)

set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
enable_testing()
add_subdirectory(test)
