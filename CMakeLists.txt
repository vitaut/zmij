cmake_minimum_required(VERSION 3.16)
project(zmij CXX C)

set(ZMIJ_STANDARD cxx_std_14)

set(ZMIJ_X86_SIMD "auto" CACHE STRING
    "x86 SIMD level: auto, sse2, ssse3, sse41")

set_property(CACHE ZMIJ_X86_SIMD PROPERTY STRINGS
    auto sse2 ssse3 sse41)

if (ZMIJ_X86_SIMD STREQUAL "sse2")
  target_compile_definitions(zmij PRIVATE ZMIJ_USE_SIMD_X86=20)
elseif (ZMIJ_X86_SIMD STREQUAL "ssse3")
  target_compile_definitions(zmij PRIVATE ZMIJ_USE_SIMD_X86=31)
elseif (ZMIJ_X86_SIMD STREQUAL "sse41")
  target_compile_definitions(zmij PRIVATE ZMIJ_USE_SIMD_X86=41)
endif()

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
