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

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

add_subdirectory(test)
