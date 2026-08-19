cmake_minimum_required(VERSION 3.16)
project(zmij CXX C)

add_subdirectory(gtest)

add_library(dragonbox dragonbox/dragonbox_to_chars.cpp)
target_include_directories(dragonbox PUBLIC .)

add_library(fmt fmt/format.cc)
target_compile_options(fmt PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)
target_include_directories(fmt PUBLIC .)
find_package(Threads)
target_link_libraries(fmt Threads::Threads)

# Adds a test that exercises only the public API.
function (add_zmij_api_test name)
  add_executable(${name} zmij-test.cc)
  target_compile_features(${name} PRIVATE ${ZMIJ_STANDARD})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(${name} dragonbox gtest fmt)
  add_test(NAME ${name} COMMAND ${name})
endfunction ()

# Same as add_zmij_api_test but also runs the implementation-detail tests.
function (add_zmij_test name)
  add_zmij_api_test(${name})
  target_sources(${name} PRIVATE zmij-impl-test.cc)
endfunction ()

add_zmij_test(zmij-test)

include(CheckCXXCompilerFlag)
if (ZMIJ_USE_SIMD)
  check_cxx_compiler_flag(-msse4.1 ZMIJ_USE_SIMD_X86)
  if (ZMIJ_USE_SIMD_X86 STREQUAL "41")
    add_zmij_test(zmij-sse4-test)
    target_compile_options(zmij-sse4-test PRIVATE -msse4.1)
    target_compile_definitions(zmij-sse4-test PRIVATE ZMIJ_USE_SIMD_X86=41)
  endif ()
endif ()

add_zmij_test(zmij-no-int128-test)
target_compile_definitions(zmij-no-int128-test PRIVATE ZMIJ_USE_INT128=0)

add_zmij_test(zmij-no-simd-test)
target_compile_definitions(zmij-no-simd-test PRIVATE ZMIJ_USE_SIMD=0)

add_zmij_test(zmij-no-builtins-test)
target_compile_definitions(zmij-no-builtins-test PRIVATE ZMIJ_NO_BUILTINS=1)

add_zmij_test(zmij-optimize-size-test)
target_compile_definitions(zmij-optimize-size-test PRIVATE ZMIJ_OPTIMIZE_SIZE=1)

add_zmij_api_test(zmij-c-test)
target_compile_definitions(zmij-c-test PRIVATE ZMIJ_C=1)
target_sources(zmij-c-test PRIVATE ../zmij.c)

set(ZMIJ_CHECK_STANDARD cxx_std_20)

add_executable(float-check float-check.cc)
target_compile_features(float-check PRIVATE ${ZMIJ_CHECK_STANDARD})
target_link_libraries(float-check fmt dragonbox zmij)

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.1
)

# Make Google Benchmark available if possible: reuse a previously fetched
# source tree, otherwise probe with `git ls-remote` before letting
# FetchContent clone. If unreachable, skip the benchmark target instead of
# breaking configure.
if (EXISTS "${FETCHCONTENT_BASE_DIR}/googlebenchmark-src/CMakeLists.txt")
  FetchContent_MakeAvailable(googlebenchmark)
else ()
  find_package(Git QUIET)
  if (GIT_EXECUTABLE)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" ls-remote --exit-code
              https://github.com/google/benchmark.git v1.9.1
      RESULT_VARIABLE _zmij_result
      OUTPUT_QUIET ERROR_QUIET
      TIMEOUT 5)
    if (_zmij_result EQUAL 0)
      FetchContent_MakeAvailable(googlebenchmark)
    endif ()
  endif ()
endif ()

if (TARGET benchmark::benchmark)
  add_executable(dtoa-benchmark benchmark.cc dtoa-benchmark.cc)
  target_compile_features(dtoa-benchmark PRIVATE cxx_std_20)
  target_link_libraries(dtoa-benchmark fmt dragonbox zmij benchmark::benchmark)

  add_executable(ftoa-benchmark benchmark.cc ftoa-benchmark.cc)
  target_compile_features(ftoa-benchmark PRIVATE cxx_std_20)
  target_link_libraries(ftoa-benchmark fmt dragonbox zmij benchmark::benchmark)
else ()
  message(STATUS
    "Google Benchmark unavailable; skipping benchmark target")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if (ipo_supported)
    set_property(TARGET float-check PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif ()
endif ()
