add_subdirectory(gtest)

add_library(dragonbox dragonbox/dragonbox_to_chars.cpp)
target_include_directories(dragonbox PUBLIC .)

function(add_zmij_test name)
  add_executable(${name} zmij-test.cc)
  target_compile_features(${name} PRIVATE ${ZMIJ_STANDARD})
  target_link_libraries(${name} dragonbox gtest)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_zmij_test(zmij-test)

add_zmij_test(zmij-no-int128-test)
target_compile_definitions(zmij-no-int128-test PRIVATE ZMIJ_USE_INT128=0)

add_zmij_test(zmij-no-simd-test)
target_compile_definitions(zmij-no-simd-test PRIVATE ZMIJ_USE_SIMD=0)

add_zmij_test(zmij-no-builtins-test)
target_compile_definitions(zmij-no-simd-test PRIVATE ZMIJ_NO_BUILTINS=1)

add_executable(pow10-test pow10-test.cc)
target_compile_features(pow10-test PRIVATE ${ZMIJ_STANDARD})
add_test(NAME pow10-test COMMAND pow10-test)
target_link_libraries(pow10-test gtest)

set(ZMIJ_CHECK_STANDARD cxx_std_20)

add_library(fmt fmt/format.cc)
if (MSVC)
    target_compile_options(fmt PUBLIC /utf-8)
endif()
target_include_directories(fmt PUBLIC .)

add_executable(float-check float-check.cc)
target_compile_features(float-check PRIVATE ${ZMIJ_CHECK_STANDARD})
target_link_libraries(float-check fmt dragonbox zmij)

add_executable(benchmark benchmark.cc zmij-benchmark.cc)
target_compile_features(benchmark PRIVATE cxx_std_20)
target_link_libraries(benchmark fmt dragonbox zmij)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  add_executable(double-check double-check.cc)
  target_compile_features(double-check PRIVATE ${ZMIJ_CHECK_STANDARD})
  target_link_libraries(double-check fmt)

  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if (ipo_supported)
    set_property(TARGET float-check PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET double-check PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  add_executable(modular-search-test modular-search-test.cc)
  target_compile_features(modular-search-test PRIVATE ${ZMIJ_CHECK_STANDARD})
  add_test(NAME modular-search-test COMMAND modular-search-test)
  target_link_libraries(modular-search-test fmt gtest)
endif ()
