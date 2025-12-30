set(ZMIJ_VERIFIER_STANDARD cxx_std_20)

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

add_executable(verify-float verify-float.cc)
target_compile_features(verify-float PRIVATE ${ZMIJ_VERIFIER_STANDARD})
target_include_directories(verify-float PRIVATE .)
target_link_libraries(verify-float dragonbox zmij)

add_executable(double-check double-check.cc)
target_compile_features(double-check PRIVATE ${ZMIJ_VERIFIER_STANDARD})
target_include_directories(double-check PRIVATE .)
