cmake_minimum_required(VERSION 3.16)
project(zmij CXX)

add_library(zmij zmij.cc zmij.h)
target_compile_features(zmij PRIVATE cxx_std_20)

add_executable(example example.cc)
target_link_libraries(example zmij)

set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
enable_testing()
add_subdirectory(gtest)

add_executable(zmij-test zmij-test.cc)
target_link_libraries(zmij-test gtest zmij)
add_test(NAME zmij-test COMMAND zmij-test)
