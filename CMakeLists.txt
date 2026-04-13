cmake_minimum_required(VERSION 3.16)
project(zmij CXX)

set(ZMIJ_STANDARD cxx_std_14)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads)

add_library(zmij zmij.cc zmij.h)
target_include_directories(zmij PUBLIC .)
target_compile_features(zmij PRIVATE ${ZMIJ_STANDARD})
target_link_libraries(zmij Threads::Threads)

add_executable(example example.cc)
target_link_libraries(example zmij)

set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
enable_testing()
add_subdirectory(test)
