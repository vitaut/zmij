cmake_minimum_required(VERSION 3.16)
project(zmij CXX C)

include(GNUInstallDirs)

set(ZMIJ_STANDARD cxx_std_14)

option(ZMIJ_USE_SIMD "Use SIMD instructions" ON)
option(ZMIJ_TEST "Generate the test targets" ON)
option(ZMIJ_EXAMPLE "Generate the example targets" ON)

add_library(zmij zmij.cc zmij.h)
add_library(zmij::zmij ALIAS zmij)
target_include_directories(
  zmij PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(zmij PRIVATE ${ZMIJ_STANDARD})
if (NOT ZMIJ_USE_SIMD)
  target_compile_definitions(zmij PRIVATE "ZMIJ_USE_SIMD=0")
endif ()
if (ZMIJ_EXAMPLE)
  add_executable(example example.cc)
  target_link_libraries(example zmij)
endif ()

if (ZMIJ_TEST)
  set(CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  enable_testing()
  add_subdirectory(test)
endif ()

install(TARGETS zmij EXPORT zmij-targets)
install(FILES zmij.h zmij-to-chars.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT zmij-targets
  FILE zmij-config.cmake
  NAMESPACE zmij::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/zmij)
