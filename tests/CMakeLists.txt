# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

foreach(_test getrfnp)
  add_executable(${_test}_test ${_test}_test.cpp)
  target_compile_features(${_test}_test PRIVATE cxx_std_17)
  target_link_libraries(${_test}_test PRIVATE pl_linalg)

  if(NOT ANDROID)
    add_test(NAME ${_test} COMMAND ${_test}_test)
  endif()
endforeach()
