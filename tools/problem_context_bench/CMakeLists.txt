# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

include("${PROJECT_SOURCE_DIR}/cmake/dependencies/json.cmake")

set(_supported_compute_header
  "${CMAKE_CURRENT_BINARY_DIR}/generated/supported_compute.hpp"
)
set(_supported_compute_script
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/supported_compute.py"
)
add_custom_command(
  OUTPUT "${_supported_compute_header}"
  COMMAND "${Python3_EXECUTABLE}" "${_supported_compute_script}"
    --output "${_supported_compute_header}"
  DEPENDS "${_supported_compute_script}"
  COMMENT "Generating supported problem-context declarations"
  VERBATIM
)

add_executable(problem_context_bench
  src/problem_context_bench.cpp
  src/benchmark_io.cpp
  src/statistics.cpp
  "${_supported_compute_header}"
)
target_include_directories(problem_context_bench PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_link_libraries(problem_context_bench PRIVATE
  pl_linalg
  nlohmann_json::nlohmann_json
)
