# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

add_library(benchmark_common STATIC src/statistics.cpp)
target_include_directories(benchmark_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(benchmark_common PRIVATE build_options)
