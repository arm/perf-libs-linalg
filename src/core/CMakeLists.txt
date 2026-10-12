# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

add_library(linalg_core INTERFACE)
target_include_directories(linalg_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/common"
    "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(linalg_core INTERFACE runtime_common)

if(LINALG_ENABLE_TUNING_OVERRIDES)
  include("${PROJECT_SOURCE_DIR}/cmake/dependencies/json.cmake")
  target_compile_definitions(linalg_core INTERFACE PL_LINALG_LIVE_TARGET=1)
  target_link_libraries(linalg_core INTERFACE nlohmann_json::nlohmann_json)
endif()
