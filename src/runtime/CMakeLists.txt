# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

add_library(runtime_common INTERFACE)
target_link_libraries(runtime_common INTERFACE build_options)
if(WIN32)
  target_link_libraries(runtime_common INTERFACE advapi32)
endif()
target_include_directories(runtime_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/src/common"
    "${CMAKE_SOURCE_DIR}/include"
)

add_library(linalg_runtime INTERFACE)
target_link_libraries(linalg_runtime INTERFACE runtime_common)

add_library(linalg_runtime_obj OBJECT
    detect/cpu_info.cpp
    detect/cpu_topology.cpp
    detect/dmi.cpp
    detect/micro_arch.cpp
    detect/numa.cpp
    detect/registry.cpp
    detect/sysctl.cpp
    detect/system.cpp
)

target_link_libraries(linalg_runtime_obj PRIVATE runtime_common)

target_sources(linalg_runtime INTERFACE
    $<TARGET_OBJECTS:linalg_runtime_obj>
)
