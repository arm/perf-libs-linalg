# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(TARGET_OS "${LINALG_TARGET_OS}")

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
# Generator script and shared Mako boilerplate used by compile_mako().
set(MAKO_GENERATOR_DIR "${KERNEL_DIR}/mako_generator")

# Include mako compilation function
list(APPEND CMAKE_MODULE_PATH "${KERNEL_DIR}/cmake")
include(compile_mako)
include(kernel_target_sources)

add_library(kernels_common INTERFACE)
target_link_libraries(kernels_common INTERFACE
    build_options
    linalg_core
)

add_subdirectory("${KERNEL_DIR}/packages/matmul/atomic_kernels")
add_subdirectory("${KERNEL_DIR}/packages/matmul/small_gemm_kernels")
add_subdirectory("${KERNEL_DIR}/packages/matmul/standard_kernels")
add_subdirectory("${KERNEL_DIR}/packages/misc")
add_subdirectory("${KERNEL_DIR}/framework")

set(KERNEL_PROVIDER_TARGETS
    atomic_kernels_obj
    small_gemm_kernels_obj
    standard_kernels_obj
    misc_kernels_obj
    framework_kernels_obj
)

set(KERNEL_INCLUDE_DIRS
    "${KERNEL_DIR}/packages/matmul/atomic_kernels"
    "${KERNEL_DIR}/packages/matmul/small_gemm_kernels"
)

# Gather kernel object providers into a single interface library.
add_library(linalg_kernels INTERFACE)
foreach(kernel_provider_target IN LISTS KERNEL_PROVIDER_TARGETS)
  target_link_libraries(${kernel_provider_target} PUBLIC kernels_common)
  target_link_libraries(linalg_kernels INTERFACE ${kernel_provider_target})
  target_sources(linalg_kernels INTERFACE $<TARGET_OBJECTS:${kernel_provider_target}>)
endforeach()
target_include_directories(linalg_kernels INTERFACE ${KERNEL_INCLUDE_DIRS})
