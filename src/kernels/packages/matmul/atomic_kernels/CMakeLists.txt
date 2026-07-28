# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SVE_FLAG --sve)
set(CPP_KERNELS ${CMAKE_CURRENT_BINARY_DIR}/atomic_cpp_kernels)
set(ASM_KERNELS ${CMAKE_CURRENT_BINARY_DIR}/atomic_asm_kernels)

# Ensure output directories exist
file(MAKE_DIRECTORY ${CPP_KERNELS})
file(MAKE_DIRECTORY ${ASM_KERNELS})

set(ATOMIC_GEMM_CPP_FILES "")
set(ATOMIC_GEMM_NEON_ASM_FILES "")
set(ATOMIC_GEMM_SVE_ASM_FILES "")

foreach(precision s d c z)
  set(CPP_FILE ${CPP_KERNELS}/atomic_kernels_${precision}.cpp)
  set(NEON_ASM_FILE ${ASM_KERNELS}/atomic_neon_${precision}gemm_kernels.S)
  set(SVE_ASM_FILE ${ASM_KERNELS}/atomic_sve_${precision}gemm_kernels.S)

  # Generate C++ file
  add_custom_command(
    OUTPUT ${CPP_FILE}
    COMMAND Python3::Interpreter ${SRC_DIR}/atomic_gen.py ${CPP_KERNELS} --dispatch ${CPP_FILE} --target-os ${TARGET_OS} ${SVE_FLAG} --dtype ${precision}
    DEPENDS ${SRC_DIR}/atomic_gen.py
    COMMENT "Generating ${CPP_FILE}"
    VERBATIM
  )
  list(APPEND ATOMIC_GEMM_CPP_FILES ${CPP_FILE})

  # Generate Assembly files
  add_custom_command(
    OUTPUT ${NEON_ASM_FILE}
    COMMAND Python3::Interpreter ${SRC_DIR}/atomic_gen.py ${ASM_KERNELS} --target-os ${TARGET_OS} --dtype ${precision}
    DEPENDS ${SRC_DIR}/atomic_gen.py
    COMMENT "Generating ${NEON_ASM_FILE}"
    VERBATIM
  )
  add_custom_command(
    OUTPUT ${SVE_ASM_FILE}
    COMMAND Python3::Interpreter ${SRC_DIR}/atomic_gen.py ${ASM_KERNELS} --target-os ${TARGET_OS} ${SVE_FLAG} --dtype ${precision}
    DEPENDS ${SRC_DIR}/atomic_gen.py
    COMMENT "Generating ${SVE_ASM_FILE}"
    VERBATIM
  )

  list(APPEND ATOMIC_GEMM_NEON_ASM_FILES ${NEON_ASM_FILE})
  list(APPEND ATOMIC_GEMM_SVE_ASM_FILES ${SVE_ASM_FILE})
endforeach()

set(GENERATED_FILES ${ATOMIC_GEMM_CPP_FILES} ${ATOMIC_GEMM_NEON_ASM_FILES} ${ATOMIC_GEMM_SVE_ASM_FILES})
# Tell CMake that these files are generated
set_source_files_properties(${GENERATED_FILES} PROPERTIES GENERATED TRUE)
# Ensure all generated files are ready before compilation
add_custom_target(generate_kernel_source_files DEPENDS ${GENERATED_FILES}
  COMMENT "Generating atomic GEMM C++ and assembly sources")

# Create an object library from the generated sources
add_library(atomic_kernels_obj OBJECT)
add_dependencies(atomic_kernels_obj generate_kernel_source_files)

# Kernel headers
target_include_directories(atomic_kernels_obj
    PRIVATE ${INCLUDE_DIR}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Dispatch C++ queries the SVE vector length, requiring SVE compile flags.
kernel_target_sources(atomic_kernels_obj FILES ${ATOMIC_GEMM_CPP_FILES} ISA SVE)
kernel_target_sources(atomic_kernels_obj FILES ${ATOMIC_GEMM_NEON_ASM_FILES} ISA NEON)
kernel_target_sources(atomic_kernels_obj FILES ${ATOMIC_GEMM_SVE_ASM_FILES} ISA SVE)
