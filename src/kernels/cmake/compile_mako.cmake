# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Generate assembly from Mako templates matched by caller-provided glob patterns.
# Output paths mirror template paths under this package's build directory, so
# templates with the same filename in different subdirectories do not collide.
# The generated source list is returned in ${out_var}.
function(compile_mako out_var)
  cmake_parse_arguments(ARG "" "" "GLOB" ${ARGN})

  set(mako_sources "")
  set(asm_sources "")
  set(asm_root_dir "${CMAKE_CURRENT_BINARY_DIR}/assembly_kernels")

  # Re-run CMake if a matching template is added or removed.
  foreach(pattern IN LISTS ARG_GLOB)
    file(GLOB matched_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${pattern}")
    list(APPEND mako_sources ${matched_sources})
  endforeach()

  if(NOT mako_sources)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  foreach(mako_file IN LISTS mako_sources)
    file(RELATIVE_PATH rel_mako_file "${CMAKE_CURRENT_SOURCE_DIR}" "${mako_file}")

    # Strip only the template suffix: neon/hgemm.S.mako -> neon/hgemm.S.
    string(REGEX REPLACE "\\.mako$" "" asm_rel_file "${rel_mako_file}")

    set(asm_file "${asm_root_dir}/${asm_rel_file}")
    get_filename_component(asm_dir "${asm_file}" DIRECTORY)
    file(MAKE_DIRECTORY "${asm_dir}")

    add_custom_command(
      OUTPUT ${asm_file}
      COMMAND Python3::Interpreter "${MAKO_GENERATOR_DIR}/mako_gen.py" ${mako_file} -o ${asm_file}
      --target-os ${TARGET_OS} --boilerplate-dir "${MAKO_GENERATOR_DIR}"
      DEPENDS ${mako_file} "${MAKO_GENERATOR_DIR}/mako_gen.py" "${MAKO_GENERATOR_DIR}/assembly_boilerplate.py.mako"
      COMMENT "Generating assembly file: ${asm_file} from ${mako_file}"
      VERBATIM
    )

    list(APPEND asm_sources ${asm_file})
  endforeach()

  set(${out_var} ${asm_sources} PARENT_SCOPE)
endfunction()
