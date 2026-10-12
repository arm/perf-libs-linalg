# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

include(CheckCompilerFlag)

# Add supported compiler options for each requested language.
function(linalg_add_compile_options)
  cmake_parse_arguments(PARSE_ARGV 0 flags "REQUIRED" "" "LANGUAGES;OPTIONS;LINK_OPTIONS")
  set(CMAKE_REQUIRED_LINK_OPTIONS ${flags_LINK_OPTIONS})
  if(flags_LINK_OPTIONS)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  endif()

  set(any_supported FALSE)
  foreach(lang IN LISTS flags_LANGUAGES)
    string(SHA256 options_hash "${flags_OPTIONS}\n${flags_LINK_OPTIONS}")
    set(supported "linalg_${lang}_${options_hash}")
    check_compiler_flag(${lang} "${flags_OPTIONS}" ${supported})
    if(${supported})
      set(LINALG_${lang}_COMPILE_OPTIONS
        ${LINALG_${lang}_COMPILE_OPTIONS} ${flags_OPTIONS} PARENT_SCOPE)
      set(any_supported TRUE)
    elseif(flags_REQUIRED)
      message(FATAL_ERROR "${lang} compiler cannot compile/link with ${flags_OPTIONS}")
    endif()
  endforeach()

  if(any_supported AND flags_LINK_OPTIONS)
    set(LINALG_LINK_OPTIONS ${LINALG_LINK_OPTIONS} ${flags_LINK_OPTIONS} PARENT_SCOPE)
  endif()
endfunction()

set(LINALG_C_COMPILE_OPTIONS)
set(LINALG_CXX_COMPILE_OPTIONS)
set(LINALG_Fortran_COMPILE_OPTIONS)
set(LINALG_LINK_OPTIONS)

linalg_add_compile_options(LANGUAGES C CXX OPTIONS -fcx-fortran-rules)
linalg_add_compile_options(LANGUAGES C CXX OPTIONS -ffp-contract=fast)

if(LINALG_SANITIZER STREQUAL "address")
  linalg_add_compile_options(
    LANGUAGES C CXX Fortran
    OPTIONS -fsanitize=address -fno-omit-frame-pointer -g
    LINK_OPTIONS -fsanitize=address
    REQUIRED
  )
endif()
