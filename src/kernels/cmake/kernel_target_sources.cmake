# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Resolve ISA tags to the compile options needed for kernel source files.
function(_kernel_resolve_isa_flags out_var)
  set(flags "")

  # Normalize and validate the ISA tags passed after out_var.
  set(isa_tags "")
  set(supported_isa NEON SVE SME2 FP16 BF16)

  foreach(isa IN LISTS ARGN)
    string(TOUPPER "${isa}" isa_tag)
    if(NOT isa_tag IN_LIST supported_isa)
      message(FATAL_ERROR "Unsupported ISA tag '${isa}'")
    endif()
    list(APPEND isa_tags "${isa_tag}")
  endforeach()

  if("SVE" IN_LIST isa_tags AND "SME2" IN_LIST isa_tags)
    message(FATAL_ERROR "Unsupported ISA tag combination: SVE SME2")
  elseif("SME2" IN_LIST isa_tags AND ("FP16" IN_LIST isa_tags OR "BF16" IN_LIST isa_tags))
    message(FATAL_ERROR "Unsupported ISA tag combination: SME2 FP16/BF16")
  elseif("SVE" IN_LIST isa_tags AND "FP16" IN_LIST isa_tags)
    set(flags "-march=armv8.2-a+sve+fp16")
  elseif("SVE" IN_LIST isa_tags AND "BF16" IN_LIST isa_tags)
    set(flags "-march=armv8.2-a+sve+bf16")
  elseif("SME2" IN_LIST isa_tags)
    set(flags "-march=armv9-a+sme2")
  elseif("SVE" IN_LIST isa_tags)
    set(flags "-march=armv8-a+sve")
  elseif("BF16" IN_LIST isa_tags)
    set(flags "-march=armv8.2-a+bf16")
  elseif("FP16" IN_LIST isa_tags)
    set(flags "-march=armv8.2-a+fp16")
  endif()

  set(${out_var} "${flags}" PARENT_SCOPE)
endfunction()

# Add kernel sources to a target from explicit files and/or glob patterns, and
# optionally apply ISA-specific compile options to those sources.
#
# Usage:
#   kernel_target_sources(<target> [FILES ...] [GLOB ...] [ISA ...])
function(kernel_target_sources target)
  cmake_parse_arguments(ARG "" "" "FILES;GLOB;ISA" ${ARGN})

  set(sources ${ARG_FILES})

  foreach(pattern IN LISTS ARG_GLOB)
    if(IS_ABSOLUTE "${pattern}")
      set(glob_pattern "${pattern}")
    else()
      set(glob_pattern "${CMAKE_CURRENT_SOURCE_DIR}/${pattern}")
    endif()

    file(GLOB matched_sources CONFIGURE_DEPENDS "${glob_pattern}")
    list(APPEND sources ${matched_sources})
  endforeach()

  if(NOT sources)
    return()
  endif()

  target_sources(${target} PRIVATE ${sources})

  if(ARG_ISA)
    _kernel_resolve_isa_flags(isa_flags ${ARG_ISA})
    if(isa_flags)
      set_property(SOURCE ${sources} APPEND PROPERTY COMPILE_OPTIONS ${isa_flags})
    endif()
  endif()
endfunction()
