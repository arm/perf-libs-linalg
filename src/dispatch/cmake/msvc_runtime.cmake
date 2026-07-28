# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

# Resolve MSVC runtime flags and name for the dispatch sub-build.
function(dispatch_get_msvc_runtime_compile_options out_options out_variant)
  if(NOT CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT)
    set(${out_options} "" PARENT_SCOPE)
    set(${out_variant} "" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(runtime "${CMAKE_MSVC_RUNTIME_LIBRARY}")
  else()
    set(runtime "${CMAKE_MSVC_RUNTIME_LIBRARY_DEFAULT}")
  endif()

  if(runtime STREQUAL "")
    set(${out_options} "" PARENT_SCOPE)
    set(${out_variant} "" PARENT_SCOPE)
    return()
  endif()

  set(config "${CMAKE_BUILD_TYPE}")
  if(NOT config)
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR
        "Dispatch Makefile builds do not support multi-config "
        "MSVC runtime library selection"
      )
    endif()
    set(config "Release")
  endif()

  string(TOLOWER "${config}" config_lower)
  if(config_lower STREQUAL "debug")
    string(REPLACE "$<$<CONFIG:Debug>:Debug>" "Debug" runtime "${runtime}")
  else()
    string(REPLACE "$<$<CONFIG:Debug>:Debug>" "" runtime "${runtime}")
  endif()

  if(runtime MATCHES "\\$<")
    message(FATAL_ERROR
      "Dispatch Makefile builds do not support generator expressions in "
      "CMAKE_MSVC_RUNTIME_LIBRARY: ${runtime}"
    )
  endif()

  set(option_var "CMAKE_CXX_COMPILE_OPTIONS_MSVC_RUNTIME_LIBRARY_${runtime}")
  if(NOT DEFINED ${option_var})
    message(FATAL_ERROR
      "MSVC runtime library '${runtime}' is not supported by the CXX compiler"
    )
  endif()

  string(TOLOWER "${runtime}" variant)
  set(${out_options} "${${option_var}}" PARENT_SCOPE)
  set(${out_variant} "${variant}" PARENT_SCOPE)
endfunction()
