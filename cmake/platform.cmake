# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT OR (Apache-2.0 WITH LLVM-exception)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LINALG_TARGET_OS "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(LINALG_TARGET_OS "mac")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LINALG_TARGET_OS "windows")
else()
  message(FATAL_ERROR "Unsupported CMAKE_SYSTEM_NAME='${CMAKE_SYSTEM_NAME}'")
endif()
